from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .pipeline import export_tutorial_to_pdf
from .storage import default_bilibili_artifact_root, resolve_bilibili_artifact_dir, resolve_bilibili_tutorial_storage

_VIDEO_ID_PATTERN = re.compile(r"\b(BV[0-9A-Za-z_]+)\b", re.IGNORECASE)


class WindsurfSkillBridgeError(RuntimeError):
    pass


def extract_bilibili_video_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = _VIDEO_ID_PATTERN.search(text)
    return str(match.group(1) or "").strip() if match else ""


def _safe_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("name") or item.get("title") or item.get("id") or item.get("value") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_asset_path(value: Any, runtime_artifact_dir: Path | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    if runtime_artifact_dir is not None:
        return str((runtime_artifact_dir / candidate).resolve())
    return raw


def _normalize_transcript_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    aliases = {
        "cc": "manual_subtitles",
        "manual": "manual_subtitles",
        "manual_subtitles": "manual_subtitles",
        "auto": "auto_subtitles",
        "automatic": "auto_subtitles",
        "automatic_subtitles": "auto_subtitles",
        "auto_subtitles": "auto_subtitles",
        "speech_to_text": "whisper",
        "stt": "whisper",
        "whisper": "whisper",
        "visual": "visual_only",
        "visual_only": "visual_only",
    }
    return aliases.get(normalized, normalized)


def _normalize_steps(value: Any, *, fallback_summary: str) -> list[dict[str, Any]]:
    raw_steps = value if isinstance(value, list) else []
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_steps, start=1):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or f"章节{index}").strip() or f"章节{index}"
            content = str(
                item.get("content")
                or item.get("summary")
                or item.get("detail")
                or item.get("text")
                or item.get("value")
                or ""
            ).strip()
            step_index = _safe_int(item.get("index")) or index
        else:
            title = f"章节{index}"
            content = str(item or "").strip()
            step_index = index
        if not content:
            continue
        steps.append(
            {
                "index": step_index,
                "title": title,
                "content": content,
            }
        )
    if steps:
        return steps
    summary = str(fallback_summary or "").strip() or "暂无可用视频总结内容。"
    return [{"index": 1, "title": "视频概览", "content": summary}]


def _normalize_ingredients(value: Any, *, fallback_terms: list[str]) -> list[dict[str, str]]:
    ingredients: list[dict[str, str]] = []
    raw_items = value if isinstance(value, list) else []
    for item in raw_items:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("display") or item.get("title") or "").strip()
            amount = str(item.get("amount") or "").strip()
            display = str(item.get("display") or "").strip() or (f"{name}（{amount}）" if name and amount else name)
        else:
            name = str(item or "").strip()
            amount = ""
            display = name
        if not name:
            continue
        ingredients.append({"name": name, "amount": amount, "display": display or name})
    if ingredients:
        return ingredients
    return [{"name": term, "amount": "", "display": term} for term in fallback_terms if str(term).strip()]


def _normalize_chapters(value: Any) -> list[str]:
    chapters: list[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            text = str(item.get("title") or item.get("name") or item.get("id") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in chapters:
            chapters.append(text)
    return chapters


def normalize_bilibili_tutorial_payload(
    raw_payload: dict[str, Any],
    *,
    video_url: str = "",
    video_id: str = "",
    runtime_artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise WindsurfSkillBridgeError("Windsurf skill result payload must be a dict")

    resolved_video_id = (
        extract_bilibili_video_id(video_id)
        or extract_bilibili_video_id(raw_payload.get("source_id"))
        or extract_bilibili_video_id(raw_payload.get("tutorial_id"))
        or extract_bilibili_video_id(raw_payload.get("source_url"))
        or extract_bilibili_video_id(video_url)
    )
    if not resolved_video_id:
        raise WindsurfSkillBridgeError("无法从 Windsurf skill result.json 中解析 BV 视频 ID")

    resolved_runtime_artifact_dir = Path(runtime_artifact_dir) if runtime_artifact_dir is not None else None
    source_url = str(raw_payload.get("source_url") or raw_payload.get("url") or video_url).strip()
    raw_title = str(raw_payload.get("title") or "").strip()
    recipe_name = str(raw_payload.get("recipe_name") or raw_payload.get("name") or "").strip()
    if not recipe_name and raw_title.endswith("视频总结"):
        recipe_name = raw_title[: -len("视频总结")].strip()
    recipe_name = recipe_name or raw_title or resolved_video_id
    title = raw_title or f"{recipe_name}视频总结"
    summary = str(raw_payload.get("summary") or raw_payload.get("description") or recipe_name).strip() or recipe_name

    key_concepts = _normalize_string_list(raw_payload.get("key_concepts") or raw_payload.get("tags"))
    tags = _normalize_string_list(raw_payload.get("tags") or key_concepts)
    if not key_concepts:
        key_concepts = list(tags)

    transcript_mode = _normalize_transcript_mode(raw_payload.get("transcript_mode") or raw_payload.get("step_source"))
    steps = _normalize_steps(raw_payload.get("steps"), fallback_summary=summary)
    ingredients = _normalize_ingredients(raw_payload.get("ingredients"), fallback_terms=key_concepts[:8] or tags[:8])
    health_goals = _normalize_string_list(raw_payload.get("health_goals"))
    scenarios = _normalize_string_list(raw_payload.get("scenarios")) or ["视频学习", "B站"]
    tips = _normalize_string_list(raw_payload.get("tips"))
    chapters = _normalize_chapters(raw_payload.get("chapters"))
    nutrition_payload = raw_payload.get("nutrition") if isinstance(raw_payload.get("nutrition"), dict) else {}

    normalized = {
        "tutorial_id": f"tutorial_bilibili_{resolved_video_id}",
        "source_type": "bilibili_video",
        "source_id": resolved_video_id,
        "source_url": source_url,
        "platform": "bilibili",
        "recipe_name": recipe_name,
        "title": title,
        "summary": summary,
        "difficulty": str(raw_payload.get("difficulty") or "").strip(),
        "time_minutes": _safe_int(raw_payload.get("time_minutes") or raw_payload.get("duration_minutes")),
        "cuisine": str(raw_payload.get("cuisine") or "").strip(),
        "calories": _safe_int(raw_payload.get("calories")),
        "tags": tags,
        "health_goals": health_goals,
        "scenarios": scenarios,
        "ingredients": ingredients,
        "key_concepts": key_concepts,
        "steps": steps,
        "step_source": transcript_mode or str(raw_payload.get("step_source") or "").strip(),
        "tips": tips,
        "nutrition": {
            "protein": _safe_float(nutrition_payload.get("protein")),
            "carbs": _safe_float(nutrition_payload.get("carbs")),
            "fat": _safe_float(nutrition_payload.get("fat")),
            "fiber": _safe_float(nutrition_payload.get("fiber")),
            "vitamins": _normalize_string_list(nutrition_payload.get("vitamins")),
        },
        "nutrition_highlights": _normalize_string_list(raw_payload.get("nutrition_highlights")),
        "related_recipes": _normalize_string_list(raw_payload.get("related_recipes")),
        "uploader": str(raw_payload.get("uploader") or raw_payload.get("author") or "").strip(),
        "transcript_mode": transcript_mode,
        "cover_path": _normalize_asset_path(raw_payload.get("cover_path"), resolved_runtime_artifact_dir),
        "subtitle_path": _normalize_asset_path(raw_payload.get("subtitle_path"), resolved_runtime_artifact_dir),
        "chapters": chapters,
    }
    return normalized


def expected_windsurf_result_json_path(
    video_url: str,
    *,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> Path:
    resolved_project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    resolved_artifact_root = Path(artifact_root) if artifact_root is not None else default_bilibili_artifact_root(resolved_project_root)
    resolved_video_id = extract_bilibili_video_id(video_url)
    if not resolved_video_id:
        raise WindsurfSkillBridgeError("无法从视频链接中解析 BV 视频 ID")
    return resolve_bilibili_artifact_dir(resolved_artifact_root, resolved_video_id) / "result.json"


def load_bilibili_tutorial_from_windsurf_result(
    *,
    video_url: str,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    result_json_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    resolved_project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    resolved_artifact_root = Path(artifact_root) if artifact_root is not None else default_bilibili_artifact_root(resolved_project_root)

    resolved_video_id = extract_bilibili_video_id(video_url)
    if not resolved_video_id:
        raise WindsurfSkillBridgeError("未提供可识别的 B 站视频链接")

    runtime_artifact_dir = resolve_bilibili_artifact_dir(resolved_artifact_root, resolved_video_id)
    storage = resolve_bilibili_tutorial_storage(
        project_root=resolved_project_root,
        artifact_root=resolved_artifact_root,
        video_id=resolved_video_id,
        tutorial_id=f"tutorial_bilibili_{resolved_video_id}",
        tutorial_title=f"{resolved_video_id}视频总结",
    )

    candidate_paths: list[Path] = []
    if result_json_path:
        candidate_paths.append(Path(result_json_path))
    candidate_paths.extend(
        [
            runtime_artifact_dir / "result.json",
            runtime_artifact_dir / f"tutorial_bilibili_{resolved_video_id}.json",
            storage.tutorial_json_path,
        ]
    )

    unique_candidate_paths: list[Path] = []
    seen_candidates: set[str] = set()
    for candidate in candidate_paths:
        normalized_candidate = str(candidate)
        if normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)
        unique_candidate_paths.append(candidate)

    resolved_result_path: Path | None = None
    for candidate in unique_candidate_paths:
        if candidate.exists():
            resolved_result_path = candidate
            break

    if resolved_result_path is None:
        expected_path = runtime_artifact_dir / "result.json"
        raise WindsurfSkillBridgeError(f"未找到 Windsurf skill 产物 JSON：`{expected_path}`")

    try:
        raw_payload = json.loads(resolved_result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WindsurfSkillBridgeError(f"无法解析 Windsurf skill result.json：{exc}") from exc

    tutorial_payload = normalize_bilibili_tutorial_payload(
        raw_payload,
        video_url=video_url,
        video_id=resolved_video_id,
        runtime_artifact_dir=runtime_artifact_dir,
    )
    storage = resolve_bilibili_tutorial_storage(
        project_root=resolved_project_root,
        artifact_root=resolved_artifact_root,
        video_id=str(tutorial_payload.get("source_id") or resolved_video_id),
        tutorial_id=str(tutorial_payload.get("tutorial_id") or f"tutorial_bilibili_{resolved_video_id}"),
        tutorial_title=str(tutorial_payload.get("title") or f"{resolved_video_id}视频总结"),
    )
    artifacts = {
        "video_id": str(tutorial_payload.get("source_id") or resolved_video_id),
        "artifact_dir": str(storage.artifact_dir),
        "transcript_mode": str(tutorial_payload.get("transcript_mode") or tutorial_payload.get("step_source") or "").strip(),
        "subtitle_path": str(tutorial_payload.get("subtitle_path") or "").strip(),
        "cover_path": str(tutorial_payload.get("cover_path") or "").strip(),
        "result_json_path": str(resolved_result_path),
        "summary_source": "windsurf_skill_result",
    }
    return tutorial_payload, artifacts, storage


def render_bilibili_tutorial_pdf_from_json(
    json_path: str | Path,
    *,
    project_root: str | Path | None = None,
    pdf_output: str | Path | None = None,
) -> Path:
    path = Path(json_path)
    if not path.exists():
        raise WindsurfSkillBridgeError(f"教程 JSON 不存在：`{path}`")

    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WindsurfSkillBridgeError(f"无法解析教程 JSON：{exc}") from exc

    tutorial_payload = normalize_bilibili_tutorial_payload(raw_payload)
    resolved_project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    storage = resolve_bilibili_tutorial_storage(
        project_root=resolved_project_root,
        video_id=str(tutorial_payload.get("source_id") or ""),
        tutorial_id=str(tutorial_payload.get("tutorial_id") or ""),
        tutorial_title=str(tutorial_payload.get("title") or ""),
    )
    resolved_pdf_output = Path(pdf_output) if pdf_output is not None else storage.pdf_path
    return export_tutorial_to_pdf(tutorial_payload, resolved_pdf_output)


def render_bilibili_tutorial_pdf_in_python(
    json_path: str | Path,
    *,
    python_executable: str | Path,
    project_root: str | Path | None = None,
    pdf_output: str | Path | None = None,
) -> Path:
    resolved_project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    script_path = resolved_project_root / "scripts" / "render_bilibili_tutorial_pdf.py"
    if not script_path.exists():
        raise WindsurfSkillBridgeError(f"PDF 渲染脚本不存在：`{script_path}`")

    resolved_json_path = Path(json_path)
    resolved_pdf_output = Path(pdf_output) if pdf_output is not None else None
    command = [str(python_executable), str(script_path), "--json-path", str(resolved_json_path)]
    if resolved_pdf_output is not None:
        command.extend(["--pdf-output", str(resolved_pdf_output)])

    result = subprocess.run(
        command,
        cwd=str(resolved_project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        message = str(result.stderr or result.stdout or "").strip() or "PDF 渲染子进程执行失败"
        raise WindsurfSkillBridgeError(message)
    if resolved_pdf_output is not None:
        return resolved_pdf_output
    stdout_lines = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
    for line in reversed(stdout_lines):
        if line.lower().startswith("pdf written:"):
            return Path(line.split(":", 1)[1].strip())
    raise WindsurfSkillBridgeError("PDF 渲染子进程未返回输出路径")


__all__ = [
    "WindsurfSkillBridgeError",
    "expected_windsurf_result_json_path",
    "extract_bilibili_video_id",
    "load_bilibili_tutorial_from_windsurf_result",
    "normalize_bilibili_tutorial_payload",
    "render_bilibili_tutorial_pdf_from_json",
    "render_bilibili_tutorial_pdf_in_python",
]
