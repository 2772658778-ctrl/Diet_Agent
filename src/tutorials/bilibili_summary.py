from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from ..graph.llm import get_graph_llm
from ..utils.logger import get_logger
from .storage import resolve_bilibili_artifact_dir

logger = get_logger(__name__)

_SUBTITLE_LANG_PRIORITY = ["zh-Hans", "zh-CN", "zh", "ai-zh", "zh-TW", "zh-Hant"]
_SUBTITLE_EXT_PRIORITY = {"srt": 0, "vtt": 1, "srv3": 2, "json3": 3, "json": 4}
_BILIBILI_REFERER = "https://www.bilibili.com/"
_BILIBILI_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
_BILIBILI_NOISE_PHRASES = (
    "一键三连",
    "关注",
    "投币",
    "点赞",
    "收藏",
    "弹幕",
    "评论区",
    "转发",
    "下期再见",
)


class BilibiliSummaryError(RuntimeError):
    pass


class TranscriptChunkSummary(BaseModel):
    section_title: str = Field(description="该片段最合适的教学小节标题")
    teaching_goal: str = Field(description="该片段的教学目标")
    condensed_notes: str = Field(description="基于片段内容整理出的讲解摘要")
    key_points: list[str] = Field(default_factory=list, description="该片段的关键知识点")
    important_examples: list[str] = Field(default_factory=list, description="该片段的重要例子、案例或类比")


class VideoSectionSummary(BaseModel):
    title: str = Field(description="最终讲义中该章节的小标题")
    summary: str = Field(description="该章节的中文教学性总结")
    key_points: list[str] = Field(default_factory=list, description="该章节需要保留的关键点")

    @model_validator(mode="before")
    @classmethod
    def _normalize_summary_aliases(cls, value: Any):
        if not isinstance(value, dict):
            return value
        if value.get("summary"):
            return value
        normalized = dict(value)
        for alias in ("condensed_notes", "section_summary", "notes", "content"):
            alias_value = str(normalized.get(alias) or "").strip()
            if alias_value:
                normalized["summary"] = alias_value
                break
        if "summary" not in normalized:
            normalized["summary"] = ""
        return normalized


class VideoLectureSummary(BaseModel):
    overall_summary: str = Field(description="整支视频的总体总结")
    key_concepts: list[str] = Field(default_factory=list, description="全片核心概念或关键词")
    sections: list[VideoSectionSummary] = Field(default_factory=list, description="按教学顺序重组后的章节")
    takeaways: list[str] = Field(default_factory=list, description="最终应给用户保留的总结与启发")


def _sanitize_filename(value: str) -> str:
    raw = str(value or "bilibili_summary").strip() or "bilibili_summary"
    sanitized = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in raw)
    sanitized = sanitized.rstrip(". ")
    return sanitized or "bilibili_summary"


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _run_yt_dlp(args: list[str], *, cwd: Path | None = None) -> str:
    command = [sys.executable, "-m", "yt_dlp", *args]
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()
    if result.returncode != 0:
        message = stderr or stdout or f"yt-dlp exited with code {result.returncode}"
        if "412" in message and "Precondition Failed" in message:
            raise BilibiliSummaryError(
                "B站返回 HTTP 412，当前抓取被平台风控拦截。请重试时传入 `cookies_from_browser`（例如 chrome），"
                "或先在配置中设置 `BILIBILI_COOKIES_FROM_BROWSER=chrome`。"
            )
        if "Could not copy Chrome cookie database" in message:
            raise BilibiliSummaryError(
                "当前无法直接复制浏览器 cookies 数据库。请先关闭对应浏览器后重试，"
                "或者改用导出的 `cookies.txt` 文件（配置 `BILIBILI_COOKIES_FILE` 或脚本参数 `--cookies-file`）。"
            )
        raise BilibiliSummaryError(message)
    return stdout


def _build_bilibili_request_headers(cookie_header: str = "") -> dict[str, str]:
    headers = {
        "User-Agent": _BILIBILI_USER_AGENT,
        "Referer": _BILIBILI_REFERER,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers


def _load_cookie_header_from_netscape_file(cookies_file: str | Path) -> str:
    path = Path(cookies_file)
    if not path.exists():
        return ""

    cookie_pairs: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        name = str(parts[-2]).strip()
        value = str(parts[-1]).strip()
        if name and value:
            cookie_pairs.append(f"{name}={value}")
    return "; ".join(cookie_pairs)


def _download_url_to_path(url: str, output_path: Path, *, cookie_header: str = "") -> Path:
    request = Request(url, headers=_build_bilibili_request_headers(cookie_header))
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())
    return output_path


def _decode_audio_with_ffmpeg(audio_path: str | Path, ffmpeg_executable: str, sample_rate: int = 16000):
    import numpy as np

    command = [
        ffmpeg_executable,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16).flatten().astype("float32") / 32768.0


def _iter_subtitle_track_candidates(track_map: Any) -> list[dict[str, str]]:
    if not isinstance(track_map, dict):
        return []

    ordered_languages: list[str] = []
    for language in [*_SUBTITLE_LANG_PRIORITY, *list(track_map.keys())]:
        normalized_language = str(language or "").strip()
        if normalized_language and normalized_language in track_map and normalized_language not in ordered_languages:
            ordered_languages.append(normalized_language)

    candidates: list[dict[str, str]] = []
    for language in ordered_languages:
        entries = track_map.get(language) or []
        normalized_entries = [
            entry for entry in entries
            if isinstance(entry, dict) and str(entry.get("url") or "").strip()
        ]
        normalized_entries.sort(
            key=lambda entry: _SUBTITLE_EXT_PRIORITY.get(str(entry.get("ext") or "").lower(), 99)
        )
        for entry in normalized_entries:
            ext = str(entry.get("ext") or "vtt").strip().lower() or "vtt"
            if ext not in _SUBTITLE_EXT_PRIORITY:
                continue
            candidates.append(
                {
                    "language": language,
                    "ext": ext,
                    "url": str(entry.get("url") or "").strip(),
                }
            )
    return candidates


def _extract_audio_track_candidates(format_entries: Any) -> list[dict[str, str]]:
    if not isinstance(format_entries, list):
        return []

    candidates: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for entry in format_entries:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        vcodec = str(entry.get("vcodec") or "").strip().lower()
        acodec = str(entry.get("acodec") or "").strip().lower()
        if vcodec != "none" or not acodec or acodec == "none":
            continue
        ext = str(entry.get("audio_ext") or entry.get("ext") or "m4a").strip().lower() or "m4a"
        candidates.append(
            {
                "url": url,
                "ext": ext,
                "abr": float(entry.get("abr") or entry.get("tbr") or 0.0),
            }
        )
        seen_urls.add(url)

    candidates.sort(key=lambda item: float(item.get("abr") or 0.0), reverse=True)
    return candidates


def _download_cover_from_metadata(metadata: dict[str, Any], output_dir: Path, *, cookie_header: str = "") -> Path | None:
    thumbnail_url = str(metadata.get("thumbnail") or "").strip()
    video_id = str(metadata.get("video_id") or "video")
    if not thumbnail_url:
        return None

    suffix = Path(thumbnail_url.split("?", 1)[0]).suffix or ".jpg"
    cover_path = output_dir / f"{video_id}_cover{suffix}"
    try:
        return _download_url_to_path(thumbnail_url, cover_path, cookie_header=cookie_header)
    except Exception as exc:
        logger.warning(f"direct thumbnail download failed: {exc}")
        return None


def _download_preferred_subtitle_from_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    cookie_header: str = "",
) -> tuple[Path | None, str]:
    video_id = str(metadata.get("video_id") or "video")
    candidate_groups = [
        (metadata.get("manual_subtitle_tracks"), "manual_subtitles"),
        (metadata.get("automatic_subtitle_tracks"), "auto_subtitles"),
    ]
    for track_map, transcript_mode in candidate_groups:
        for candidate in _iter_subtitle_track_candidates(track_map):
            subtitle_path = output_dir / f"{video_id}.{candidate['language']}.{candidate['ext']}"
            try:
                downloaded_path = _download_url_to_path(candidate["url"], subtitle_path, cookie_header=cookie_header)
                return downloaded_path, transcript_mode
            except Exception as exc:
                logger.warning(
                    f"direct subtitle download failed for {candidate['language']} ({candidate['ext']}): {exc}"
                )
    return None, "none"


def _download_preferred_audio_from_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    cookie_header: str = "",
) -> Path | None:
    video_id = str(metadata.get("video_id") or "video")
    for candidate in list(metadata.get("audio_track_candidates") or []):
        ext = str(candidate.get("ext") or "m4a").strip().lower() or "m4a"
        audio_path = output_dir / f"{video_id}_audio_direct.{ext}"
        try:
            return _download_url_to_path(str(candidate.get("url") or ""), audio_path, cookie_header=cookie_header)
        except Exception as exc:
            logger.warning(f"direct audio download failed for {ext}: {exc}")
    return None


def _extract_primary_info(info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(info, dict):
        raise BilibiliSummaryError("invalid yt-dlp metadata payload")

    entries = info.get("entries") or []
    if isinstance(entries, list) and entries:
        raise BilibiliSummaryError(
            "检测到该 B 站链接可能是分P/多段视频，请先明确要处理的分P范围，再执行总结。"
        )

    video_id = str(info.get("id") or info.get("bvid") or "").strip()
    title = str(info.get("title") or video_id or "未命名视频").strip() or "未命名视频"
    uploader = str(info.get("uploader") or info.get("channel") or info.get("creator") or "").strip()
    description = str(info.get("description") or "").strip()
    webpage_url = str(info.get("webpage_url") or info.get("original_url") or "").strip()
    duration_seconds = int(info.get("duration") or 0)
    thumbnail = str(info.get("thumbnail") or "").strip()
    tags = [str(item).strip() for item in list(info.get("tags") or []) if str(item).strip()]
    chapters = []
    for item in list(info.get("chapters") or []):
        if not isinstance(item, dict):
            continue
        chapter_title = str(item.get("title") or "").strip()
        if chapter_title:
            chapters.append(chapter_title)

    manual_subtitle_tracks = info.get("subtitles") if isinstance(info.get("subtitles"), dict) else {}
    automatic_subtitle_tracks = info.get("automatic_captions") if isinstance(info.get("automatic_captions"), dict) else {}
    manual_subtitles = list(manual_subtitle_tracks.keys())
    automatic_subtitles = list(automatic_subtitle_tracks.keys())
    audio_track_candidates = _extract_audio_track_candidates(
        [
            *list(info.get("requested_formats") or []),
            *list(info.get("formats") or []),
        ]
    )

    return {
        "video_id": video_id,
        "title": title,
        "uploader": uploader,
        "description": description,
        "webpage_url": webpage_url,
        "duration_seconds": duration_seconds,
        "thumbnail": thumbnail,
        "tags": tags,
        "chapters": chapters,
        "manual_subtitles": manual_subtitles,
        "automatic_subtitles": automatic_subtitles,
        "manual_subtitle_tracks": manual_subtitle_tracks,
        "automatic_subtitle_tracks": automatic_subtitle_tracks,
        "audio_track_candidates": audio_track_candidates,
    }


def inspect_bilibili_metadata(url: str, *, cookies_from_browser: str = "", cookies_file: str | Path = "") -> dict[str, Any]:
    args = [
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--add-header",
        "Referer:https://www.bilibili.com/",
    ]
    if cookies_file:
        args.extend(["--cookies", str(cookies_file)])
    elif cookies_from_browser:
        args.extend(["--cookies-from-browser", cookies_from_browser])
    args.append(url)
    stdout = _run_yt_dlp(args)
    try:
        raw_info = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BilibiliSummaryError(f"无法解析 yt-dlp 元数据输出: {exc}") from exc
    return _extract_primary_info(raw_info)


def _subtitle_score(path: Path) -> tuple[int, int, str]:
    name = path.name
    for index, language in enumerate(_SUBTITLE_LANG_PRIORITY):
        if language in name:
            return (0, index, name)
    if ".srt" in name:
        return (1, 0, name)
    return (2, 0, name)


def _find_subtitle_file(artifact_dir: Path, video_id: str) -> Path | None:
    candidates = sorted(
        [
            *artifact_dir.glob(f"{video_id}*.srt"),
            *artifact_dir.glob(f"{video_id}*.vtt"),
            *artifact_dir.glob(f"{video_id}*.srv3"),
            *artifact_dir.glob(f"{video_id}*.json3"),
        ],
        key=_subtitle_score,
    )
    return candidates[0] if candidates else None


def _find_cover_file(artifact_dir: Path, video_id: str) -> Path | None:
    candidates = [
        *artifact_dir.glob(f"{video_id}*.jpg"),
        *artifact_dir.glob(f"{video_id}*.jpeg"),
        *artifact_dir.glob(f"{video_id}*.png"),
        *artifact_dir.glob(f"{video_id}*.webp"),
    ]
    return sorted(candidates)[0] if candidates else None


def download_bilibili_artifacts(
    url: str,
    *,
    artifact_dir: str | Path,
    cookies_from_browser: str = "",
    cookies_file: str | Path = "",
) -> dict[str, Any]:
    metadata = inspect_bilibili_metadata(url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file)
    output_dir = _ensure_dir(resolve_bilibili_artifact_dir(artifact_dir, str(metadata.get("video_id") or "")))
    cookie_header = _load_cookie_header_from_netscape_file(cookies_file) if cookies_file else ""
    output_template = str(output_dir / "%(id)s.%(ext)s")
    args = [
        "--skip-download",
        "--no-warnings",
        "--add-header",
        "Referer:https://www.bilibili.com/",
        "--write-thumbnail",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        ",".join(_SUBTITLE_LANG_PRIORITY),
        "--convert-subs",
        "srt",
        "-o",
        output_template,
    ]
    if cookies_file:
        args.extend(["--cookies", str(cookies_file)])
    elif cookies_from_browser:
        args.extend(["--cookies-from-browser", cookies_from_browser])
    args.append(url)
    try:
        _run_yt_dlp(args, cwd=output_dir)
    except BilibiliSummaryError as exc:
        if not cookies_file:
            raise
        logger.warning(f"yt-dlp artifact download failed, falling back to direct fetch: {exc}")
        subtitle_path, transcript_mode = _download_preferred_subtitle_from_metadata(
            metadata,
            output_dir,
            cookie_header=cookie_header,
        )
        cover_path = _download_cover_from_metadata(metadata, output_dir, cookie_header=cookie_header)
        return {
            **metadata,
            "artifact_dir": str(output_dir),
            "subtitle_path": str(subtitle_path) if subtitle_path else "",
            "cover_path": str(cover_path) if cover_path else "",
            "transcript_mode": transcript_mode,
        }

    subtitle_path = _find_subtitle_file(output_dir, metadata["video_id"])
    cover_path = _find_cover_file(output_dir, metadata["video_id"])
    transcript_mode = "manual_subtitles" if metadata["manual_subtitles"] else "auto_subtitles" if metadata["automatic_subtitles"] else "none"
    return {
        **metadata,
        "artifact_dir": str(output_dir),
        "subtitle_path": str(subtitle_path) if subtitle_path else "",
        "cover_path": str(cover_path) if cover_path else "",
        "transcript_mode": transcript_mode,
    }


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(int(seconds * 1000), 0)
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1_000
    milliseconds = total_milliseconds % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _write_whisper_srt(segments: list[dict[str, Any]], output_path: Path) -> Path:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_format_srt_timestamp(float(segment.get('start') or 0.0))} --> {_format_srt_timestamp(float(segment.get('end') or 0.0))}",
                text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _transcribe_with_whisper(
    url: str,
    *,
    artifact_dir: Path,
    metadata: dict[str, Any] | None = None,
    cookies_from_browser: str = "",
    cookies_file: str | Path = "",
    whisper_model: str = "base",
) -> Path | None:
    if not _has_module("whisper"):
        return None

    audio_template = str(artifact_dir / "%(id)s_audio.%(ext)s")
    args = [
        "-x",
        "--audio-format",
        "wav",
        "--no-warnings",
        "--add-header",
        "Referer:https://www.bilibili.com/",
        "-o",
        audio_template,
    ]
    if cookies_file:
        args.extend(["--cookies", str(cookies_file)])
    elif cookies_from_browser:
        args.extend(["--cookies-from-browser", cookies_from_browser])
    args.append(url)
    cookie_header = _load_cookie_header_from_netscape_file(cookies_file) if cookies_file else ""
    try:
        _run_yt_dlp(args, cwd=artifact_dir)
        audio_candidates = sorted(artifact_dir.glob("*_audio.wav"))
    except BilibiliSummaryError as exc:
        if not cookies_file or metadata is None:
            raise
        logger.warning(f"yt-dlp audio download failed, falling back to direct audio fetch: {exc}")
        direct_audio_path = _download_preferred_audio_from_metadata(
            metadata,
            artifact_dir,
            cookie_header=cookie_header,
        )
        audio_candidates = [direct_audio_path] if direct_audio_path else []

    if not audio_candidates:
        return None

    import whisper

    ffmpeg_executable = ""
    try:
        if _has_module("imageio_ffmpeg"):
            try:
                import imageio_ffmpeg

                ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                logger.warning(f"failed to prepare bundled ffmpeg from imageio-ffmpeg: {exc}")

        model = whisper.load_model(whisper_model)
        audio_input: Any = str(audio_candidates[0])
        if ffmpeg_executable:
            audio_input = _decode_audio_with_ffmpeg(audio_candidates[0], ffmpeg_executable)
        result = model.transcribe(audio_input, language="zh", verbose=False)
    except FileNotFoundError as exc:
        raise BilibiliSummaryError(
            "Whisper 转写需要可用的 ffmpeg。请安装系统 ffmpeg，或在当前环境中安装 `imageio-ffmpeg` 以使用内置二进制。"
        ) from exc

    srt_path = artifact_dir / f"{audio_candidates[0].stem}.whisper.srt"
    return _write_whisper_srt(list(result.get("segments") or []), srt_path)


def ensure_bilibili_transcript(
    url: str,
    *,
    artifact_dir: str | Path,
    cookies_from_browser: str = "",
    cookies_file: str | Path = "",
    whisper_model: str = "base",
) -> dict[str, Any]:
    artifacts = download_bilibili_artifacts(
        url,
        artifact_dir=artifact_dir,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    subtitle_path = Path(artifacts["subtitle_path"]) if artifacts.get("subtitle_path") else None
    if subtitle_path and subtitle_path.exists():
        return artifacts

    existing_subtitle = _find_subtitle_file(Path(artifacts["artifact_dir"]), str(artifacts.get("video_id") or ""))
    if existing_subtitle and existing_subtitle.exists():
        artifacts["subtitle_path"] = str(existing_subtitle)
        if "whisper" in existing_subtitle.name.lower():
            artifacts["transcript_mode"] = "whisper"
        return artifacts

    whisper_srt = _transcribe_with_whisper(
        url,
        artifact_dir=Path(artifacts["artifact_dir"]),
        metadata=artifacts,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        whisper_model=whisper_model,
    )
    if whisper_srt is not None and whisper_srt.exists():
        artifacts["subtitle_path"] = str(whisper_srt)
        artifacts["transcript_mode"] = "whisper"
        return artifacts

    raise BilibiliSummaryError(
        "未获取到可用字幕。当前实现已尝试 CC/自动字幕；若视频仍无字幕，请在环境中安装 openai-whisper 与 ffmpeg，以启用语音转写回退。"
    )


def _clean_transcript_line(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return ""
    if any(phrase in normalized and len(normalized) <= 24 for phrase in _BILIBILI_NOISE_PHRASES):
        return ""
    return normalized


def load_subtitle_entries(subtitle_path: str | Path) -> list[dict[str, str]]:
    path = Path(subtitle_path)
    content = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if path.suffix.lower() in {".json", ".json3"} or content.lstrip().startswith("{"):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            body_entries = payload.get("body") or []
            if isinstance(body_entries, list):
                entries: list[dict[str, str]] = []
                for item in body_entries:
                    if not isinstance(item, dict):
                        continue
                    cleaned_text = _clean_transcript_line(str(item.get("content") or ""))
                    if not cleaned_text:
                        continue
                    start_value = float(item.get("from") or item.get("start") or 0.0)
                    end_value = float(item.get("to") or item.get("end") or start_value)
                    entries.append(
                        {
                            "start": _format_srt_timestamp(start_value),
                            "end": _format_srt_timestamp(end_value),
                            "text": cleaned_text,
                        }
                    )
                if entries:
                    return entries
            event_entries = payload.get("events") or []
            if isinstance(event_entries, list):
                entries = []
                for event in event_entries:
                    if not isinstance(event, dict):
                        continue
                    segments = event.get("segs") or []
                    text = "".join(
                        str(segment.get("utf8") or "")
                        for segment in segments
                        if isinstance(segment, dict)
                    )
                    cleaned_text = _clean_transcript_line(text)
                    if not cleaned_text:
                        continue
                    start_ms = float(event.get("tStartMs") or 0.0)
                    duration_ms = float(event.get("dDurationMs") or 0.0)
                    entries.append(
                        {
                            "start": _format_srt_timestamp(start_ms / 1000.0),
                            "end": _format_srt_timestamp((start_ms + duration_ms) / 1000.0),
                            "text": cleaned_text,
                        }
                    )
                if entries:
                    return entries
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    entries: list[dict[str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        if "-->" in lines[0]:
            time_line = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 3 and "-->" in lines[1]:
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            continue
        cleaned_text = _clean_transcript_line(" ".join(text_lines))
        if not cleaned_text:
            continue
        start_time, _, end_time = time_line.partition("-->")
        entries.append(
            {
                "start": start_time.strip(),
                "end": end_time.strip(),
                "text": cleaned_text,
            }
        )
    return entries


def _chunk_subtitle_entries(
    entries: list[dict[str, str]],
    *,
    max_chars: int = 3200,
    max_entries: int = 120,
) -> list[list[dict[str, str]]]:
    if not entries:
        return []

    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_chars = 0
    for entry in entries:
        entry_chars = len(entry.get("text") or "")
        if current and (len(current) >= max_entries or current_chars + entry_chars > max_chars):
            chunks.append(current)
            overlap = current[-5:] if len(current) > 5 else current[-2:]
            current = list(overlap)
            current_chars = sum(len(item.get("text") or "") for item in current)
        current.append(entry)
        current_chars += entry_chars
    if current:
        chunks.append(current)
    return chunks


def _serialize_metadata_context(metadata: dict[str, Any]) -> str:
    payload = {
        "title": metadata.get("title"),
        "uploader": metadata.get("uploader"),
        "duration_seconds": metadata.get("duration_seconds"),
        "description": metadata.get("description"),
        "tags": metadata.get("tags") or [],
        "chapters": metadata.get("chapters") or [],
        "transcript_mode": metadata.get("transcript_mode"),
        "webpage_url": metadata.get("webpage_url"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def summarize_transcript_chunk(metadata: dict[str, Any], entries: list[dict[str, str]], *, chunk_index: int) -> TranscriptChunkSummary:
    llm = get_graph_llm()
    transcript_text = "\n".join(f"[{item['start']}] {item['text']}" for item in entries)
    system_prompt = (
        "你是一名教学型视频讲义整理助手。请基于 B 站视频的真实教学内容，总结当前片段的教学目标、关键点和例子。"
        "不要按字幕机械复述，不要保留‘一键三连/关注投币/下期再见’等平台话术。"
        "输出应该便于后续整合成结构化中文讲义。"
    )
    human_prompt = (
        f"视频元数据：\n{_serialize_metadata_context(metadata)}\n\n"
        f"当前片段编号：{chunk_index}\n"
        f"当前片段字幕：\n{transcript_text}"
    )
    return llm.with_structured_output(TranscriptChunkSummary).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )


def _fallback_integrate_video_summary(
    metadata: dict[str, Any],
    chunk_summaries: list[TranscriptChunkSummary],
) -> VideoLectureSummary:
    section_order: list[str] = []
    section_buckets: dict[str, dict[str, list[str]]] = {}
    concept_candidates: list[str] = []
    takeaway_candidates: list[str] = []

    for index, summary in enumerate(chunk_summaries, start=1):
        title = str(summary.section_title or f"片段 {index}").strip() or f"片段 {index}"
        if title not in section_buckets:
            section_order.append(title)
            section_buckets[title] = {"notes": [], "key_points": []}

        note = str(summary.condensed_notes or "").strip()
        if note and note not in section_buckets[title]["notes"]:
            section_buckets[title]["notes"].append(note)

        for point in list(summary.key_points or []):
            normalized_point = str(point).strip()
            if not normalized_point:
                continue
            if normalized_point not in section_buckets[title]["key_points"]:
                section_buckets[title]["key_points"].append(normalized_point)
            if normalized_point not in concept_candidates:
                concept_candidates.append(normalized_point)

        for example in list(summary.important_examples or []):
            normalized_example = str(example).strip()
            if normalized_example and normalized_example not in takeaway_candidates:
                takeaway_candidates.append(normalized_example)

    sections = [
        VideoSectionSummary(
            title=title,
            summary="\n".join(section_buckets[title]["notes"]) or "暂无章节摘要。",
            key_points=section_buckets[title]["key_points"][:6],
        )
        for title in section_order
    ]

    overall_parts = [
        str(metadata.get("description") or "").strip(),
        *[section.summary for section in sections[:3] if str(section.summary or "").strip()],
    ]
    overall_summary = " ".join(part for part in overall_parts if part).strip() or "这是一支围绕视频主题展开的结构化讲解。"

    key_concepts: list[str] = []
    for item in [*(metadata.get("tags") or []), *concept_candidates]:
        text = str(item).strip()
        if text and text not in key_concepts:
            key_concepts.append(text)

    takeaways: list[str] = []
    for item in [*takeaway_candidates, *concept_candidates, *(metadata.get("tags") or [])]:
        text = str(item).strip()
        if text and text not in takeaways:
            takeaways.append(text)

    return VideoLectureSummary(
        overall_summary=overall_summary,
        key_concepts=key_concepts[:10],
        sections=sections,
        takeaways=takeaways[:8],
    )


def integrate_video_summary(metadata: dict[str, Any], chunk_summaries: list[TranscriptChunkSummary]) -> VideoLectureSummary:
    llm = get_graph_llm()
    segment_payload = [summary.model_dump() for summary in chunk_summaries]
    system_prompt = (
        "你是一名资深课程编辑。请把多个字幕片段总结整合成一份结构化中文视频讲义提纲。"
        "要求：先讲动机，再讲核心概念，再讲机制/步骤，再讲例子或证据，最后给出收束性 takeaway。"
        "不要简单按时间顺序罗列字幕，要重组为教学顺序；同时过滤 B 站平台套话和无教学价值寒暄。"
    )
    human_prompt = (
        f"视频元数据：\n{_serialize_metadata_context(metadata)}\n\n"
        f"片段总结：\n{json.dumps(segment_payload, ensure_ascii=False, indent=2)}"
    )
    try:
        return llm.with_structured_output(VideoLectureSummary).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
    except Exception as exc:
        logger.warning(f"structured video summary integration failed, using deterministic fallback: {exc}")
        return _fallback_integrate_video_summary(metadata, chunk_summaries)


def build_bilibili_tutorial_payload(metadata: dict[str, Any], lecture_summary: VideoLectureSummary) -> dict[str, Any]:
    video_id = str(metadata.get("video_id") or "bilibili_video").strip() or "bilibili_video"
    title = str(metadata.get("title") or video_id).strip() or video_id
    key_concepts = []
    for item in [*(lecture_summary.key_concepts or []), *(metadata.get("tags") or [])]:
        text = str(item).strip()
        if text and text not in key_concepts:
            key_concepts.append(text)

    steps: list[dict[str, Any]] = []
    for index, section in enumerate(lecture_summary.sections or [], start=1):
        content_parts = [str(section.summary or "").strip()]
        if section.key_points:
            content_parts.append(f"关键点：{'；'.join(str(point).strip() for point in section.key_points if str(point).strip())}。")
        steps.append(
            {
                "index": index,
                "title": str(section.title or f"章节{index}").strip() or f"章节{index}",
                "content": " ".join(part for part in content_parts if part).strip(),
            }
        )

    if not steps:
        steps.append(
            {
                "index": 1,
                "title": "视频概览",
                "content": str(lecture_summary.overall_summary or metadata.get("description") or title).strip(),
            }
        )

    duration_seconds = int(metadata.get("duration_seconds") or 0)
    time_minutes = max(int(round(duration_seconds / 60)), 1) if duration_seconds else 0
    ingredients = [{"name": concept, "amount": "", "display": concept} for concept in key_concepts[:8]]
    takeaways = [str(item).strip() for item in lecture_summary.takeaways or [] if str(item).strip()]
    chapters = [str(item).strip() for item in metadata.get("chapters") or [] if str(item).strip()]

    return {
        "tutorial_id": f"tutorial_bilibili_{video_id}",
        "source_type": "bilibili_video",
        "source_id": video_id,
        "source_url": str(metadata.get("webpage_url") or "").strip(),
        "platform": "bilibili",
        "recipe_name": title,
        "title": f"{title}视频总结",
        "summary": str(lecture_summary.overall_summary or metadata.get("description") or title).strip(),
        "difficulty": "",
        "time_minutes": time_minutes,
        "cuisine": "",
        "calories": 0,
        "tags": list(key_concepts[:8]),
        "health_goals": [],
        "scenarios": ["视频学习", "B站"],
        "ingredients": ingredients,
        "key_concepts": key_concepts,
        "steps": steps,
        "step_source": str(metadata.get("transcript_mode") or "subtitles"),
        "tips": takeaways,
        "nutrition": {"protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "vitamins": []},
        "nutrition_highlights": [],
        "related_recipes": [],
        "uploader": str(metadata.get("uploader") or "").strip(),
        "transcript_mode": str(metadata.get("transcript_mode") or "").strip(),
        "cover_path": str(metadata.get("cover_path") or "").strip(),
        "subtitle_path": str(metadata.get("subtitle_path") or "").strip(),
        "chapters": chapters,
    }


def build_bilibili_tutorial_from_url(
    url: str,
    *,
    artifact_dir: str | Path,
    cookies_from_browser: str = "",
    cookies_file: str | Path = "",
    whisper_model: str = "base",
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = ensure_bilibili_transcript(
        url,
        artifact_dir=artifact_dir,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        whisper_model=whisper_model,
    )
    subtitle_path = artifacts.get("subtitle_path")
    if not subtitle_path:
        raise BilibiliSummaryError("未生成可用字幕文件")

    subtitle_entries = load_subtitle_entries(subtitle_path)
    if not subtitle_entries:
        raise BilibiliSummaryError("字幕文件为空或无法解析")

    transcript_path = Path(artifacts["artifact_dir"]) / f"{_sanitize_filename(artifacts['video_id'])}.transcript.txt"
    transcript_path.write_text(
        "\n".join(f"[{item['start']}] {item['text']}" for item in subtitle_entries),
        encoding="utf-8",
    )

    chunk_summaries = [
        summarize_transcript_chunk(artifacts, chunk, chunk_index=index)
        for index, chunk in enumerate(_chunk_subtitle_entries(subtitle_entries), start=1)
    ]
    lecture_summary = integrate_video_summary(artifacts, chunk_summaries)
    tutorial_payload = build_bilibili_tutorial_payload(artifacts, lecture_summary)
    metadata_path = Path(artifacts["artifact_dir"]) / f"{_sanitize_filename(artifacts['video_id'])}.metadata.json"
    metadata_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts["metadata_path"] = str(metadata_path)
    artifacts["transcript_path"] = str(transcript_path)
    return tutorial_payload, artifacts


__all__ = [
    "BilibiliSummaryError",
    "build_bilibili_tutorial_from_url",
    "build_bilibili_tutorial_payload",
    "download_bilibili_artifacts",
    "ensure_bilibili_transcript",
    "inspect_bilibili_metadata",
    "integrate_video_summary",
    "load_subtitle_entries",
    "summarize_transcript_chunk",
]
