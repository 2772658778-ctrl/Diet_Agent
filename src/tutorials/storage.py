from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecipeTutorialStoragePaths:
    json_path: Path
    pdf_dir: Path


@dataclass(frozen=True)
class BilibiliTutorialStoragePaths:
    video_id: str
    content_dir: Path
    tutorial_json_path: Path
    pdf_dir: Path
    pdf_path: Path
    artifact_dir: Path


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root)
    return Path(__file__).resolve().parents[2]


def sanitize_storage_name(value: str, fallback: str = "tutorial") -> str:
    raw = str(value or fallback).strip() or fallback
    sanitized = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in raw)
    sanitized = sanitized.rstrip(". ")
    return sanitized or fallback


def default_bilibili_artifact_root(project_root: str | Path | None = None) -> Path:
    return _project_root(project_root) / "artifacts" / "bilibili_summaries"


def resolve_recipe_tutorial_storage(
    *,
    project_root: str | Path | None = None,
    dataset_name: str = "recipes_v2",
) -> RecipeTutorialStoragePaths:
    root = _project_root(project_root)
    dataset = sanitize_storage_name(dataset_name, "recipes")
    content_dir = root / "data" / "tutorials" / "recipes"
    return RecipeTutorialStoragePaths(
        json_path=content_dir / f"recipe_tutorials.{dataset}.json",
        pdf_dir=root / "artifacts" / "tutorial_pdfs" / "recipes",
    )


def resolve_bilibili_artifact_dir(artifact_root: str | Path, video_id: str) -> Path:
    root = Path(artifact_root)
    video_segment = sanitize_storage_name(video_id, "bilibili_video")
    if root.name == video_segment:
        return root
    return root / video_segment


def resolve_bilibili_tutorial_storage(
    *,
    video_id: str,
    tutorial_id: str = "",
    tutorial_title: str = "",
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> BilibiliTutorialStoragePaths:
    root = _project_root(project_root)
    resolved_video_id = sanitize_storage_name(video_id, "bilibili_video")
    content_dir = root / "data" / "tutorials" / "bilibili" / resolved_video_id
    resolved_tutorial_id = sanitize_storage_name(
        tutorial_id or f"tutorial_bilibili_{resolved_video_id}",
        "tutorial_bilibili",
    )
    resolved_title = sanitize_storage_name(
        tutorial_title or tutorial_id or resolved_video_id,
        resolved_video_id,
    )
    resolved_artifact_root = artifact_root if artifact_root is not None else default_bilibili_artifact_root(root)
    return BilibiliTutorialStoragePaths(
        video_id=resolved_video_id,
        content_dir=content_dir,
        tutorial_json_path=content_dir / f"{resolved_tutorial_id}.json",
        pdf_dir=root / "artifacts" / "tutorial_pdfs" / "bilibili" / resolved_video_id,
        pdf_path=root / "artifacts" / "tutorial_pdfs" / "bilibili" / resolved_video_id / f"{resolved_title}.pdf",
        artifact_dir=resolve_bilibili_artifact_dir(resolved_artifact_root, resolved_video_id),
    )
