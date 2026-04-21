import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.tutorials import (
    build_bilibili_tutorial_from_url,
    build_tutorial_chunk_corpus,
    default_bilibili_artifact_root,
    export_tutorial_to_pdf,
    resolve_bilibili_tutorial_storage,
    save_tutorial_to_json,
)
from src.utils.logger import configure_console_utf8
from src.vectorstore.tutorial_store import batch_add_tutorial_chunks, get_tutorial_collection_count


def _default_output_dir() -> Path:
    return default_bilibili_artifact_root(PROJECT_ROOT)


def main() -> int:
    configure_console_utf8()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a generic Bilibili video tutorial summary from a video URL")
    parser.add_argument("url", help="Bilibili video URL")
    parser.add_argument("--output-dir", default=str(_default_output_dir()), help="artifact output root directory")
    parser.add_argument("--json-output", default="", help="tutorial json output path")
    parser.add_argument("--pdf-output", default="", help="tutorial pdf output path")
    parser.add_argument("--cookies-from-browser", default=settings.bilibili_cookies_from_browser, help="browser name for yt-dlp cookies, e.g. chrome")
    parser.add_argument("--cookies-file", default=settings.bilibili_cookies_file, help="path to exported cookies.txt for yt-dlp")
    parser.add_argument("--whisper-model", default=settings.bilibili_whisper_model, help="whisper model name when subtitle fallback is needed")
    parser.add_argument("--skip-pdf", action="store_true", help="skip tutorial PDF export")
    parser.add_argument("--skip-ingest", action="store_true", help="skip vectorstore ingest")
    parser.add_argument("--collection-name", default=settings.bilibili_summary_collection_name, help="target collection name for bilibili tutorial chunks")
    parser.add_argument("--max-steps-per-chunk", type=int, default=3, help="maximum section steps per chunk")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tutorial_payload, artifacts = build_bilibili_tutorial_from_url(
        args.url,
        artifact_dir=output_dir,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies_file,
        whisper_model=args.whisper_model,
    )

    storage = resolve_bilibili_tutorial_storage(
        project_root=PROJECT_ROOT,
        artifact_root=output_dir,
        video_id=str(tutorial_payload.get("source_id") or artifacts.get("video_id") or ""),
        tutorial_id=str(tutorial_payload.get("tutorial_id") or ""),
        tutorial_title=str(tutorial_payload.get("title") or ""),
    )
    tutorial_json_path = Path(args.json_output) if args.json_output else storage.tutorial_json_path
    tutorial_json_path = save_tutorial_to_json(tutorial_payload, tutorial_json_path)
    print(f"tutorial json written: {tutorial_json_path}")
    print(f"video title: {tutorial_payload['recipe_name']}")
    print(f"subtitle mode: {artifacts.get('transcript_mode') or '-'}")
    print(f"artifact dir: {artifacts.get('artifact_dir') or storage.artifact_dir}")
    if artifacts.get("subtitle_path"):
        print(f"subtitle path: {artifacts['subtitle_path']}")
    if artifacts.get("cover_path"):
        print(f"cover path: {artifacts['cover_path']}")

    if not args.skip_pdf:
        pdf_path = Path(args.pdf_output) if args.pdf_output else storage.pdf_path
        pdf_path = export_tutorial_to_pdf(tutorial_payload, pdf_path)
        print(f"pdf written: {pdf_path}")

    if not args.skip_ingest:
        chunk_corpus = build_tutorial_chunk_corpus([tutorial_payload], max_steps_per_chunk=args.max_steps_per_chunk)
        before_count = get_tutorial_collection_count(args.collection_name)
        success_count, fail_count, errors = batch_add_tutorial_chunks(chunk_corpus, collection_name=args.collection_name)
        after_count = get_tutorial_collection_count(args.collection_name)
        print(f"tutorial chunk corpus size: {len(chunk_corpus)}")
        if before_count is not None:
            print(f"tutorial collection count before ingest: {before_count}")
        print(f"tutorial chunks ingested: {success_count}")
        print(f"tutorial chunk failures: {fail_count}")
        if after_count is not None:
            print(f"tutorial collection count after ingest: {after_count}")
        if errors:
            print("ingest errors:")
            for error in errors:
                print(f"  - {error}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
