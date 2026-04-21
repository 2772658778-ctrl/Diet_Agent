import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import configure_console_utf8
from src.tutorials import (
    build_recipe_tutorials,
    build_tutorial_chunk_corpus,
    export_tutorials_to_pdf,
    load_recipes_from_file,
    resolve_recipe_tutorial_storage,
    save_tutorials_to_json,
)
from src.vectorstore.tutorial_store import batch_add_tutorial_chunks, get_tutorial_collection_count


def _default_storage(input_path: str | Path) -> tuple[Path, Path]:
    storage = resolve_recipe_tutorial_storage(
        project_root=PROJECT_ROOT,
        dataset_name=Path(input_path).stem,
    )
    return storage.json_path, storage.pdf_dir


def main() -> int:
    configure_console_utf8()
    parser = argparse.ArgumentParser(description="Build tutorial JSON, PDF, and vectorstore chunks from recipe data")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "recipes_v2.json"), help="recipe source json path")
    parser.add_argument("--json-output", default="", help="tutorial json output path")
    parser.add_argument("--pdf-output-dir", default="", help="tutorial pdf output directory")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N recipes")
    parser.add_argument("--skip-pdf", action="store_true", help="skip pdf export")
    parser.add_argument("--skip-ingest", action="store_true", help="skip vectorstore ingest")
    parser.add_argument("--max-steps-per-chunk", type=int, default=3, help="maximum tutorial steps per chunk")
    args = parser.parse_args()

    default_json_output, default_pdf_output = _default_storage(args.input)
    json_output = Path(args.json_output) if args.json_output else default_json_output
    pdf_output_dir = Path(args.pdf_output_dir) if args.pdf_output_dir else default_pdf_output

    recipes = load_recipes_from_file(args.input)
    if args.limit and args.limit > 0:
        recipes = recipes[: args.limit]

    tutorials = build_recipe_tutorials(recipes)
    json_path = save_tutorials_to_json(tutorials, json_output)
    print(f"tutorial json written: {json_path}")
    print(f"tutorial count: {len(tutorials)}")

    if not args.skip_pdf:
        pdf_paths = export_tutorials_to_pdf(tutorials, pdf_output_dir)
        print(f"pdf exported: {len(pdf_paths)}")
        print(f"pdf output dir: {Path(pdf_output_dir).resolve()}")

    if not args.skip_ingest:
        chunk_corpus = build_tutorial_chunk_corpus(tutorials, max_steps_per_chunk=args.max_steps_per_chunk)
        before_count = get_tutorial_collection_count()
        success_count, fail_count, errors = batch_add_tutorial_chunks(chunk_corpus)
        after_count = get_tutorial_collection_count()
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
