import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tutorials.windsurf_skill_bridge import render_bilibili_tutorial_pdf_from_json
from src.utils.logger import configure_console_utf8


def main() -> int:
    configure_console_utf8()
    parser = argparse.ArgumentParser(description="Render a Bilibili tutorial PDF from canonical tutorial JSON")
    parser.add_argument("--json-path", required=True, help="path to tutorial json")
    parser.add_argument("--pdf-output", default="", help="optional explicit pdf output path")
    args = parser.parse_args()

    pdf_path = render_bilibili_tutorial_pdf_from_json(
        args.json_path,
        project_root=PROJECT_ROOT,
        pdf_output=args.pdf_output or None,
    )
    print(f"pdf written: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
