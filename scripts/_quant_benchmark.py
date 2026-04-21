import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation import QuantitativeBenchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "quant_benchmark_seed.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "scripts" / "quant_benchmark_report.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "scripts" / "quant_benchmark_report.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-output", default=str(DEFAULT_MD_OUTPUT))
    parser.add_argument("--version-label", default="quant-benchmark")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--skip-generation-metrics", action="store_true")
    parser.add_argument("--skip-ragas-metrics", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    json_output = Path(args.json_output)
    md_output = Path(args.md_output)
    case_ids = [str(case_id).strip() for case_id in args.case_ids if str(case_id).strip()]
    skip_generation_metrics = bool(args.skip_generation_metrics or args.fast)
    skip_ragas_metrics = bool(args.skip_ragas_metrics or args.fast)

    benchmark = QuantitativeBenchmark(
        dataset_path=str(dataset_path),
        enable_generation_metrics=not skip_generation_metrics,
        enable_ragas_metrics=not skip_ragas_metrics,
    )
    print(
        f"Running quant benchmark on {dataset_path} ... "
        f"case_ids={case_ids or 'all'} fast={bool(args.fast)} "
        f"skip_generation_metrics={skip_generation_metrics} skip_ragas_metrics={skip_ragas_metrics}",
        flush=True,
    )
    results = benchmark.run_all(
        show_progress=True,
        case_ids=case_ids or None,
        skip_generation_metrics=skip_generation_metrics,
        skip_ragas_metrics=skip_ragas_metrics,
        skip_graph_eval=bool(args.fast),
    )
    results["json_output_path"] = str(json_output)
    results["md_output_path"] = str(md_output)

    json_output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_output.write_text(
        benchmark.generate_report(results, version_label=args.version_label),
        encoding="utf-8",
    )

    print(json.dumps(results.get("summary", {}), ensure_ascii=False, indent=2))
    print(f"JSON report written to: {json_output}")
    print(f"Markdown report written to: {md_output}")


if __name__ == "__main__":
    main()
