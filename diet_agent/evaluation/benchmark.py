"""Evaluation facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.quant_benchmark import QuantBenchmarkCase, QuantBenchmarkResult, QuantitativeBenchmark


def benchmark(
    dataset_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    resolved_dataset_path = dataset_path or str(
        Path(__file__).resolve().parents[2] / "src" / "evaluation" / "test_dataset.json"
    )
    enable_generation_metrics = kwargs.pop("enable_generation_metrics", True)
    enable_ragas_metrics = kwargs.pop("enable_ragas_metrics", True)
    runner = QuantitativeBenchmark(
        dataset_path=resolved_dataset_path,
        enable_generation_metrics=enable_generation_metrics,
        enable_ragas_metrics=enable_ragas_metrics,
    )
    return runner.run_all(**kwargs)


__all__ = ["benchmark", "QuantBenchmarkCase", "QuantBenchmarkResult", "QuantitativeBenchmark"]
