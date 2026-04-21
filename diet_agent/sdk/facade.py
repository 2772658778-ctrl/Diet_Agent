"""Stable SDK facade for Diet Agent."""

from __future__ import annotations

from typing import Any

from diet_agent.app import build_app as build_reference_app
from diet_agent.evaluation import benchmark as run_benchmark
from diet_agent.runtime import arun_diet_agent_stream, build_diet_graph, register_skill as register_runtime_skill, run_diet_agent


def run(*args: Any, **kwargs: Any):
    return run_diet_agent(*args, **kwargs)


def stream(*args: Any, **kwargs: Any):
    return arun_diet_agent_stream(*args, **kwargs)


def build_app(*args: Any, **kwargs: Any):
    return build_reference_app(*args, **kwargs)


def register_skill(name: str, skill_spec: dict[str, Any]) -> dict[str, Any]:
    return register_runtime_skill(name, skill_spec)


def benchmark(
    dataset_path: str | None = None,
    *,
    case_ids: list[str] | None = None,
    show_progress: bool = False,
    skip_generation_metrics: bool = False,
    skip_ragas_metrics: bool = False,
    skip_graph_eval: bool = False,
    enable_generation_metrics: bool = True,
    enable_ragas_metrics: bool = True,
) -> dict[str, Any]:
    return run_benchmark(
        dataset_path=dataset_path,
        case_ids=case_ids,
        show_progress=show_progress,
        skip_generation_metrics=skip_generation_metrics,
        skip_ragas_metrics=skip_ragas_metrics,
        skip_graph_eval=skip_graph_eval,
        enable_generation_metrics=enable_generation_metrics,
        enable_ragas_metrics=enable_ragas_metrics,
    )


__all__ = [
    "run",
    "stream",
    "build_app",
    "build_diet_graph",
    "register_skill",
    "benchmark",
]
