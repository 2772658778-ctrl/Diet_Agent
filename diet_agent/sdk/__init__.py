"""Public SDK facade for Diet Agent."""

from .facade import benchmark, build_app, register_skill, run, stream

__all__ = ["run", "stream", "benchmark", "register_skill", "build_app"]
