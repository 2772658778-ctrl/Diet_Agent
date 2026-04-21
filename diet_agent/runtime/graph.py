"""Runtime graph wrappers."""


def build_diet_graph(*args, **kwargs):
    from src.graph.diet_graph import build_diet_graph as _build_diet_graph
    return _build_diet_graph(*args, **kwargs)


def run_diet_agent(*args, **kwargs):
    from src.graph.diet_graph import run_diet_agent as _run_diet_agent
    return _run_diet_agent(*args, **kwargs)


def arun_diet_agent_stream(*args, **kwargs):
    from src.graph.diet_graph import arun_diet_agent_stream as _arun_diet_agent_stream
    return _arun_diet_agent_stream(*args, **kwargs)


__all__ = ["build_diet_graph", "run_diet_agent", "arun_diet_agent_stream"]
