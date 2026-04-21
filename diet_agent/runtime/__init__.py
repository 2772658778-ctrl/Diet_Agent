"""Runtime exports for Diet Agent."""


def build_diet_graph(*args, **kwargs):
    from .graph import build_diet_graph as _build_diet_graph
    return _build_diet_graph(*args, **kwargs)


def run_diet_agent(*args, **kwargs):
    from .graph import run_diet_agent as _run_diet_agent
    return _run_diet_agent(*args, **kwargs)


def arun_diet_agent_stream(*args, **kwargs):
    from .graph import arun_diet_agent_stream as _arun_diet_agent_stream
    return _arun_diet_agent_stream(*args, **kwargs)


def register_skill(*args, **kwargs):
    from .skills import register_skill as _register_skill
    return _register_skill(*args, **kwargs)


def select_skill(*args, **kwargs):
    from .skills import select_skill as _select_skill
    return _select_skill(*args, **kwargs)


def select_skill_name(*args, **kwargs):
    from .skills import select_skill_name as _select_skill_name
    return _select_skill_name(*args, **kwargs)


def get_skill_spec(*args, **kwargs):
    from .skills import get_skill_spec as _get_skill_spec
    return _get_skill_spec(*args, **kwargs)


def list_skill_specs(*args, **kwargs):
    from .skills import list_skill_specs as _list_skill_specs
    return _list_skill_specs(*args, **kwargs)


def get_skill_runtime_policy(*args, **kwargs):
    from .policy import get_skill_runtime_policy as _get_skill_runtime_policy
    return _get_skill_runtime_policy(*args, **kwargs)


def get_skill_capability_status(*args, **kwargs):
    from .policy import get_skill_capability_status as _get_skill_capability_status
    return _get_skill_capability_status(*args, **kwargs)


def resolve_skill_runtime(*args, **kwargs):
    from .policy import resolve_skill_runtime as _resolve_skill_runtime
    return _resolve_skill_runtime(*args, **kwargs)


def build_skill_quality_signals(*args, **kwargs):
    from .policy import build_skill_quality_signals as _build_skill_quality_signals
    return _build_skill_quality_signals(*args, **kwargs)


def get_skill_assets(*args, **kwargs):
    from .assets import get_skill_assets as _get_skill_assets
    return _get_skill_assets(*args, **kwargs)


def list_skill_assets(*args, **kwargs):
    from .assets import list_skill_assets as _list_skill_assets
    return _list_skill_assets(*args, **kwargs)


def __getattr__(name):
    if name == "SkillSpec":
        from .skills import SkillSpec as _SkillSpec
        return _SkillSpec
    if name == "SkillAssetView":
        from .assets import SkillAssetView as _SkillAssetView
        return _SkillAssetView
    raise AttributeError(name)


__all__ = [
    "build_diet_graph",
    "run_diet_agent",
    "arun_diet_agent_stream",
    "SkillAssetView",
    "SkillSpec",
    "build_skill_quality_signals",
    "get_skill_assets",
    "get_skill_capability_status",
    "get_skill_spec",
    "get_skill_runtime_policy",
    "list_skill_assets",
    "list_skill_specs",
    "register_skill",
    "resolve_skill_runtime",
    "select_skill",
    "select_skill_name",
]
