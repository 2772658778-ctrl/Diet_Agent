"""Diet Agent public package."""

__version__ = "0.1.0"


def run(*args, **kwargs):
    from .sdk import run
    return run(*args, **kwargs)


def stream(*args, **kwargs):
    from .sdk import stream
    return stream(*args, **kwargs)


def benchmark(*args, **kwargs):
    from .sdk import benchmark
    return benchmark(*args, **kwargs)


def register_skill(*args, **kwargs):
    from .sdk import register_skill
    return register_skill(*args, **kwargs)


def build_app(*args, **kwargs):
    from .sdk import build_app
    return build_app(*args, **kwargs)


__all__ = [
    "__version__",
    "run",
    "stream",
    "benchmark",
    "register_skill",
    "build_app",
]
