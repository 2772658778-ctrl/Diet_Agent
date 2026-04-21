"""Reference app exports for Diet Agent."""


def build_app(*args, **kwargs):
    from .factory import build_app as _build_app
    return _build_app(*args, **kwargs)


def get_app():
    from .factory import app as _app
    return _app


__all__ = ["build_app", "get_app"]
