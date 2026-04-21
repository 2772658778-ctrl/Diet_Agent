"""Reference app factory wrappers."""


def build_app(*args, **kwargs):
    from src.api.main import create_app
    return create_app(*args, **kwargs)


def __getattr__(name):
    if name == "app":
        from src.api.main import app as _app
        return _app
    raise AttributeError(name)


__all__ = ["app", "build_app"]
