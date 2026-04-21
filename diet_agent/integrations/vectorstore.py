"""Vector store integration adapters."""

from src.vectorstore.chroma_client import get_or_connect_vectorstore, get_vectorstore, init_vectorstore


def add_recipe(*args, **kwargs):
    from src.vectorstore import add_recipe as _add_recipe

    return _add_recipe(*args, **kwargs)


def update_recipe(*args, **kwargs):
    from src.vectorstore import update_recipe as _update_recipe

    return _update_recipe(*args, **kwargs)


def batch_add_recipes(*args, **kwargs):
    from src.vectorstore import batch_add_recipes as _batch_add_recipes

    return _batch_add_recipes(*args, **kwargs)


def batch_update_recipes(*args, **kwargs):
    from src.vectorstore import batch_update_recipes as _batch_update_recipes

    return _batch_update_recipes(*args, **kwargs)


def add_recipes_from_file(*args, **kwargs):
    from src.vectorstore import add_recipes_from_file as _add_recipes_from_file

    return _add_recipes_from_file(*args, **kwargs)


def batch_add_tutorial_chunks(*args, **kwargs):
    from src.vectorstore import batch_add_tutorial_chunks as _batch_add_tutorial_chunks

    return _batch_add_tutorial_chunks(*args, **kwargs)


def delete_recipe(*args, **kwargs):
    from src.vectorstore import delete_recipe as _delete_recipe

    return _delete_recipe(*args, **kwargs)


def get_tutorial_vectorstore(*args, **kwargs):
    from src.vectorstore import get_tutorial_vectorstore as _get_tutorial_vectorstore

    return _get_tutorial_vectorstore(*args, **kwargs)


def get_or_connect_tutorial_vectorstore(*args, **kwargs):
    from src.vectorstore import get_or_connect_tutorial_vectorstore as _get_or_connect_tutorial_vectorstore

    return _get_or_connect_tutorial_vectorstore(*args, **kwargs)


def init_kg_enhanced_vectorstore(*args, **kwargs):
    from src.vectorstore.kg_enhanced_client import init_kg_enhanced_vectorstore as _init_kg_enhanced_vectorstore

    return _init_kg_enhanced_vectorstore(*args, **kwargs)


def get_kg_enhanced_vectorstore(*args, **kwargs):
    from src.vectorstore.kg_enhanced_client import get_vectorstore as _get_kg_enhanced_vectorstore

    return _get_kg_enhanced_vectorstore(*args, **kwargs)


__all__ = [
    "add_recipe",
    "update_recipe",
    "batch_add_recipes",
    "batch_update_recipes",
    "add_recipes_from_file",
    "batch_add_tutorial_chunks",
    "delete_recipe",
    "get_or_connect_vectorstore",
    "get_vectorstore",
    "get_tutorial_vectorstore",
    "get_or_connect_tutorial_vectorstore",
    "init_vectorstore",
    "init_kg_enhanced_vectorstore",
    "get_kg_enhanced_vectorstore",
]
