"""
向量存储模块

提供向量数据库的初始化、管理和检索功能
"""

from .chroma_client import (
    init_vectorstore,
    get_vectorstore,
    VectorStoreManager
)
from .data_loader import build_document_text
from .incremental_update import (
    add_recipe,
    update_recipe,
    batch_add_recipes,
    batch_update_recipes,
    add_recipes_from_file,
    delete_recipe
)
from .tutorial_store import (
    batch_add_tutorial_chunks,
    get_or_connect_tutorial_vectorstore,
    get_tutorial_vectorstore,
)

__all__ = [
    'init_vectorstore',
    'get_vectorstore',
    'VectorStoreManager',
    'build_document_text',
    'add_recipe',
    'update_recipe',
    'batch_add_recipes',
    'batch_update_recipes',
    'add_recipes_from_file',
    'delete_recipe',
    'batch_add_tutorial_chunks',
    'get_or_connect_tutorial_vectorstore',
    'get_tutorial_vectorstore',
]
