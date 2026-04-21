# -*- coding: utf-8 -*-
"""
FastAPI 接口模块 (Phase 4)

提供 Diet Agent 的 HTTP API 和 SSE 流式接口：
- app: FastAPI 应用实例
- ChatRequest / ChatResponse: 请求/响应模型
- create_app: 应用工厂函数

复用模块:
- src/graph/diet_graph.py::build_diet_graph(), run_diet_agent()
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from __future__ import annotations

from typing import Any

from .schemas import ChatRequest, ChatResponse, StreamEvent


def __getattr__(name: str) -> Any:
    if name == "app":
        from .main import app

        return app
    if name == "create_app":
        from .main import create_app

        return create_app
    raise AttributeError(f"module 'src.api' has no attribute {name!r}")

__all__ = [
    "app",
    "create_app",
    "ChatRequest",
    "ChatResponse",
    "StreamEvent",
]
