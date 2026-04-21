# -*- coding: utf-8 -*-
"""
SSE 流式生成器

封装 LangGraph astream_events() 为 Server-Sent Events 格式，
支持文本 chunk 逐字输出和节点进度通知。

复用模块:
- src/utils/logger.py::get_logger()
"""

import json
from typing import AsyncGenerator, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _format_sse_event(event_type: str, data: dict) -> str:
    """将事件格式化为 SSE 格式

    Args:
        event_type: 事件类型（chunk / node_start / node_end / done / error）
        data: 事件数据字典

    Returns:
        SSE 格式字符串：data: {"event": "...", ...}\n\n
    """
    payload = {"event": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_node_event(event: dict) -> Optional[dict]:
    """从 astream_events 提取节点开始/结束事件

    Args:
        event: LangGraph 事件字典

    Returns:
        提取的节点事件字典，无关事件返回 None
    """
    event_type = event.get("event", "")
    name = event.get("name", "")

    if event_type == "on_chain_start" and name in (
        "router", "planner", "retriever", "generator", "evaluator"
    ):
        return {"type": "node_start", "node": name}
    elif event_type == "on_chain_end" and name in (
        "router", "planner", "retriever", "generator", "evaluator"
    ):
        return {"type": "node_end", "node": name}
    return None


async def stream_graph_events(
    graph,
    initial_state: dict,
    config: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """将 LangGraph astream_events 转换为 SSE 事件流

    Args:
        graph: 编译后的 LangGraph 实例
        initial_state: LangGraph 初始状态
        config: LangGraph 运行配置（可包含 callbacks）

    Yields:
        SSE 格式的字符串：data: {"event": "...", "content": "..."}\n\n
    """
    try:
        async for event in graph.astream_events(
            initial_state, version="v2", config=config
        ):
            # 文本 chunk（LLM 流式输出）
            if event.get("event") == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk", None)
                if chunk is not None:
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk
                    if content:
                        yield _format_sse_event("chunk", {"content": content})

            # 节点进度事件
            node_event = _extract_node_event(event)
            if node_event is not None:
                yield _format_sse_event(
                    node_event["type"], {"node": node_event["node"]}
                )

    except Exception as e:
        logger.error(f"SSE 流式生成异常: {e}", exc_info=True)
        yield _format_sse_event("error", {"content": str(e)})

    # 流结束标记
    yield _format_sse_event("done", {"content": ""})
