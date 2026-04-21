# -*- coding: utf-8 -*-
"""
FastAPI 应用入口

提供 Diet Agent 的 HTTP API：
- POST /chat         — 同步调用，返回完整回复
- POST /chat/stream  — SSE 流式调用，逐 chunk 返回
- GET  /health       — 健康检查

复用模块:
- src/graph/diet_graph.py::build_diet_graph(), run_diet_agent()
- src/api/schemas.py::ChatRequest, ChatResponse
- src/api/streaming.py::stream_graph_events()
- src/observability/langfuse_integration.py::get_langfuse_callback()
- src/observability/structured_logger.py::RequestLogger
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from diet_agent.app.service import (
    build_chat_metadata,
    build_recent_session_summaries,
    build_session_history_view,
    build_recommended_recipe_summaries,
    collect_request_memory_context,
    hydrate_final_state,
    sync_memory_artifacts,
)
from diet_agent.integrations.database import PostgreSQLClient
from diet_agent.integrations.observability import RequestLogger, get_langfuse_callback
from diet_agent.runtime import build_diet_graph, run_diet_agent
from diet_agent.user import (
    get_session_store,
    load_stable_user_preferences,
    write_memory_loopback,
)
from ..config import get_settings
from ..graph.schemas import normalize_extracted_params
from ..graph.state import build_initial_diet_agent_state
from ..utils.logger import get_logger
from ..utils.token_usage import normalize_token_usage
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionHistoryResponse,
    SessionListResponse,
)

logger = get_logger(__name__)
settings = get_settings()
_session_store = get_session_store()
_session_history = _session_store._session_history
_feedback_memory = _session_store._feedback_memory
_recommendation_memory = _session_store._recommendation_memory
_demo_page_path = Path(__file__).resolve().parents[2] / "web_chat_demo.html"


# ── Graph 单例 ───────────────────────────────────────────────────────────

_compiled_graph = None


def _get_stable_user_preferences(user_id: str) -> dict:
    return load_stable_user_preferences(user_id)


def _remember_feedback_event(user_id: str, feedback: "FeedbackPayload") -> None:
    _session_store.remember_feedback_event(user_id, feedback)


def _get_recent_feedback_signals(user_id: str) -> dict:
    return _session_store.get_recent_feedback_signals(user_id)


def _get_recent_recommended_recipes(user_id: str) -> list[dict]:
    return _session_store.get_recent_recommended_recipes(user_id)


def _write_memory_loopback(
    request: ChatRequest,
    request_id: str,
    response_text: str,
    final_state: dict,
) -> tuple[str, bool]:
    return write_memory_loopback(
        request=request,
        request_id=request_id,
        response_text=response_text,
        final_state=final_state,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时编译 graph"""
    global _compiled_graph
    _compiled_graph = build_diet_graph()
    logger.info("FastAPI 启动完成，LangGraph 已编译")
    yield
    logger.info("FastAPI 关闭")


def _build_initial_state(request: ChatRequest, request_id: str) -> dict:
    """从请求构建 LangGraph 初始 State

    Args:
        request: 聊天请求
        request_id: 请求唯一标识

    Returns:
        LangGraph 初始状态字典
    """
    uid = request.user_id or ""
    memory_context = collect_request_memory_context(
        _session_store,
        uid,
        session_id=request.session_id,
    )
    state = build_initial_diet_agent_state(
        messages=list(memory_context["history_messages"]) + [HumanMessage(content=request.query)],
        user_id=uid,
        request_id=request_id,
    )
    state["extracted_params"] = normalize_extracted_params({
        "available_ingredients": request.available_ingredients,
        "allergies": request.allergies,
        "disliked_ingredients": request.disliked_ingredients,
        "max_cooking_time": request.max_cooking_time,
        "health_goal": request.health_goal,
        "meal_type": request.meal_type,
        "prefer_inventory_first": request.prefer_inventory_first,
    })
    if memory_context["recent_feedback_signals"]:
        state["recent_feedback_signals"] = dict(memory_context["recent_feedback_signals"])
    if memory_context["recent_recommended_recipes"]:
        state["recent_recommended_recipes"] = [
            dict(item) for item in memory_context["recent_recommended_recipes"]
        ]
    state["stable_user_preferences"] = _get_stable_user_preferences(uid)
    return state


def _get_langfuse_config(request_id: str, request: ChatRequest) -> Optional[dict]:
    """构建包含 Langfuse 回调的 config（若启用）

    Args:
        request_id: 请求唯一标识
        request: 聊天请求

    Returns:
        LangGraph config dict，或 None
    """
    config: dict = {"request_id": request_id}
    try:
        handler = get_langfuse_callback(
            trace_name="diet_agent",
            user_id=request.user_id,
            session_id=request.session_id or "",
            metadata={"request_id": request_id},
        )
        if handler is not None:
            config["callbacks"] = [handler]
    except Exception:
        pass  # Langfuse 不可用时降级
    return config


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例（工厂函数）

    Returns:
        配置好路由和中间件的 FastAPI 实例
    """
    application = FastAPI(
        title="Diet Agent API",
        description="智能饮食推荐 Agent — LangGraph + Adaptive RAG",
        version="4.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins.split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 路由注册 ─────────────────────────────────────────────────────────

    @application.get("/health", response_model=HealthResponse)
    async def health():
        """健康检查端点（增强版：检查全部组件连通性）"""
        components = {
            "graph": "compiled" if _compiled_graph is not None else "not_ready",
            "langfuse": "enabled" if settings.langfuse_enabled else "disabled",
        }

        # PostgreSQL 连通性
        try:
            pg = PostgreSQLClient()
            with pg.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            components["postgres"] = "connected"
        except Exception:
            components["postgres"] = "disconnected"

        # ChromaDB 连通性
        try:
            import chromadb
            client = chromadb.Client()
            client.heartbeat()
            components["chromadb"] = "connected"
        except Exception:
            components["chromadb"] = "disconnected"

        overall = "ok" if components.get("graph") == "compiled" else "degraded"

        return HealthResponse(
            status=overall,
            version="4.0.0",
            components=components,
        )

    @application.get("/users/{user_id}/sessions", response_model=SessionListResponse)
    async def list_recent_sessions(user_id: str):
        sessions = build_recent_session_summaries(_session_store, user_id)
        return SessionListResponse(user_id=user_id, sessions=sessions)

    @application.get(
        "/users/{user_id}/sessions/{session_id}/history",
        response_model=SessionHistoryResponse,
    )
    async def get_session_history(user_id: str, session_id: str):
        history_view = build_session_history_view(_session_store, user_id, session_id)
        return SessionHistoryResponse(**history_view)

    @application.get("/demo", include_in_schema=False)
    async def get_web_chat_demo():
        if not _demo_page_path.exists():
            raise HTTPException(status_code=404, detail="Demo 页面不存在")
        return FileResponse(_demo_page_path, media_type="text/html")

    @application.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """同步聊天端点

        调用 run_diet_agent()，返回完整回复。
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # 结构化日志
        req_logger = None
        if settings.structured_log_enabled:
            req_logger = RequestLogger(request_id)

        try:
            config = _get_langfuse_config(request_id, request)
            uid = request.user_id or ""
            memory_context = collect_request_memory_context(
                _session_store,
                uid,
                session_id=request.session_id,
            )
            prev_messages = memory_context["history_messages"]
            recent_feedback_signals = memory_context["recent_feedback_signals"]
            recent_recommended_recipes = memory_context["recent_recommended_recipes"]
            stable_user_preferences = _get_stable_user_preferences(uid)
            response_text, final_state = run_diet_agent(
                query=request.query,
                user_id=uid,
                config=config,
                return_state=True,
                history_messages=prev_messages,
                initial_extracted_params={
                    "available_ingredients": request.available_ingredients,
                    "allergies": request.allergies,
                    "disliked_ingredients": request.disliked_ingredients,
                    "max_cooking_time": request.max_cooking_time,
                    "health_goal": request.health_goal,
                    "meal_type": request.meal_type,
                    "prefer_inventory_first": request.prefer_inventory_first,
                },
                initial_recent_feedback_signals=recent_feedback_signals,
                initial_recent_recommended_recipes=recent_recommended_recipes,
                initial_stable_user_preferences=stable_user_preferences,
            )
            _session_store.set_history(
                uid,
                list(final_state.get("messages", [])),
                session_id=request.session_id,
            )
            final_state = hydrate_final_state(
                final_state,
                recent_feedback_signals=recent_feedback_signals,
                recent_recommended_recipes=recent_recommended_recipes,
                stable_user_preferences=stable_user_preferences,
            )

            latency_ms = (time.monotonic() - start_time) * 1000
            intent = final_state.get("intent", "")
            retrieved = final_state.get("retrieved_docs", [])
            reranked = final_state.get("reranked_docs", [])
            retrieval_stats = final_state.get("retrieval_stats", {})
            evaluation = final_state.get("evaluation", {})
            missing_slots = final_state.get("missing_slots", [])
            interaction_id, feedback_logged = _write_memory_loopback(
                request=request,
                request_id=request_id,
                response_text=response_text,
                final_state=final_state,
            )
            if request.feedback is not None:
                _remember_feedback_event(uid, request.feedback)
                if not feedback_logged:
                    feedback_logged = True
            recommended_recipes = build_recommended_recipe_summaries(final_state)
            memory_runtime = sync_memory_artifacts(
                final_state,
                user_id=uid,
                session_store=_session_store,
                request_feedback=request.feedback,
                recommended_recipes=recommended_recipes,
                stable_user_preferences=stable_user_preferences,
            )
            metadata = build_chat_metadata(
                final_state,
                prev_messages=prev_messages,
                retrieval_stats=retrieval_stats,
                evaluation=evaluation,
                missing_slots=missing_slots,
                interaction_id=interaction_id,
                feedback_logged=feedback_logged,
                recommended_recipes=memory_runtime["recommended_recipes"],
                effective_feedback_signals=memory_runtime["effective_feedback_signals"],
                effective_recommendation_anchors=memory_runtime["effective_recommendation_anchors"],
                stable_user_preferences=stable_user_preferences,
            )

            if req_logger:
                eval_result = final_state.get("evaluation", {})
                token_usage = normalize_token_usage(final_state.get("token_usage"))
                req_logger.set_intent(intent)
                req_logger.set_retrieval_stats(len(retrieved), len(reranked))
                req_logger.set_retrieval_metrics(retrieval_stats)
                if token_usage:
                    req_logger.set_token_usage(
                        prompt_tokens=token_usage.get("prompt", 0),
                        completion_tokens=token_usage.get("completion", 0),
                    )
                if isinstance(eval_result, dict):
                    evaluation_pass = eval_result.get("passed")
                    if evaluation_pass is None and "is_satisfactory" in eval_result:
                        evaluation_pass = eval_result.get("is_satisfactory")
                    if isinstance(evaluation_pass, bool):
                        req_logger.set_evaluation_result(evaluation_pass)
                req_logger.set_skill_runtime(
                    active_skill=metadata.get("active_skill", ""),
                    capability_status=metadata.get("skill_capability", {}),
                    quality_signals=metadata.get("quality_signals", {}),
                )
                req_logger.set_response_metadata(
                    response_type=final_state.get("response_type", "recommendation"),
                    history_message_count=len(prev_messages),
                    interaction_id=interaction_id,
                    feedback_logged=feedback_logged,
                )
                req_logger.finalize()

            return ChatResponse(
                response=response_text,
                request_id=request_id,
                intent=intent,
                latency_ms=round(latency_ms, 2),
                token_usage=final_state.get("token_usage") or None,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"聊天请求失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent 调用失败: {str(e)}")

    @application.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """SSE 流式聊天端点

        调用 graph.astream_events()，逐 chunk 返回。
        """
        request_id = str(uuid.uuid4())

        if _compiled_graph is None:
            raise HTTPException(status_code=503, detail="Graph 尚未就绪")

        initial_state = _build_initial_state(request, request_id)
        config = _get_langfuse_config(request_id, request)

        from .streaming import stream_graph_events

        async def event_generator():
            async for sse_line in stream_graph_events(
                _compiled_graph, initial_state, config=config
            ):
                yield sse_line

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── 全局异常处理 ──────────────────────────────────────────────────────

    @application.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"未捕获异常: {exc}", exc_info=True)
        return {"error": "内部服务器错误", "detail": str(exc)}

    return application


app = create_app()
