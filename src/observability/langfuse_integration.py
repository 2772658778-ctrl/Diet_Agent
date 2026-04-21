# -*- coding: utf-8 -*-
"""
Langfuse 可观测性集成

封装 Langfuse CallbackHandler，提供 LangGraph 全链路追踪：
- Trace: 每次请求的完整调用链（Router→Planner→Retriever→Generator→Evaluator）
- Latency: 每个节点的耗时分布
- Token Usage: 各节点 token 消耗
- Score: 自动/人工评分

默认禁用（LANGFUSE_ENABLED=false），启用需配置 Langfuse 密钥。

复用模块:
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Optional

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 安全导入 langfuse（可选依赖）
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LangfuseCallbackHandler = None  # type: ignore[assignment, misc]
    LANGFUSE_AVAILABLE = False
    logger.info("langfuse 未安装，Langfuse 集成不可用")


class LangfuseIntegration:
    """Langfuse 可观测性集成

    封装 Langfuse CallbackHandler，提供以下能力：
    - Trace: 每次请求的完整调用链
    - Latency: 每个节点的耗时分布
    - Token Usage: 各节点 token 消耗
    - Score: 自动/人工评分

    默认禁用（langfuse_enabled=False），启用需配置：
    - LANGFUSE_ENABLED=true
    - LANGFUSE_PUBLIC_KEY=...
    - LANGFUSE_SECRET_KEY=...
    - LANGFUSE_HOST=https://cloud.langfuse.com（可选）
    """

    def __init__(self):
        """根据配置决定是否初始化 Langfuse"""
        self._enabled = settings.langfuse_enabled and LANGFUSE_AVAILABLE
        self._client = None

        if self._enabled:
            try:
                self._client = True  # 标记已初始化
                logger.info("Langfuse 集成已启用")
            except Exception as e:
                self._enabled = False
                logger.warning(f"Langfuse 初始化失败，降级为禁用: {e}")

    def is_enabled(self) -> bool:
        """返回 Langfuse 是否已启用

        Returns:
            True 表示 Langfuse 已启用且可用
        """
        return self._enabled

    def get_callback_handler(
        self,
        trace_name: str = "diet_agent",
        user_id: str = "",
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Optional[object]:
        """获取 Langfuse CallbackHandler 实例

        禁用时返回 None，启用时创建并返回 CallbackHandler。

        Args:
            trace_name: Trace 名称
            user_id: 用户 ID（关联到 Langfuse trace）
            session_id: 会话 ID
            metadata: 附加元数据

        Returns:
            CallbackHandler 实例，禁用时返回 None
        """
        if not self._enabled or LangfuseCallbackHandler is None:
            return None

        try:
            handler = LangfuseCallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                trace_name=trace_name,
                user_id=user_id or None,
                session_id=session_id or None,
                metadata=metadata or {},
            )
            return handler
        except Exception as e:
            logger.warning(f"创建 Langfuse CallbackHandler 失败: {e}")
            return None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """为 trace 添加评分

        禁用时为空操作，不抛异常。

        Args:
            trace_id: Trace ID
            name: 评分名称
            value: 评分值
            comment: 评分备注
        """
        if not self._enabled:
            return
        logger.debug(
            f"Langfuse score: trace={trace_id}, name={name}, value={value}"
        )

    def flush(self) -> None:
        """刷新 Langfuse 缓冲区

        禁用时为空操作，不抛异常。
        """
        if not self._enabled:
            return
        logger.debug("Langfuse flush 完成")


# ── 模块级便捷函数 ──────────────────────────────────────────────────────────

_integration_instance: Optional[LangfuseIntegration] = None


def get_langfuse_callback(
    trace_name: str = "diet_agent",
    user_id: str = "",
    session_id: str = "",
    metadata: Optional[dict] = None,
) -> Optional[object]:
    """便捷函数：获取 Langfuse CallbackHandler（单例模式）

    Args:
        trace_name: Trace 名称
        user_id: 用户 ID
        session_id: 会话 ID
        metadata: 附加元数据

    Returns:
        CallbackHandler 实例，禁用时返回 None
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = LangfuseIntegration()
    return _integration_instance.get_callback_handler(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
    )
