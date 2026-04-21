# -*- coding: utf-8 -*-
"""
结构化请求日志

为每次 API 请求记录完整的结构化日志（JSON 格式），
便于线上延迟分析、token 消耗追踪和质量监控。

复用模块:
- src/utils/logger.py::get_logger()
"""

import time
import json
from datetime import datetime, timezone
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RequestLogger:
    """结构化请求日志器

    为每次 API 请求记录完整的结构化日志，包括：
    - request_id: 请求唯一标识
    - intent: 意图分类结果
    - retrieval_count / reranked_count: 检索/精排文档数
    - phase_latency_ms: 各阶段延迟
    - total_latency_ms: 总延迟
    - token_usage: LLM token 消耗
    - evaluation_pass: 评估是否通过
    """

    def __init__(self, request_id: str):
        """
        初始化请求日志

        Args:
            request_id: 请求唯一标识（通常为 UUID）
        """
        self.request_id = request_id
        self._start_time = time.monotonic()
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._phase_timings: dict[str, dict] = {}
        self._intent = ""
        self._retrieval_count = 0
        self._reranked_count = 0
        self._retrieval_metrics: dict = {}
        self._token_usage: dict = {}
        self._evaluation_pass: Optional[bool] = None
        self._response_type = ""
        self._active_skill = ""
        self._skill_capability: dict = {}
        self._quality_signals: dict = {}
        self._history_message_count = 0
        self._interaction_id = ""
        self._feedback_logged = False

    def start_phase(self, phase: str) -> None:
        """标记某阶段开始

        Args:
            phase: 阶段名称（如 router / retriever / reranker / generator / evaluator）
        """
        self._phase_timings[phase] = {"start": time.monotonic()}

    def end_phase(self, phase: str) -> None:
        """标记某阶段结束

        Args:
            phase: 阶段名称
        """
        if phase in self._phase_timings and "start" in self._phase_timings[phase]:
            elapsed = (time.monotonic() - self._phase_timings[phase]["start"]) * 1000
            self._phase_timings[phase]["elapsed_ms"] = round(elapsed, 2)

    def set_intent(self, intent: str) -> None:
        """记录意图分类结果

        Args:
            intent: 意图名称
        """
        self._intent = intent

    def set_retrieval_stats(self, retrieved: int, reranked: int) -> None:
        """记录检索/精排文档数

        Args:
            retrieved: 检索返回的文档数
            reranked: 精排后保留的文档数
        """
        self._retrieval_count = retrieved
        self._reranked_count = reranked

    def set_retrieval_metrics(self, metrics: dict) -> None:
        """记录检索约束相关统计。"""
        self._retrieval_metrics = metrics or {}

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录 LLM token 使用量

        Args:
            prompt_tokens: Prompt token 数
            completion_tokens: Completion token 数
        """
        self._token_usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        }

    def set_evaluation_result(self, passed: bool) -> None:
        """记录评估结果

        Args:
            passed: 评估是否通过
        """
        self._evaluation_pass = passed

    def set_skill_runtime(
        self,
        active_skill: str,
        capability_status: Optional[dict] = None,
        quality_signals: Optional[dict] = None,
    ) -> None:
        self._active_skill = active_skill or ""
        self._skill_capability = dict(capability_status or {})
        self._quality_signals = dict(quality_signals or {})

    def set_response_metadata(
        self,
        response_type: str,
        history_message_count: int,
        interaction_id: str = "",
        feedback_logged: bool = False,
    ) -> None:
        """记录响应元数据，便于真实请求链路排障。"""
        self._response_type = response_type or ""
        self._history_message_count = history_message_count
        self._interaction_id = interaction_id or ""
        self._feedback_logged = feedback_logged

    def finalize(self) -> dict:
        """计算 total_latency_ms，输出完整结构化日志 dict 并写入 logger

        Returns:
            结构化日志字典
        """
        total_latency = (time.monotonic() - self._start_time) * 1000

        phase_latency_ms = {}
        for phase, timing in self._phase_timings.items():
            if "elapsed_ms" in timing:
                phase_latency_ms[phase] = timing["elapsed_ms"]

        log_data = {
            "request_id": self.request_id,
            "timestamp": self._timestamp,
            "intent": self._intent,
            "active_skill": self._active_skill,
            "response_type": self._response_type,
            "retrieval_count": self._retrieval_count,
            "reranked_count": self._reranked_count,
            "retrieval_metrics": self._retrieval_metrics,
            "skill_capability": self._skill_capability,
            "quality_signals": self._quality_signals,
            "history_message_count": self._history_message_count,
            "interaction_id": self._interaction_id,
            "feedback_logged": self._feedback_logged,
            "phase_latency_ms": phase_latency_ms,
            "total_latency_ms": round(total_latency, 2),
            "token_usage": self._token_usage,
            "evaluation_pass": self._evaluation_pass,
        }

        logger.info(f"[StructuredLog] {json.dumps(log_data, ensure_ascii=False)}")
        return log_data
