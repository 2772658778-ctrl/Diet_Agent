# -*- coding: utf-8 -*-
"""
分层记忆管理器

实现三层记忆架构：
- WorkingMemory: 工作记忆（最近 N 轮对话，内存存储）
- EpisodicMemory: 情景记忆（会话摘要，LLM 摘要 + 内存缓存）
- SemanticMemory: 语义记忆（用户画像/偏好，PostgreSQL 持久化）
"""

from typing import List, Dict, Any, Optional
from collections import deque

from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()


# ── WorkingMemory ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """工作记忆：最近 N 轮对话（内存存储）

    使用 deque(maxlen=window_size * 2) 保存最近的 user+assistant 消息对。
    无持久化，进程退出即清空。
    """

    def __init__(self, window_size: int = 5) -> None:
        """初始化工作记忆

        Args:
            window_size: 保留的对话轮数（每轮包含 user+assistant 两条消息）
        """
        self.window_size = window_size
        self._messages: deque = deque(maxlen=window_size * 2)
        logger.debug(f"WorkingMemory 初始化: window_size={window_size}")

    def add(self, role: str, content: str) -> None:
        """添加一条消息到工作记忆

        Args:
            role: 消息角色（user / assistant / system）
            content: 消息内容
        """
        self._messages.append({"role": role, "content": content})
        logger.debug(f"WorkingMemory 添加消息: role={role}, 内容长度={len(content)}")

    def get_messages(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """获取最近消息列表

        Args:
            max_tokens: 可选的 token 上限（超出则从最旧消息开始裁剪）

        Returns:
            消息列表 [{"role": "...", "content": "..."}]
        """
        messages = list(self._messages)

        if max_tokens is None or not messages:
            return messages

        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            total = 0
            kept: List[Dict[str, str]] = []
            for msg in reversed(messages):
                tokens = len(enc.encode(msg.get("content", ""))) + 4
                if total + tokens > max_tokens:
                    break
                kept.append(msg)
                total += tokens
            return list(reversed(kept))
        except Exception:
            return messages

    def clear(self) -> None:
        """清空工作记忆"""
        self._messages.clear()
        logger.debug("WorkingMemory 已清空")

    def __len__(self) -> int:
        return len(self._messages)


# ── EpisodicMemory ────────────────────────────────────────────────────────────

class EpisodicMemory:
    """情景记忆：会话摘要（LLM 摘要 + 内存缓存）

    每隔 update_threshold 条新消息触发一次 LLM 摘要更新。
    摘要缓存在内存中，一次会话有效。
    避免频繁调用 LLM：只有当新消息积累超过阈值时才重新生成摘要。
    """

    def __init__(
        self,
        compressor: Any,
        update_threshold: int = 6,
    ) -> None:
        """初始化情景记忆

        Args:
            compressor: ContextCompressor 实例（用于调用 _generate_summary）
            update_threshold: 触发摘要更新的新消息数阈值
        """
        self._compressor = compressor
        self._update_threshold = update_threshold
        self._summary: str = ""
        self._pending_messages: List[Dict[str, str]] = []
        logger.debug(
            f"EpisodicMemory 初始化: update_threshold={update_threshold}, "
            f"compressor={'已就绪' if compressor else '无（摘要不可用）'}"
        )

    def update(self, messages: List[Dict[str, str]]) -> None:
        """用当前对话更新情景摘要

        仅当 pending 消息数达到阈值时触发 LLM 摘要，避免频繁调用。

        Args:
            messages: 本次新增的消息列表
        """
        self._pending_messages.extend(messages)

        if len(self._pending_messages) < self._update_threshold:
            logger.debug(
                f"EpisodicMemory: 待更新消息数 {len(self._pending_messages)} "
                f"< 阈值 {self._update_threshold}，跳过摘要生成"
            )
            return

        self._regenerate_summary()

    def _regenerate_summary(self) -> None:
        """调用 ContextCompressor._generate_summary 生成/更新摘要"""
        if self._compressor is None:
            logger.warning("EpisodicMemory: compressor 未初始化，无法生成摘要")
            self._pending_messages = []
            return

        try:
            if self._summary:
                combined = [
                    {"role": "system", "content": f"之前的会话摘要：{self._summary}"}
                ] + self._pending_messages
                new_summary = self._compressor._generate_summary(combined)
            else:
                new_summary = self._compressor._generate_summary(self._pending_messages)

            self._summary = new_summary
            self._pending_messages = []
            logger.info(f"EpisodicMemory 摘要已更新，长度: {len(self._summary)}")
        except Exception as e:
            logger.warning(f"EpisodicMemory 摘要生成失败，保留旧摘要: {e}")

    def get_summary(self) -> str:
        """获取当前会话摘要

        Returns:
            摘要字符串，尚未生成时返回空字符串
        """
        return self._summary

    def clear(self) -> None:
        """清空本次会话摘要和待处理消息"""
        self._summary = ""
        self._pending_messages = []
        logger.debug("EpisodicMemory 已清空")


# ── SemanticMemory ────────────────────────────────────────────────────────────

class SemanticMemory:
    """语义记忆：用户画像/偏好（PostgreSQL 持久化）

    从 PostgreSQL 加载用户的基础信息（get_user）和偏好（get_user_preferences），
    合并为完整画像。无数据库连接时降级返回空画像，不抛异常。
    """

    def __init__(
        self,
        postgres_client: Optional[Any] = None,
    ) -> None:
        """初始化语义记忆

        Args:
            postgres_client: PostgreSQLClient 实例，为 None 时降级为内存模式
        """
        self._db = postgres_client
        logger.debug(
            f"SemanticMemory 初始化: db={'已连接' if postgres_client else '无连接（降级模式）'}"
        )

    def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        """从 PostgreSQL 加载用户画像 + 偏好

        Args:
            user_id: 用户 ID

        Returns:
            合并的用户画像字典，无数据或出错时返回空字典
        """
        if not self._db or not user_id:
            logger.debug("SemanticMemory: 无数据库连接或 user_id 为空，返回空画像")
            return {}

        try:
            user_info = self._db.get_user(user_id) or {}
            preferences = self._db.get_user_preferences(user_id) or {}
            profile = {**user_info, **preferences}
            logger.info(f"SemanticMemory: 加载用户 '{user_id}' 画像，字段数: {len(profile)}")
            return profile
        except Exception as e:
            logger.warning(f"SemanticMemory: 加载用户画像失败，返回空画像: {e}")
            return {}

    def update_preferences(
        self,
        user_id: str,
        new_prefs: Dict[str, Any],
    ) -> None:
        """更新用户偏好

        Args:
            user_id: 用户 ID
            new_prefs: 新偏好字典（键值对将被合并到数据库记录）
        """
        if not self._db or not user_id:
            logger.warning("SemanticMemory: 无数据库连接，无法更新偏好")
            return

        try:
            self._db.set_user_preferences(user_id, **new_prefs)
            logger.info(
                f"SemanticMemory: 更新用户 '{user_id}' 偏好，字段: {list(new_prefs.keys())}"
            )
        except Exception as e:
            logger.error(f"SemanticMemory: 更新偏好失败: {e}")

    def format_profile(self, profile: Dict[str, Any]) -> str:
        """将用户画像格式化为文本（供 prompt 注入）

        示例输出：用户画像：健康目标=减肥，口味偏好=清淡/酸甜，过敏=海鲜

        Args:
            profile: 用户画像字典

        Returns:
            格式化的用户画像文本，空画像返回空字符串
        """
        if not profile:
            return ""

        parts: List[str] = []

        health_goal = profile.get("health_goal")
        if health_goal:
            parts.append(f"健康目标={health_goal}")

        fav_tastes = profile.get("favorite_tastes", [])
        if fav_tastes:
            parts.append(f"口味偏好={'/'.join(str(t) for t in fav_tastes)}")

        allergies = profile.get("allergies", [])
        if allergies:
            parts.append(f"过敏={'/'.join(str(a) for a in allergies)}")

        disliked = profile.get("disliked_ingredients", [])
        if disliked:
            parts.append(f"不喜欢={'/'.join(str(d) for d in disliked)}")

        max_time = profile.get("max_cooking_time")
        if max_time:
            parts.append(f"时间限制={max_time}分钟")

        target_cal = profile.get("target_calories")
        if target_cal:
            parts.append(f"热量目标={target_cal}卡")

        if not parts:
            return ""

        formatted = "用户画像：" + "，".join(parts)
        logger.debug(f"SemanticMemory: 格式化用户画像，长度: {len(formatted)}")
        return formatted


# ── TieredMemoryManager ───────────────────────────────────────────────────────

class TieredMemoryManager:
    """三层记忆管理器

    统一管理 WorkingMemory、EpisodicMemory、SemanticMemory，
    提供统一的 add_message / get_context / clear_session 接口。
    """

    def __init__(
        self,
        window_size: int = 5,
        postgres_client: Optional[Any] = None,
    ) -> None:
        """初始化三层记忆管理器

        Args:
            window_size: WorkingMemory 窗口大小（对话轮数）
            postgres_client: PostgreSQLClient 实例，为 None 时 SemanticMemory 降级
        """
        compressor = None
        try:
            from .context_compressor import ContextCompressor
            compressor = ContextCompressor(window_size=window_size)
        except Exception as e:
            logger.warning(
                f"ContextCompressor 初始化失败，EpisodicMemory 将无法生成摘要: {e}"
            )

        update_threshold = getattr(settings, "context_episodic_update_threshold", 6)

        self.working = WorkingMemory(window_size=window_size)
        self.episodic = EpisodicMemory(
            compressor=compressor,
            update_threshold=update_threshold,
        )
        self.semantic = SemanticMemory(postgres_client=postgres_client)

        logger.info(
            f"TieredMemoryManager 初始化: window_size={window_size}, "
            f"db={'已连接' if postgres_client else '无连接'}"
        )

    def add_message(self, role: str, content: str) -> None:
        """添加消息到工作记忆，同时触发情景记忆更新

        Args:
            role: 消息角色（user / assistant）
            content: 消息内容
        """
        self.working.add(role, content)
        self.episodic.update([{"role": role, "content": content}])
        logger.debug(f"TieredMemoryManager: 添加消息 role={role}")

    def get_context(
        self,
        user_id: Optional[str],
        token_budget: int,
    ) -> Dict[str, Any]:
        """从三层记忆中按优先级提取上下文

        优先级：语义记忆（固定）> 工作记忆 > 情景记忆

        Args:
            user_id: 用户 ID（用于加载语义记忆）
            token_budget: 总 token 预算

        Returns:
            {
                "user_profile": str,       # 语义记忆格式化文本
                "working_messages": list,  # 工作记忆消息列表
                "episodic_summary": str,   # 情景记忆摘要
                "token_usage": dict        # 各层 token 消耗统计
            }
        """
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

            def count_tokens(text: str) -> int:
                return len(enc.encode(text)) if text else 0
        except Exception:
            def count_tokens(text: str) -> int:  # type: ignore[misc]
                return len(text) // 4

        remaining_budget = token_budget
        token_usage: Dict[str, int] = {}

        # 1. 语义记忆（用户画像）
        raw_profile = self.semantic.load_user_profile(user_id or "")
        profile_text = self.semantic.format_profile(raw_profile)
        profile_tokens = count_tokens(profile_text)
        token_usage["profile"] = profile_tokens
        remaining_budget -= profile_tokens

        # 2. 工作记忆（最近对话）
        working_budget = max(0, remaining_budget // 2)
        working_msgs = self.working.get_messages(max_tokens=working_budget)
        working_tokens = sum(count_tokens(m.get("content", "")) + 4 for m in working_msgs)
        token_usage["working"] = working_tokens
        remaining_budget -= working_tokens

        # 3. 情景记忆（会话摘要，用剩余预算）
        episodic_summary = self.episodic.get_summary()
        episodic_tokens = count_tokens(episodic_summary)
        if episodic_tokens > remaining_budget and remaining_budget > 0:
            chars_to_keep = remaining_budget * 4
            episodic_summary = episodic_summary[:chars_to_keep]
            episodic_tokens = count_tokens(episodic_summary)
        token_usage["episodic"] = episodic_tokens

        logger.info(
            f"TieredMemoryManager.get_context: profile={profile_tokens}, "
            f"working={working_tokens}, episodic={episodic_tokens} tokens"
        )

        return {
            "user_profile": profile_text,
            "working_messages": working_msgs,
            "episodic_summary": episodic_summary,
            "token_usage": token_usage,
        }

    def clear_session(self) -> None:
        """清空工作记忆和情景记忆（会话结束时调用）

        语义记忆（PostgreSQL）不受影响。
        """
        self.working.clear()
        self.episodic.clear()
        logger.info("TieredMemoryManager: 会话记忆已清空（语义记忆保留）")
