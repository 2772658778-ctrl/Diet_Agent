# -*- coding: utf-8 -*-
"""
动态上下文组装器

根据 token 预算，按优先级将 system instruction、用户画像、
压缩历史、检索上下文和当前查询组装为最终 prompt。
"""

from typing import List, Dict, Any, Optional

import tiktoken

from diet_agent.runtime import get_skill_runtime_policy

from ..config import get_settings
from ..database.postgres_client import get_postgres_client
from ..utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()

# 组装器内部默认 system prompt（用于 InstructionHierarchy 不可用时的 fallback）
_DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的智能饮食助手，帮助用户找到合适的食谱、分析营养成分、检查食材搭配。"
    "请基于提供的参考文档生成回复，不要编造信息，回复简洁、专业、友好。"
)


class ContextAssembler:
    """动态上下文组装器

    将多种信息源按优先级和 token 预算组装成最终 prompt。

    Token 预算分配（总预算如 8K tokens）：
    ┌────────────────────────────────────┐
    │ System Instruction    (~500 tokens)│  ← 固定，来自 InstructionHierarchy
    │ User Profile/Prefs   (~200 tokens)│  ← 从 SemanticMemory（PostgreSQL）加载
    │ Compressed History   (~1000 tokens)│  ← 从工作记忆获取
    │ Retrieved Context    (~2000 tokens)│  ← RAG 检索结果（外部传入）
    │ Current Query         (~100 tokens)│  ← 用户输入
    │ [Buffer for Response] (剩余 tokens)│  ← 留给生成
    └────────────────────────────────────┘

    优先级（高 → 低，token 不足时从低优先级开始裁剪）：
    1. System Instruction（固定，不裁剪）
    2. Current Query（固定，不裁剪）
    3. Retrieved Docs（可裁剪）
    4. User Profile（可裁剪）
    5. History（最低优先级，可压缩/丢弃）
    """

    def __init__(
        self,
        token_budget: int = 8000,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        """初始化动态上下文组装器

        Args:
            token_budget: 总 token 预算（包含生成预留）
            model: 用于 token 计数的模型名称
        """
        self.token_budget = token_budget
        self.model = model

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self._instruction_hierarchy: Optional[Any] = None
        self._memory_manager: Optional[Any] = None

        logger.info(
            f"ContextAssembler 初始化: token_budget={token_budget}, model={model}"
        )

    # ── 延迟初始化辅助模块 ──────────────────────────────────────────────────────

    def _get_instruction_hierarchy(self) -> Optional[Any]:
        """延迟初始化 InstructionHierarchy，初始化失败时返回 None"""
        if self._instruction_hierarchy is None:
            try:
                from .instruction_hierarchy import InstructionHierarchy
                self._instruction_hierarchy = InstructionHierarchy()
            except Exception as e:
                logger.warning(f"InstructionHierarchy 初始化失败: {e}")
        return self._instruction_hierarchy

    def _get_memory_manager(self) -> Optional[Any]:
        """延迟初始化 TieredMemoryManager，初始化失败时返回 None"""
        if self._memory_manager is None:
            try:
                from .memory_manager import TieredMemoryManager
                self._memory_manager = TieredMemoryManager()
            except Exception as e:
                logger.warning(f"TieredMemoryManager 初始化失败: {e}")
        return self._memory_manager

    # ── Token 工具方法 ─────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """计算字符串的 token 数

        Args:
            text: 待计算的字符串

        Returns:
            token 数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """将文本截断到指定 token 数

        Args:
            text: 原始文本
            max_tokens: 最大 token 数

        Returns:
            截断后的文本（保证 token 数 ≤ max_tokens）
        """
        if not text or max_tokens <= 0:
            return ""

        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def _allocate_budget(self, total_budget: int) -> Dict[str, int]:
        """计算各部分的 token 分配

        预留 response_reserve_ratio（默认 50%）给生成，
        其余按比例分配给各上下文部分。

        Args:
            total_budget: 总 token 预算

        Returns:
            各部分 token 分配字典
        """
        reserve_ratio = getattr(settings, "context_response_reserve_ratio", 0.5)
        runtime_policy = None
        if isinstance(total_budget, dict):
            runtime_policy = total_budget
            total_budget = int(runtime_policy.get("total_budget") or self.token_budget)

        usable = max(1, int(total_budget * (1 - reserve_ratio)))

        budget_weights = {
            "system": 0.15,
            "user_profile": 0.06,
            "history": 0.30,
            "retrieved_docs": 0.60,
            "query": 0.06,
        }
        token_caps = {
            "system": 500,
            "user_profile": 200,
            "history": 1000,
            "retrieved_docs": 2000,
            "query": 200,
        }

        if isinstance(runtime_policy, dict):
            raw_weights = runtime_policy.get("budget_weights") or {}
            if isinstance(raw_weights, dict):
                for key, value in raw_weights.items():
                    if key in budget_weights:
                        try:
                            numeric_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if numeric_value > 0:
                            budget_weights[key] = numeric_value

            raw_caps = runtime_policy.get("token_caps") or {}
            if isinstance(raw_caps, dict):
                for key, value in raw_caps.items():
                    if key in token_caps:
                        try:
                            numeric_cap = int(value)
                        except (TypeError, ValueError):
                            continue
                        if numeric_cap > 0:
                            token_caps[key] = numeric_cap

        raw = {
            "system": min(token_caps["system"], int(usable * budget_weights["system"])),
            "user_profile": min(token_caps["user_profile"], int(usable * budget_weights["user_profile"])),
            "history": min(token_caps["history"], int(usable * budget_weights["history"])),
            "retrieved_docs": min(token_caps["retrieved_docs"], int(usable * budget_weights["retrieved_docs"])),
            "query": min(token_caps["query"], int(usable * budget_weights["query"])),
        }

        total_allocated = sum(raw.values())
        if total_allocated > usable:
            scale = usable / total_allocated
            raw = {k: max(20, int(v * scale)) for k, v in raw.items()}

        logger.debug(f"token 预算分配: {raw}")
        return raw

    def _format_few_shot_messages(
        self,
        examples: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        if not examples or max_tokens <= 0:
            return []

        formatted_messages: List[Dict[str, str]] = []
        token_used = 0
        for item in examples:
            user_text = str(item.get("user") or "").strip()
            assistant_text = str(item.get("assistant") or "").strip()
            if not user_text or not assistant_text:
                continue

            pair_tokens = self._count_tokens(user_text) + self._count_tokens(assistant_text) + 8
            if token_used + pair_tokens > max_tokens:
                break

            formatted_messages.append({"role": "user", "content": user_text})
            formatted_messages.append({"role": "assistant", "content": assistant_text})
            token_used += pair_tokens

        return formatted_messages

    # ── 格式化工具方法 ─────────────────────────────────────────────────────────

    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """将用户画像格式化为文本

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

        recent_feedback_summary = profile.get("recent_feedback_summary")
        if recent_feedback_summary:
            parts.append(f"最近反馈={recent_feedback_summary}")

        if not parts:
            return ""

        return "用户画像：" + "，".join(parts)

    def _format_retrieved_docs(
        self,
        docs: List[Dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """将检索文档格式化并截断至 token 限制

        Args:
            docs: 检索文档列表
            max_tokens: 最大 token 数

        Returns:
            格式化并截断后的字符串，无文档时返回空字符串
        """
        if not docs:
            return ""

        parts: List[str] = []
        token_used = 0
        raw_token_total = 0

        for i, doc in enumerate(docs, 1):
            name = doc.get("name", doc.get("text", "未知"))
            description = doc.get("description", "")
            cuisine = doc.get("cuisine", "")
            time_val = doc.get("time", "")
            calories = doc.get("calories", "")

            part = f"【参考{i}】{name}"
            if description:
                part += f"：{description}"
            if cuisine:
                part += f"（{cuisine}菜）"
            if time_val:
                part += f"，{time_val}分钟"
            if calories:
                part += f"，{calories}卡"

            part_tokens = self._count_tokens(part)
            raw_token_total += part_tokens
            if token_used + part_tokens > max_tokens:
                continue

            parts.append(part)
            token_used += part_tokens

        return "\n\n".join(parts)

    def assemble(
        self,
        query: str,
        user_id: Optional[str],
        history: List[Dict[str, str]],
        retrieved_docs: List[Dict[str, Any]],
        intent: str = "recipe_search",
        skill_name: Optional[str] = None,
        token_budget: Optional[int] = None,
        stable_user_preferences: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """核心方法：按优先级组装消息列表

        Args:
            query: 当前用户查询
            user_id: 用户 ID（用于加载语义记忆）
            history: 对话历史 [{"role": "...", "content": "..."}]
            retrieved_docs: RAG 检索文档列表
            intent: 用户意图（用于选择 system instruction）
            skill_name: 技能名称（用于选择 system instruction）
            token_budget: 可选的覆盖 token 预算

        Returns:
            消息列表 [{"role": "system/user/assistant", "content": "..."}]，
            可直接传入 LLM.invoke()
        """
        budget = token_budget if token_budget is not None else self.token_budget
        runtime_policy = get_skill_runtime_policy(skill_name)
        context_policy = runtime_policy.context_assembly
        allocation = self._allocate_budget(
            {
                "total_budget": budget,
                "budget_weights": dict(context_policy.budget_weights),
                "token_caps": dict(context_policy.token_caps),
            }
        )

        messages: List[Dict[str, str]] = []
        token_log: Dict[str, int] = {}

        # 1. System Instruction（最高优先级，固定）
        system_content = ""
        try:
            hierarchy = self._get_instruction_hierarchy()
            if hierarchy:
                system_content = hierarchy.get_system_instruction(intent, skill_name=skill_name)
        except Exception as e:
            logger.warning(f"获取 system instruction 失败，使用默认指令: {e}")

        if not system_content:
            system_content = _DEFAULT_SYSTEM_PROMPT

        if context_policy.system_suffix_lines:
            system_content = "\n\n".join([
                system_content,
                *context_policy.system_suffix_lines,
            ])

        system_content = self._truncate_to_budget(system_content, allocation["system"])

        # 2. User Profile（语义记忆，注入 system instruction 末尾）
        user_profile_text = ""
        if user_id or stable_user_preferences:
            try:
                raw_profile = dict(stable_user_preferences or {})
                if user_id and not raw_profile:
                    memory_mgr = self._get_memory_manager()
                    if memory_mgr:
                        raw_profile = memory_mgr.semantic.load_user_profile(user_id) or {}
                try:
                    if user_id:
                        postgres_client = get_postgres_client()
                        feedbacks = postgres_client.get_user_feedbacks(user_id, limit=3) if postgres_client else []
                        summary_parts = []
                        for item in feedbacks:
                            recipe_id = str(item.get("recipe_id") or "").strip()
                            rating = item.get("rating")
                            comment = str(item.get("comment") or "").strip()
                            if recipe_id and rating:
                                part = f"{recipe_id}:{rating}分"
                                if comment:
                                    part += f"({comment})"
                                summary_parts.append(part)
                        if summary_parts:
                            raw_profile = {
                                **(raw_profile or {}),
                                "recent_feedback_summary": "；".join(summary_parts),
                            }
                except Exception as e:
                    logger.warning(f"加载最近反馈失败: {e}")
                user_profile_text = self._format_user_profile(raw_profile)
                user_profile_text = self._truncate_to_budget(
                    user_profile_text, allocation["user_profile"]
                )
            except Exception as e:
                logger.warning(f"加载用户画像失败: {e}")

        if user_profile_text:
            system_content = f"{system_content}\n\n## 当前用户信息\n{user_profile_text}"

        messages.append({"role": "system", "content": system_content})
        token_log["system"] = self._count_tokens(system_content)

        few_shot_messages = self._format_few_shot_messages(
            [dict(item) for item in context_policy.few_shot_examples],
            min(allocation["history"], context_policy.few_shot_budget or allocation["history"]),
        )
        few_shot_tokens = 0
        if few_shot_messages:
            messages.extend(few_shot_messages)
            few_shot_tokens = sum(
                self._count_tokens(message.get("content", "")) + 4
                for message in few_shot_messages
            )
        token_log["few_shot"] = few_shot_tokens

        # 3. History（最低优先级，从最新消息往前取到 budget 耗尽）
        if history:
            history_budget = max(0, allocation["history"] - few_shot_tokens)
            history_tokens = 0
            history_msgs: List[Dict[str, str]] = []

            for msg in reversed(history):
                t = self._count_tokens(msg.get("content", "")) + 4
                if history_tokens + t > history_budget:
                    break
                history_msgs.append(msg)
                history_tokens += t

            history_msgs = list(reversed(history_msgs))
            messages.extend(history_msgs)
            token_log["history"] = history_tokens
        else:
            token_log["history"] = 0

        # 4. Retrieved Docs + Current Query（组装为最终用户消息）
        raw_doc_tokens = self._count_tokens(str(retrieved_docs)) if retrieved_docs else 0
        docs_text = self._format_retrieved_docs(
            retrieved_docs, allocation["retrieved_docs"]
        )
        token_log["retrieved_docs"] = self._count_tokens(docs_text)
        token_log["raw_doc_tokens"] = raw_doc_tokens

        if docs_text:
            user_message = (
                f"{context_policy.docs_heading}：\n{docs_text}\n\n"
                f"{context_policy.query_heading}：{query}"
            )
        else:
            user_message = query

        user_message_budget = allocation["query"] + allocation["retrieved_docs"]
        user_message = self._truncate_to_budget(user_message, user_message_budget)
        messages.append({"role": "user", "content": user_message})
        token_log["query"] = self._count_tokens(query)

        total_tokens = sum(token_log.values())
        logger.info(
            f"ContextAssembler.assemble 完成: messages={len(messages)}, "
            f"token_log={token_log}, total={total_tokens}"
        )

        return messages
