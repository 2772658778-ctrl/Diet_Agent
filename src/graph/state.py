"""
LangGraph 全局 State 定义

定义 DietAgentState TypedDict，作为 LangGraph StateGraph 的共享状态。
所有节点通过读写 State 中的字段进行数据传递。
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from .schemas import build_default_recipe_query_constraints, normalize_extracted_params


class DietAgentState(TypedDict):
    """饮食 Agent 全局状态
    
    LangGraph StateGraph 的共享状态定义，各节点通过读写此 State 进行通信。
    
    Attributes:
        messages: 对话历史，使用 add_messages reducer 自动追加消息
        intent: 路由结果（recipe_search / nutrition_query / ingredient_check / chitchat）
        confidence: 路由置信度 (0.0 ~ 1.0)
        extracted_params: 从用户查询中提取的参数
        plan: 规划步骤列表
        retrieved_docs: 检索结果列表
        reranked_docs: 精排结果列表
        response: 生成的回复文本
        evaluation: 质量评估结果
        current_step: 当前执行步骤索引
        retry_count: 重试计数（Evaluator 不通过时递增）
        query_complexity: 查询复杂度 (simple/complex/ambiguous)，Phase 1 新增
        transformed_queries: 查询变换结果列表，Phase 1 新增
        self_rag_judgements: Self-RAG 各阶段判断结果，Phase 1 新增
        retrieval_strategy: 实际使用的检索策略，Phase 1 新增
        eval_metrics: 评测指标结果，Phase 2 新增（仅 benchmark 模式使用）
    """
    messages: Annotated[list, add_messages]
    intent: str
    confidence: float
    extracted_params: dict
    plan: list[str]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    response: str
    evaluation: dict
    current_step: int
    retry_count: int
    retrieval_stats: dict
    clarification_needed: bool
    clarification_question: str
    missing_slots: list[str]
    response_type: str
    # Phase 1 新增
    query_complexity: str
    query_features: dict
    transformed_queries: list[str]
    self_rag_judgements: dict
    retrieval_strategy: str
    # Phase 2 新增（可选，仅 benchmark 模式使用）
    eval_metrics: dict
    skip_graph_eval: bool
    # Phase 3 新增
    user_id: str                   # 用户 ID（用于加载语义记忆）
    assembled_context: list        # 组装后的上下文消息列表
    active_skill: str              # 当前激活的 Skill 名称
    memory_stats: dict             # 记忆使用统计 {"profile_tokens": ..., "working_tokens": ..., "episodic_tokens": ...}
    goal_type: str                 # planner 判定的本轮目标类型
    planner_next_action: str       # planner 判定的下一步动作
    inherit_followup_direction: bool  # 是否继承上一轮推荐方向
    followup_mode: str            # follow-up 模式（如 anchored_followup）
    followup_anchor_names: list[str]  # follow-up 推荐锚点名称
    recent_feedback_signals: dict  # 最近反馈信号（用于 PostgreSQL 不可用时的最小反馈闭环）
    recent_recommended_recipes: list[dict]  # 最近一轮推荐结果锚点（用于 follow-up 推荐）
    # Phase 4 新增
    request_id: str                # 请求唯一标识（UUID）
    phase_timings: dict            # 各阶段耗时记录 {"router": 120, "retriever": 200, ...}
    token_usage: dict              # LLM token 使用统计 {"prompt": ..., "completion": ...}
    stable_user_preferences: dict  # 稳定用户偏好（用于 planner/retriever/generator 的统一 memory 入口）
    applied_stable_preference_keys: list[str]  # 本轮被 stable preference 补足并真正生效的约束键


STABLE_PREFERENCE_CONSTRAINT_KEYS = (
    "allergies",
    "disliked_ingredients",
    "max_cooking_time",
    "health_goal",
    "meal_type",
)


def normalize_followup_contract(followup_mode: str, followup_anchor_names: list[str] | None = None) -> tuple[str, list[str]]:
    normalized_mode = str(followup_mode or "").strip()
    if normalized_mode not in {"anchored_followup", "generic_followup"}:
        return "", []

    if normalized_mode != "anchored_followup":
        return normalized_mode, []

    normalized_anchor_names: list[str] = []
    seen_names: set[str] = set()
    for name in followup_anchor_names or []:
        normalized_name = str(name).strip()
        if not normalized_name or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        normalized_anchor_names.append(normalized_name)

    return normalized_mode, normalized_anchor_names


def normalize_stable_user_preferences(stable_user_preferences: dict | None) -> dict:
    if not isinstance(stable_user_preferences, dict):
        return {}

    normalized_preferences = {
        str(key).strip(): value
        for key, value in stable_user_preferences.items()
        if str(key).strip()
    }
    overlapping_keys = [
        key for key in STABLE_PREFERENCE_CONSTRAINT_KEYS if key in normalized_preferences
    ]
    if not overlapping_keys:
        return normalized_preferences

    normalized_constraints = normalize_extracted_params({
        key: normalized_preferences.get(key)
        for key in overlapping_keys
    })
    for key in overlapping_keys:
        value = normalized_constraints.get(key)
        if value in (None, "", []):
            normalized_preferences.pop(key, None)
            continue
        normalized_preferences[key] = value

    return normalized_preferences


def apply_stable_preferences(extracted_params: dict | None, stable_user_preferences: dict | None) -> tuple[dict, list[str]]:
    resolved_params = normalize_extracted_params(extracted_params or {})
    normalized_preferences = normalize_stable_user_preferences(stable_user_preferences)
    applied_keys: list[str] = []

    for key in STABLE_PREFERENCE_CONSTRAINT_KEYS:
        candidate_value = normalized_preferences.get(key)
        if candidate_value in (None, "", []):
            continue

        current_value = resolved_params.get(key)
        if key in {"allergies", "disliked_ingredients"}:
            if current_value:
                continue
            resolved_params[key] = list(candidate_value)
        else:
            if current_value not in (None, "", []):
                continue
            resolved_params[key] = candidate_value

        applied_keys.append(key)

    return normalize_extracted_params(resolved_params), applied_keys


def build_initial_diet_agent_state(
    messages: list | None = None,
    user_id: str = "",
    request_id: str = "",
) -> dict[str, Any]:
    return {
        "messages": list(messages or []),
        "intent": "",
        "confidence": 0.0,
        "extracted_params": build_default_recipe_query_constraints(),
        "plan": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": "",
        "evaluation": {},
        "current_step": 0,
        "retry_count": 0,
        "retrieval_stats": {},
        "clarification_needed": False,
        "clarification_question": "",
        "missing_slots": [],
        "response_type": "recommendation",
        "query_complexity": "",
        "query_features": {},
        "transformed_queries": [],
        "self_rag_judgements": {},
        "retrieval_strategy": "standard",
        "eval_metrics": {},
        "skip_graph_eval": False,
        "user_id": user_id,
        "assembled_context": [],
        "active_skill": "",
        "goal_type": "",
        "planner_next_action": "retrieve",
        "inherit_followup_direction": False,
        "followup_mode": "",
        "followup_anchor_names": [],
        "memory_stats": {},
        "recent_feedback_signals": {},
        "recent_recommended_recipes": [],
        "request_id": request_id,
        "phase_timings": {},
        "token_usage": {},
        "stable_user_preferences": {},
        "applied_stable_preference_keys": [],
    }
