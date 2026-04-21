"""
规划节点

仅对 recipe_search 意图触发，使用 LLM + Structured Output 生成执行计划。
其他意图跳过规划，直接返回空计划。

复用模块:
- src/llm/qwen_client.py::get_llm()
- src/utils/logger.py::get_logger()
"""

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import DietAgentState, apply_stable_preferences, normalize_followup_contract, normalize_stable_user_preferences
from ..schemas import ClarificationDecision, Plan, normalize_extracted_params
from ..llm import get_graph_llm
from diet_agent.runtime import resolve_skill_runtime
from ...utils.logger import get_logger


logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """你是一个饮食方案规划器。根据用户的查询和提取的参数，生成一个有序的执行步骤列表。

## 规划原则

1. 步骤应该具体、可执行
2. 通常包含：检索食谱 → 过滤条件 → 检查搭配 → 排序推荐
3. 根据参数调整步骤（如有时间限制则增加时间过滤步骤）

## 示例

用户查询："我有鸡蛋和番茄，想做个快手菜"
参数：{"ingredients": ["鸡蛋", "番茄"], "time_limit": 30}

步骤：
1. 根据食材"鸡蛋、番茄"检索相关食谱
2. 过滤烹饪时间在30分钟以内的食谱
3. 检查食材搭配是否合理
4. 按匹配度和评分排序推荐

用户查询："推荐减肥餐"
参数：{"health_goal": "减肥"}

步骤：
1. 检索适合减肥的低卡食谱
2. 按营养成分筛选高蛋白低脂食谱
3. 排序并推荐最佳选项
"""

CLARIFICATION_SLOT_LABELS = {
    "available_ingredients": "食材",
    "max_cooking_time": "时间",
    "health_goal": "目标",
    "dietary_restrictions": "饮食限制",
    "meal_type": "餐次",
}

CLARIFICATION_QUESTIONS = {
    "available_ingredients": "你现在手头有哪些食材？告诉我 2-4 样就行，我可以按现有食材优先给你推荐。",
    "max_cooking_time": "你更希望 15 分钟快手、30 分钟家常，还是 1 小时内都可以？",
    "health_goal": "你这次更偏向减脂、增肌、控糖，还是单纯想吃得清淡一些？",
    "dietary_restrictions": "你有没有需要避开的饮食限制，比如过敏原或不喜欢吃的食材？",
    "meal_type": "这次更想做早餐、午餐、晚餐，还是加餐/夜宵？",
}

DEFAULT_CLARIFICATION_SLOTS = (
    "available_ingredients",
    "max_cooking_time",
    "health_goal",
    "dietary_restrictions",
)

CLARIFICATION_DECISION_PROMPT = """你是一个饮食助手的澄清决策器。

你的目标是优先保证用户体验，而不是机械补全槽位。
如果已经能基于现有信息直接给出一批合理、可执行的候选，就不要追问。
只有在直接回答大概率会变得空泛、不可执行或误导时，才追问一个最关键的问题。
特别是，当用户发出通用推荐请求（如“再给我推荐”）时，如果最近有推荐或反馈锚点，则不应过度澄清。

可直接推荐的强信号包括：
- 明确食材
- 明确时间预算
- 明确健康目标（如减脂、控糖、高蛋白、养胃、低油）
- 明确餐次或菜品类型，且和其他信号组合后足以推荐

更适合追问的弱信号包括：
- 只有“健康点、清淡点、随便做点什么”这类泛偏好
- 只有“适合全家一起吃、家庭健康”这类泛人群/泛目标描述
- 只有餐次，没有其他约束
- 只有天气、季节感受，没有做饭锚点

输出要求：
- clarification_needed=true 时，只追问一个最关键问题
- 优先追问顺序：食材 > 时间 > 健康目标 > 饮食限制
- question 必须直接、自然，像真实助手
- clarification_needed=false 时，question 置空
"""


FOLLOWUP_RECOMMENDATION_PHRASES = (
    "再给我推荐",
    "再推荐",
    "再来一道",
    "换一道",
    "类似的",
    "同类",
    "同方向",
    "继续推荐",
    "还有吗",
)

SOFT_PREFERENCE_HEALTH_GOALS = {
    "清淡口味",
}

NON_ACTIONABLE_HEALTH_GOALS = {
    "家庭健康",
    "健康",
    "健康一点",
    "均衡饮食",
}

SPECIFIC_RECIPE_TOPIC_MARKERS = (
    "鱼",
    "虾",
    "鸡胸肉",
    "鸡肉",
    "牛肉",
    "猪肉",
    "豆腐",
    "番茄",
    "鸡蛋",
    "西兰花",
    "生菜",
    "面",
    "粥",
    "汤",
    "沙拉",
    "蛋羹",
    "蒸蛋",
    "蒸菜",
    "凉拌",
    "素菜",
)

WEAK_RECIPE_QUERY_MARKERS = (
    "全家",
    "家人",
    "一起吃",
    "这个天气",
    "天气",
    "清淡",
    "健康点",
    "随便",
)


def _looks_like_followup_recipe_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(phrase in normalized_query for phrase in FOLLOWUP_RECOMMENDATION_PHRASES)


def _is_non_actionable_health_goal(health_goal: str) -> bool:
    normalized_goal = str(health_goal or "").strip()
    return normalized_goal in NON_ACTIONABLE_HEALTH_GOALS


def _is_soft_preference_health_goal(health_goal: str) -> bool:
    normalized_goal = str(health_goal or "").strip()
    return normalized_goal in SOFT_PREFERENCE_HEALTH_GOALS


def _has_specific_recipe_topic_anchor(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(marker in normalized_query for marker in SPECIFIC_RECIPE_TOPIC_MARKERS)


def _looks_like_weak_anchor_recipe_query(query: str, extracted_params: dict) -> bool:
    normalized_query = str(query or "").strip()
    normalized_params = normalize_extracted_params(extracted_params)
    health_goal = str(normalized_params.get("health_goal") or "").strip()

    if _is_non_actionable_health_goal(health_goal):
        return True
    if _is_soft_preference_health_goal(health_goal) and not _has_specific_recipe_topic_anchor(normalized_query):
        return True
    return any(marker in normalized_query for marker in WEAK_RECIPE_QUERY_MARKERS) and not _has_specific_recipe_topic_anchor(normalized_query)


def _has_followup_recommendation_anchor(decision_input: dict) -> bool:
    recent_feedback_signals = decision_input.get("recent_feedback_signals", {}) or {}
    recent_recommended_recipes = decision_input.get("recent_recommended_recipes", []) or []
    return bool(
        recent_recommended_recipes
        or recent_feedback_signals.get("liked_recipe_ids")
        or recent_feedback_signals.get("disliked_recipe_ids")
    )


def _extract_followup_anchor_names(decision_input: dict) -> list[str]:
    recent_recommended_recipes = decision_input.get("recent_recommended_recipes", []) or []
    anchor_names: list[str] = []
    seen_names: set[str] = set()

    for item in recent_recommended_recipes[:3]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        anchor_names.append(name)

    return anchor_names


def _resolve_followup_mode(decision_input: dict) -> str:
    user_query = str(decision_input.get("query", "")).strip()
    if not _looks_like_followup_recipe_request(user_query):
        return ""
    if _has_followup_recommendation_anchor(decision_input):
        return "anchored_followup"
    return "generic_followup"


def _resolve_clarification_slots(skill_spec) -> list[str]:
    configured_slots: list[str] = []
    clarification_policy = dict(getattr(skill_spec, "clarification_policy", {}) or {})
    clarification_mode = str(clarification_policy.get("mode") or "").strip()

    if clarification_mode in {"disabled", "skip", "never"}:
        return []

    if skill_spec is not None:
        slot_priority = skill_spec.clarification_policy.get("slot_priority")
        if isinstance(slot_priority, (list, tuple)):
            for slot_name in slot_priority:
                normalized_slot_name = str(slot_name or "").strip()
                if normalized_slot_name in CLARIFICATION_QUESTIONS and normalized_slot_name not in configured_slots:
                    configured_slots.append(normalized_slot_name)

        for slot_name in skill_spec.required_slots:
            normalized_slot_name = str(slot_name or "").strip()
            if normalized_slot_name in CLARIFICATION_QUESTIONS and normalized_slot_name not in configured_slots:
                configured_slots.append(normalized_slot_name)

    if configured_slots:
        return configured_slots

    return list(DEFAULT_CLARIFICATION_SLOTS)


def _clarification_disabled(skill_spec) -> bool:
    clarification_policy = dict(getattr(skill_spec, "clarification_policy", {}) or {})
    clarification_mode = str(clarification_policy.get("mode") or "").strip()
    return clarification_mode in {"disabled", "skip", "never"}


def _is_slot_missing(extracted_params: dict, slot_name: str) -> bool:
    if slot_name == "available_ingredients":
        return not bool(extracted_params.get("available_ingredients"))
    if slot_name == "max_cooking_time":
        return extracted_params.get("max_cooking_time") is None
    if slot_name == "health_goal":
        health_goal = str(extracted_params.get("health_goal") or "").strip()
        return not health_goal or _is_non_actionable_health_goal(health_goal)
    if slot_name == "dietary_restrictions":
        dietary_restrictions = (
            list(extracted_params.get("allergies") or [])
            + list(extracted_params.get("disliked_ingredients") or [])
        )
        return not dietary_restrictions
    if slot_name == "meal_type":
        return not str(extracted_params.get("meal_type") or "").strip()

    value = extracted_params.get(slot_name)
    if isinstance(value, list):
        return len(value) == 0
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return not bool(value)


def _get_missing_slots(extracted_params: dict, skill_spec=None) -> list[str]:
    normalized_params = normalize_extracted_params(extracted_params)
    return [
        slot_name
        for slot_name in _resolve_clarification_slots(skill_spec)
        if _is_slot_missing(normalized_params, slot_name)
    ]


def _build_decision_payload(
    clarification_needed: bool,
    missing_slots: list[str],
    question: str,
) -> dict:
    return {
        "clarification_needed": clarification_needed,
        "clarification_question": question if clarification_needed else "",
        "missing_slots": missing_slots,
        "response_type": "clarification" if clarification_needed else "recommendation",
    }


def _resolve_goal_type(intent: str, followup_mode: str, active_skill: str = "") -> str:
    if intent != "recipe_search":
        return intent or ""
    if active_skill in {"meal_planning", "followup_recommendation"}:
        return active_skill
    if followup_mode in {"anchored_followup", "generic_followup"}:
        return "followup_recommendation"
    return "recipe_recommendation"


def _has_actionable_followup_constraints(extracted_params: dict) -> bool:
    normalized_params = normalize_extracted_params(extracted_params)
    return bool(
        normalized_params.get("available_ingredients")
        or normalized_params.get("allergies")
        or normalized_params.get("disliked_ingredients")
        or normalized_params.get("max_cooking_time") is not None
        or normalized_params.get("health_goal")
        or normalized_params.get("meal_type")
        or normalized_params.get("prefer_inventory_first")
    )


def _select_direct_followup_candidates(followup_anchor_names: list[str], recent_recommended_recipes: list[dict]) -> list[dict]:
    anchor_name_set = {str(name).strip() for name in followup_anchor_names if str(name).strip()}
    selected_candidates: list[dict] = []
    seen_names: set[str] = set()

    for item in recent_recommended_recipes[:5]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen_names or name in anchor_name_set:
            continue
        seen_names.add(name)
        selected_candidates.append(dict(item))

    return selected_candidates


def _resolve_planner_next_action(
    clarification_needed: bool,
    followup_mode: str,
    followup_anchor_names: list[str],
    recent_recommended_recipes: list[dict],
    extracted_params: dict,
    skill_spec=None,
) -> str:
    if clarification_needed:
        return "clarify"
    if followup_mode == "anchored_followup":
        if _has_actionable_followup_constraints(extracted_params):
            return "retrieve"
        if _select_direct_followup_candidates(followup_anchor_names, recent_recommended_recipes):
            return "direct_followup"

    planner_policy = dict(getattr(skill_spec, "planner_policy", {}) or {})
    default_next_action = str(planner_policy.get("default_next_action") or "").strip()
    return default_next_action or "retrieve"


def _build_rule_based_clarification_decision(decision_input: dict, skill_spec=None) -> dict:
    user_query = str(decision_input.get("query", "")).strip()
    extracted_params = normalize_extracted_params(decision_input.get("extracted_params", {}))
    missing_slots = _get_missing_slots(extracted_params, skill_spec)
    has_ingredients = bool(extracted_params.get("available_ingredients"))
    has_time = extracted_params.get("max_cooking_time") is not None
    health_goal = str(extracted_params.get("health_goal") or "").strip()
    has_goal = bool(health_goal)
    has_non_actionable_goal = _is_non_actionable_health_goal(health_goal)
    has_soft_goal = _is_soft_preference_health_goal(health_goal)
    has_strong_goal = has_goal and not has_non_actionable_goal and not has_soft_goal
    has_meal_type = bool(extracted_params.get("meal_type"))
    has_dietary_restrictions = bool(
        list(extracted_params.get("allergies") or [])
        + list(extracted_params.get("disliked_ingredients") or [])
    )
    has_specific_topic_anchor = _has_specific_recipe_topic_anchor(user_query)
    has_weak_anchor = _looks_like_weak_anchor_recipe_query(user_query, extracted_params)

    clarification_needed = not (
        has_ingredients
        or has_time
        or has_strong_goal
        or ((has_soft_goal or has_non_actionable_goal) and has_specific_topic_anchor)
        or (has_meal_type and has_dietary_restrictions)
    )
    if not clarification_needed and has_weak_anchor and not (
        has_ingredients
        or has_time
        or has_specific_topic_anchor
        or has_dietary_restrictions
    ):
        clarification_needed = True
    next_slot = missing_slots[0] if missing_slots else ""

    return _build_decision_payload(
        clarification_needed=clarification_needed,
        missing_slots=missing_slots,
        question=CLARIFICATION_QUESTIONS.get(next_slot, ""),
    )


def _build_clarification_decision(decision_input: dict, skill_spec=None) -> dict:
    user_query = str(decision_input.get("query", "")).strip()
    extracted_params = normalize_extracted_params(decision_input.get("extracted_params", {}))
    clarification_slots = _resolve_clarification_slots(skill_spec)
    if _clarification_disabled(skill_spec):
        return _build_decision_payload(
            clarification_needed=False,
            missing_slots=[],
            question="",
        )
    if _looks_like_followup_recipe_request(user_query) and _has_followup_recommendation_anchor(decision_input):
        return _build_decision_payload(
            clarification_needed=False,
            missing_slots=_get_missing_slots(extracted_params, skill_spec),
            question="",
        )

    fallback_decision = _build_rule_based_clarification_decision({
        "query": user_query,
        "extracted_params": extracted_params,
    }, skill_spec=skill_spec)

    if not fallback_decision["clarification_needed"] or not user_query:
        return fallback_decision

    try:
        llm = get_graph_llm()
        structured_llm = llm.with_structured_output(ClarificationDecision)
        result: ClarificationDecision = structured_llm.invoke([
            SystemMessage(content=CLARIFICATION_DECISION_PROMPT),
            HumanMessage(content=(
                f"当前 skill：{getattr(skill_spec, 'skill_id', '')}\n"
                f"优先澄清槽位：{clarification_slots}\n"
                f"用户查询：{user_query}\n标准化约束：{extracted_params}"
            )),
        ])

        missing_slots = [
            slot for slot in result.missing_slots if slot in clarification_slots
        ]
        if not missing_slots:
            missing_slots = list(fallback_decision["missing_slots"])
        next_slot = missing_slots[0] if missing_slots else ""
        question = result.question.strip() if result.clarification_needed else ""
        if result.clarification_needed and not question:
            question = CLARIFICATION_QUESTIONS.get(next_slot, "")

        return _build_decision_payload(
            clarification_needed=result.clarification_needed,
            missing_slots=missing_slots,
            question=question,
        )
    except Exception as e:
        logger.warning(f"澄清决策 LLM 失败，回退到规则判定: {e}")
        return fallback_decision


def planner_node(state: DietAgentState) -> dict:
    """规划节点
    
    仅对 recipe_search 意图触发。使用 LLM + Structured Output 基于
    用户查询和提取的参数生成执行计划。
    
    Args:
        state: LangGraph 全局状态
    
    Returns:
        包含 plan 和 current_step 的字典
    """
    intent = state.get("intent", "")
    logger.info(f"Planner 节点开始执行, intent={intent}")
    
    # 非 recipe_search 意图跳过规划
    if intent != "recipe_search":
        logger.info(f"意图 '{intent}' 无需规划，跳过")
        return {"plan": [], "current_step": 0}
    
    # 获取用户查询
    messages = state.get("messages", [])
    user_query = ""
    if messages:
        last_message = messages[-1]
        user_query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    stable_user_preferences = normalize_stable_user_preferences(
        state.get("stable_user_preferences", {})
    )
    extracted_params, applied_stable_preference_keys = apply_stable_preferences(
        state.get("extracted_params", {}),
        stable_user_preferences,
    )
    followup_mode = _resolve_followup_mode({
        "query": user_query,
        "recent_feedback_signals": state.get("recent_feedback_signals", {}) or {},
        "recent_recommended_recipes": state.get("recent_recommended_recipes", []) or [],
    })
    followup_anchor_names = _extract_followup_anchor_names({
        "recent_recommended_recipes": state.get("recent_recommended_recipes", []) or [],
    })
    followup_mode, followup_anchor_names = normalize_followup_contract(
        followup_mode,
        followup_anchor_names,
    )
    resolved_skill = resolve_skill_runtime(
        intent=intent,
        params={
            **extracted_params,
            "query": user_query,
            "followup_mode": followup_mode,
            "recent_feedback_signals": state.get("recent_feedback_signals", {}) or {},
            "recent_recommended_recipes": state.get("recent_recommended_recipes", []) or [],
        },
    )
    active_skill = resolved_skill.skill_name
    active_skill_spec = resolved_skill.skill_spec
    goal_type = _resolve_goal_type(intent, followup_mode, active_skill)
    inherit_followup_direction = followup_mode in {"anchored_followup", "generic_followup"}
    clarification_decision = _build_clarification_decision({
        "query": user_query,
        "extracted_params": extracted_params,
        "recent_feedback_signals": state.get("recent_feedback_signals", {}) or {},
        "recent_recommended_recipes": state.get("recent_recommended_recipes", []) or [],
    }, skill_spec=active_skill_spec)
    planner_next_action = _resolve_planner_next_action(
        clarification_needed=clarification_decision["clarification_needed"],
        followup_mode=followup_mode,
        followup_anchor_names=followup_anchor_names,
        recent_recommended_recipes=state.get("recent_recommended_recipes", []) or [],
        extracted_params=extracted_params,
        skill_spec=active_skill_spec,
    )
    
    logger.info(f"生成规划: query='{user_query[:100]}', params={extracted_params}")
    if clarification_decision["clarification_needed"]:
        logger.info(
            "命中单轮澄清: missing_slots=%s, question=%s",
            clarification_decision["missing_slots"],
            clarification_decision["clarification_question"],
        )
        return {
            "plan": [],
            "current_step": 0,
            "active_skill": active_skill,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "goal_type": goal_type,
            "planner_next_action": planner_next_action,
            "inherit_followup_direction": inherit_followup_direction,
            "followup_mode": followup_mode,
            "followup_anchor_names": followup_anchor_names,
            **clarification_decision,
        }
    
    if planner_next_action == "direct_followup":
        logger.info("命中 direct_followup，跳过额外规划，直接进入生成")
        return {
            "plan": ["直接延续上一轮候选生成 follow-up 推荐"],
            "current_step": 0,
            "active_skill": active_skill,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "goal_type": goal_type,
            "planner_next_action": planner_next_action,
            "inherit_followup_direction": inherit_followup_direction,
            "followup_mode": followup_mode,
            "followup_anchor_names": followup_anchor_names,
            **clarification_decision,
        }

    if planner_next_action == "subgraph_candidate":
        logger.info("命中 subgraph_candidate，当前冻结为结构化提纲路径")
        return {
            "plan": ["整理本周目标与餐次范围", "筛选可复用候选方向", "先输出结构化饮食安排提纲"],
            "current_step": 0,
            "active_skill": active_skill,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "goal_type": goal_type,
            "planner_next_action": planner_next_action,
            "inherit_followup_direction": inherit_followup_direction,
            "followup_mode": followup_mode,
            "followup_anchor_names": followup_anchor_names,
            **clarification_decision,
        }

    try:
        llm = get_graph_llm()
        structured_llm = llm.with_structured_output(Plan)
        
        planning_input = f"用户查询：{user_query}\n提取的参数：{extracted_params}"
        
        result: Plan = structured_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=planning_input)
        ])
        
        logger.info(f"规划完成: {len(result.steps)} 个步骤, 理由: {result.reasoning[:100]}")
        
        return {
            "plan": result.steps,
            "current_step": 0,
            "active_skill": active_skill,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "goal_type": goal_type,
            "planner_next_action": planner_next_action,
            "inherit_followup_direction": inherit_followup_direction,
            "followup_mode": followup_mode,
            "followup_anchor_names": followup_anchor_names,
            **clarification_decision,
        }
        
    except Exception as e:
        logger.error(f"Planner 节点执行失败: {e}", exc_info=True)
        # 降级：使用默认计划
        return {
            "plan": ["检索相关食谱", "筛选并推荐"],
            "current_step": 0,
            "active_skill": active_skill,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "goal_type": goal_type,
            "planner_next_action": planner_next_action,
            "inherit_followup_direction": inherit_followup_direction,
            "followup_mode": followup_mode,
            "followup_anchor_names": followup_anchor_names,
            **clarification_decision,
        }
