"""Skill registry helpers for runtime-facing integrations."""

from __future__ import annotations

from typing import Any

from src.context.instruction_hierarchy import BUILTIN_SKILLS, SkillRegistry
from .contracts import SkillSpec, normalize_skill_spec


_FIRST_PASS_RUNTIME_SKILLS: dict[str, dict[str, Any]] = {
    "recipe_recommendation": {
        "intent_scope": ["recipe_search"],
        "required_slots": [
            "available_ingredients",
            "max_cooking_time",
            "health_goal",
            "dietary_restrictions",
        ],
        "clarification_policy": {
            "mode": "single_turn",
            "prefer_explicit_constraints": True,
            "slot_priority": [
                "available_ingredients",
                "max_cooking_time",
                "health_goal",
                "dietary_restrictions",
            ],
        },
        "planner_policy": {"default_next_action": "retrieve"},
        "retrieval_profile": {
            "profile": "recipe_search_primary",
            "hard_filter_policy": "recipe_constraints",
            "rerank_bias": ["inventory_match", "goal_fit"],
            "primary_source": "recipe",
            "secondary_sources": ["tutorial"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "evidence_enrichment",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {"mode": "grounded_recipe_first"},
        "response_contract": {"style": "recommendation_markdown", "max_recipe_count": 3},
        "fallback_policy": {"on_low_evidence": "conservative_recommendation"},
        "quality_rubric": {"prioritize": ["constraint_hit", "grounded_recommendation"]},
        "ui_metadata": {"label": "食谱推荐"},
    },
    "followup_recommendation": {
        "description": "延续上一轮推荐方向的 follow-up 食谱推荐",
        "intent_scope": ["recipe_search"],
        "required_slots": [
            "available_ingredients",
            "max_cooking_time",
            "health_goal",
            "dietary_restrictions",
        ],
        "prompt_template": (
            "你是一名专业的饮食推荐师。当前任务是延续上一轮推荐方向，"
            "先承接用户已经认可的方向，再给出不完全重复的新候选；"
            "如果证据不足或只有重复项，要明确说明边界与原因。"
        ),
        "clarification_policy": {
            "mode": "single_turn",
            "prefer_explicit_constraints": True,
            "slot_priority": [
                "available_ingredients",
                "max_cooking_time",
                "health_goal",
                "dietary_restrictions",
            ],
        },
        "planner_policy": {
            "default_next_action": "retrieve",
            "allow_direct_followup": True,
        },
        "retrieval_profile": {
            "profile": "recipe_search_followup",
            "hard_filter_policy": "recipe_constraints",
            "rerank_bias": ["inventory_match", "goal_fit", "feedback_preference"],
            "primary_source": "recipe",
            "secondary_sources": ["tutorial"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "evidence_enrichment",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {"mode": "grounded_recipe_followup"},
        "response_contract": {
            "style": "followup_recommendation_markdown",
            "max_recipe_count": 3,
        },
        "fallback_policy": {"on_low_evidence": "conservative_recommendation"},
        "quality_rubric": {
            "prioritize": ["grounded_recommendation", "constraint_hit", "consistency"],
        },
        "ui_metadata": {"label": "延续推荐"},
    },
    "nutrition_analysis": {
        "intent_scope": ["nutrition_query"],
        "required_slots": ["health_goal"],
        "clarification_policy": {
            "mode": "ask_when_subject_missing",
            "slot_priority": ["health_goal"],
        },
        "planner_policy": {"default_next_action": "retrieve"},
        "retrieval_profile": {
            "profile": "nutrition_grounded",
            "hard_filter_policy": "nutrition_evidence",
            "primary_source": "recipe",
            "secondary_sources": ["tutorial"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "evidence_enrichment",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {
            "mode": "grounded_first",
            "allow_general_advice_when_insufficient": True,
        },
        "response_contract": {
            "style": "nutrition_analysis_markdown",
            "require_evidence_boundary": True,
        },
        "fallback_policy": {"on_low_evidence": "general_advice_only"},
        "quality_rubric": {"prioritize": ["faithfulness", "answer_relevancy"]},
        "ui_metadata": {"label": "营养分析"},
    },
    "meal_planning": {
        "intent_scope": ["recipe_search"],
        "required_slots": ["health_goal", "meal_type"],
        "clarification_policy": {
            "mode": "ask_for_goal_and_scope",
            "slot_priority": ["health_goal", "meal_type"],
        },
        "planner_policy": {"default_next_action": "subgraph_candidate"},
        "retrieval_profile": {
            "profile": "meal_planning_candidates",
            "hard_filter_policy": "weekly_consistency",
        },
        "evidence_policy": {"mode": "grounded_plan_outline"},
        "response_contract": {"style": "weekly_plan_markdown", "allow_subgraph": True},
        "fallback_policy": {"on_low_evidence": "outline_only"},
        "quality_rubric": {"prioritize": ["variety", "consistency", "goal_fit"]},
        "ui_metadata": {"label": "一周饮食计划"},
    },
    "recipe_tutorial": {
        "description": "基于教程知识库回答菜谱做法、步骤和制作教程问题",
        "intent_scope": ["recipe_search"],
        "prompt_template": (
            "你是一名烹饪教程助手。优先基于教程知识库中的步骤、食材和提示回答用户的做法问题；"
            "如果当前证据没有直接覆盖，先说明边界，再补充明确标注的通用建议。"
        ),
        "clarification_policy": {"mode": "disabled"},
        "planner_policy": {"default_next_action": "retrieve"},
        "retrieval_profile": {
            "profile": "tutorial_search",
            "hard_filter_policy": "tutorial_evidence",
            "primary_source": "tutorial",
            "secondary_sources": ["recipe"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "fallback_then_merge",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {
            "mode": "grounded_tutorial_first",
            "allow_general_advice_when_insufficient": True,
            "separate_evidence_from_general_advice": True,
        },
        "response_contract": {
            "style": "tutorial_markdown",
            "max_recipe_count": 1,
            "require_evidence_boundary": True,
        },
        "fallback_policy": {"on_low_evidence": "native_general_cooking"},
        "quality_rubric": {"prioritize": ["faithfulness", "answer_relevancy"]},
        "ui_metadata": {"label": "菜谱教程"},
    },
    "bilibili_tutorial_summary": {
        "description": "将 B 站视频链接总结为结构化中文教程，并产出 JSON/PDF/向量入库结果",
        "intent_scope": ["video_summary"],
        "prompt_template": (
            "你是一名教学型视频总结助手。收到 B 站视频链接后，"
            "需要基于真实视频元数据与字幕内容整理出结构化中文讲义式总结，"
            "并明确返回生成的 JSON、PDF 与入库结果。"
        ),
        "clarification_policy": {"mode": "disabled"},
        "planner_policy": {"default_next_action": "generate_only"},
        "retrieval_profile": {"profile": "video_summary_generation"},
        "evidence_policy": {
            "mode": "grounded_video_summary",
            "allow_general_advice_when_insufficient": False,
        },
        "response_contract": {
            "style": "video_summary_markdown",
            "require_evidence_boundary": True,
        },
        "fallback_policy": {"on_low_evidence": "general_advice_only"},
        "quality_rubric": {"prioritize": ["faithfulness", "answer_relevancy"]},
        "ui_metadata": {"label": "B站视频总结"},
    },
    "nutrition_analysis_conservative": {
        "description": "基于证据边界的保守型营养分析",
        "intent_scope": ["nutrition_query"],
        "required_tools": ["get_nutrition_advice"],
        "required_slots": ["health_goal"],
        "prompt_template": (
            "你是一名注册营养师。仅根据已提供的证据给出营养分析；"
            "当证据不足时，必须明确说明边界，只提供通用饮食建议，不给出未经支持的精确数值。"
        ),
        "clarification_policy": {"mode": "ask_when_subject_missing"},
        "planner_policy": {"default_next_action": "retrieve"},
        "retrieval_profile": {
            "profile": "nutrition_grounded",
            "hard_filter_policy": "nutrition_evidence",
            "primary_source": "recipe",
            "secondary_sources": ["tutorial"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "evidence_enrichment",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {
            "mode": "grounded_first",
            "allow_general_advice_when_insufficient": True,
            "separate_evidence_from_general_advice": True,
        },
        "response_contract": {
            "style": "nutrition_conservative",
            "require_evidence_boundary": True,
        },
        "fallback_policy": {"on_low_evidence": "general_advice_only"},
        "quality_rubric": {"prioritize": ["faithfulness", "evidence_boundary"]},
        "ui_metadata": {"label": "保守型营养分析"},
    },
    "ingredient_check_conservative": {
        "description": "基于证据边界的保守型食材搭配检查",
        "intent_scope": ["ingredient_check"],
        "required_tools": ["check_ingredient_pairing"],
        "prompt_template": (
            "你是一名谨慎的饮食搭配顾问。回答时必须区分文档支持的结论与通用饮食建议；"
            "证据不足时不要把通用经验说成已被当前检索结果直接支持。"
        ),
        "clarification_policy": {"mode": "ask_when_pairing_missing"},
        "planner_policy": {"default_next_action": "retrieve"},
        "retrieval_profile": {
            "profile": "ingredient_pairing_grounded",
            "hard_filter_policy": "pairing_evidence",
            "primary_source": "recipe",
            "secondary_sources": ["tutorial"],
            "allow_cross_source_fallback": True,
            "secondary_usage": "evidence_enrichment",
            "secondary_trigger_min_doc_count": 1,
            "secondary_max_docs": 2,
        },
        "evidence_policy": {
            "mode": "grounded_first",
            "separate_evidence_from_general_advice": True,
        },
        "response_contract": {
            "style": "ingredient_check_conservative",
            "require_evidence_boundary": True,
        },
        "fallback_policy": {"on_low_evidence": "general_advice_only"},
        "quality_rubric": {"prioritize": ["evidence_boundary", "safety"]},
        "ui_metadata": {"label": "保守型食材搭配检查"},
    },
}


def _merged_skill_payload(name: str) -> dict[str, Any] | None:
    runtime_defaults = _FIRST_PASS_RUNTIME_SKILLS.get(name) or {}
    legacy_skill = BUILTIN_SKILLS.get(name) if isinstance(BUILTIN_SKILLS.get(name), dict) else {}
    if not runtime_defaults and not legacy_skill:
        return None
    merged = dict(runtime_defaults)
    merged.update(legacy_skill)
    return merged


def get_registry() -> SkillRegistry:
    return SkillRegistry()


def list_skills() -> list[str]:
    names = list(BUILTIN_SKILLS.keys())
    for name in _FIRST_PASS_RUNTIME_SKILLS.keys():
        if name not in names:
            names.append(name)
    return names


def get_skill(name: str) -> dict[str, Any] | None:
    skill = _merged_skill_payload(name)
    return dict(skill) if skill is not None else None


def get_skill_spec(name: str) -> SkillSpec | None:
    skill = get_skill(name)
    if skill is None:
        return None
    return normalize_skill_spec(name, skill)


def list_skill_specs() -> dict[str, SkillSpec]:
    return {
        name: spec
        for name in list_skills()
        if (spec := get_skill_spec(name)) is not None
    }


def register_skill(name: str, skill_spec: SkillSpec | dict[str, Any]) -> dict[str, Any]:
    normalized_spec = normalize_skill_spec(name, skill_spec)
    normalized_dict = normalized_spec.to_dict()
    BUILTIN_SKILLS[name] = normalized_dict
    return dict(normalized_dict)


def _looks_like_followup_recipe_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(
        phrase in normalized_query
        for phrase in (
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
    )


def _looks_like_recipe_tutorial_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(
        phrase in normalized_query
        for phrase in (
            "怎么做",
            "做法",
            "步骤",
            "分步骤",
            "制作",
            "教程",
        )
    )


def _looks_like_tutorial_followup_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(
        phrase in normalized_query
        for phrase in (
            "详细一点",
            "再详细一点",
            "再展开说说",
            "展开说说",
            "再展开",
            "具体步骤",
            "具体怎么做",
            "讲细一点",
            "说细一点",
            "展开一下",
        )
    )


def _has_followup_recommendation_anchor(params: dict[str, Any]) -> bool:
    recent_feedback_signals = params.get("recent_feedback_signals", {}) or {}
    recent_recommended_recipes = params.get("recent_recommended_recipes", []) or []
    return bool(
        recent_recommended_recipes
        or recent_feedback_signals.get("liked_recipe_ids")
        or recent_feedback_signals.get("disliked_recipe_ids")
    )


def select_skill(intent: str, params: dict[str, Any] | None = None) -> tuple[str, SkillSpec] | None:
    normalized_params = params or {}
    matched: list[tuple[str, SkillSpec]] = [
        (name, spec)
        for name, spec in list_skill_specs().items()
        if spec.supports_intent(intent)
    ]

    if not matched:
        return None

    query_hint = str(normalized_params.get("query") or "").lower()
    followup_mode_hint = str(normalized_params.get("followup_mode") or "").strip()
    goal_type_hint = str(normalized_params.get("goal_type") or "").strip()
    tutorial_topic_anchor = str(normalized_params.get("tutorial_topic_anchor") or "").strip()
    if intent == "recipe_search" and (
        followup_mode_hint in {"anchored_followup", "generic_followup"}
        or goal_type_hint == "followup_recommendation"
        or (
            _looks_like_followup_recipe_request(query_hint)
            and _has_followup_recommendation_anchor(normalized_params)
        )
    ):
        followup_spec = get_skill_spec("followup_recommendation")
        if followup_spec is not None and followup_spec.supports_intent(intent):
            return "followup_recommendation", followup_spec

    if intent == "recipe_search" and any(
        keyword in query_hint for keyword in ("week", "一周", "计划", "plan")
    ):
        meal_planning_spec = get_skill_spec("meal_planning")
        if meal_planning_spec is not None and meal_planning_spec.supports_intent(intent):
            return "meal_planning", meal_planning_spec

    if intent == "recipe_search" and (
        _looks_like_recipe_tutorial_request(query_hint)
        or (tutorial_topic_anchor and _looks_like_tutorial_followup_request(query_hint))
    ):
        tutorial_spec = get_skill_spec("recipe_tutorial")
        if tutorial_spec is not None and tutorial_spec.supports_intent(intent):
            return "recipe_tutorial", tutorial_spec

    if intent == "video_summary":
        bilibili_summary_spec = get_skill_spec("bilibili_tutorial_summary")
        if bilibili_summary_spec is not None and bilibili_summary_spec.supports_intent(intent):
            return "bilibili_tutorial_summary", bilibili_summary_spec

    if intent == "nutrition_query":
        conservative_spec = get_skill_spec("nutrition_analysis_conservative")
        if conservative_spec is not None and conservative_spec.supports_intent(intent):
            return "nutrition_analysis_conservative", conservative_spec

    if intent == "ingredient_check":
        conservative_spec = get_skill_spec("ingredient_check_conservative")
        if conservative_spec is not None and conservative_spec.supports_intent(intent):
            return "ingredient_check_conservative", conservative_spec

    return matched[0]


def select_skill_name(intent: str, params: dict[str, Any] | None = None) -> str:
    selection = select_skill(intent, params)
    return selection[0] if selection else ""


__all__ = [
    "BUILTIN_SKILLS",
    "SkillRegistry",
    "SkillSpec",
    "get_registry",
    "list_skills",
    "get_skill",
    "get_skill_spec",
    "list_skill_specs",
    "register_skill",
    "select_skill",
    "select_skill_name",
]
