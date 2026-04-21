"""
Structured Output 模型定义

定义 LangGraph 各节点使用的 Pydantic 模型，用于 LLM Structured Output：
- IntentClassification: 意图分类结果
- Plan: 执行计划
- EvaluationResult: 质量评估结果
- RetrievalJudgement: 检索必要性判断 (Phase 1)
- RelevanceJudgement: 文档相关性判断 (Phase 1)
- HallucinationJudgement: 幻觉检测 (Phase 1)
- UsefulnessJudgement: 有用性判断 (Phase 1)
- QueryComplexity: 查询复杂度分类 (Phase 1)
- RecipeQueryConstraints: 标准化的食谱查询约束模型
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class RecipeQueryConstraints(BaseModel):
    available_ingredients: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    max_cooking_time: int | None = Field(default=None)
    health_goal: str | None = Field(default=None)
    meal_type: str | None = Field(default=None)
    prefer_inventory_first: bool = Field(default=False)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.replace("，", ",").split(",")]
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_health_goal_value(value: Any) -> str | None:
    text = _normalize_optional_text(value)
    if not text:
        return None

    goal_aliases = (
        (("降血脂", "血脂高", "控脂", "低脂"), "控脂"),
        (("减肥", "瘦身", "减脂"), "减脂"),
        (("少油", "低油"), "低油"),
        (("清淡",), "清淡口味"),
        (("低糖", "控糖"), "控糖"),
    )
    for aliases, canonical in goal_aliases:
        if any(alias in text for alias in aliases):
            return canonical
    return text


def _normalize_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def build_default_recipe_query_constraints() -> dict[str, Any]:
    return RecipeQueryConstraints().model_dump()


def normalize_extracted_params(raw_params: dict | None) -> dict[str, Any]:
    params = raw_params or {}
    normalized = build_default_recipe_query_constraints()

    available_ingredients = params.get("available_ingredients", params.get("ingredients"))
    allergies = params.get("allergies", params.get("dietary_restrictions"))

    normalized["available_ingredients"] = _normalize_string_list(available_ingredients)
    normalized["allergies"] = _normalize_string_list(allergies)
    normalized["disliked_ingredients"] = _normalize_string_list(
        params.get("disliked_ingredients")
    )
    normalized["max_cooking_time"] = _normalize_optional_int(
        params.get("max_cooking_time", params.get("time_limit"))
    )
    normalized["health_goal"] = _normalize_health_goal_value(params.get("health_goal"))
    normalized["meal_type"] = _normalize_optional_text(params.get("meal_type"))
    normalized["prefer_inventory_first"] = _normalize_bool(
        params.get("prefer_inventory_first", False)
    )

    recipe_params = RecipeQueryConstraints(**normalized).model_dump()
    passthrough_fields = {
        "video_url": _normalize_optional_text(params.get("video_url", params.get("url"))),
        "video_platform": _normalize_optional_text(params.get("video_platform", params.get("platform"))),
        "summary_scope": _normalize_optional_text(params.get("summary_scope")),
        "tutorial_topic_anchor": _normalize_optional_text(params.get("tutorial_topic_anchor", params.get("topic_anchor"))),
    }
    recipe_params.update({key: value for key, value in passthrough_fields.items() if value is not None})
    return recipe_params


class IntentClassification(BaseModel):
    """意图分类结果
    
    用于 Router 节点的 Structured Output，将用户查询分类为五种意图之一。
    
    Attributes:
        intent: 意图类别
        confidence: 分类置信度 (0.0 ~ 1.0)
        extracted_params: 从查询中提取的参数（食材、时间限制等）
    """
    intent: Literal["recipe_search", "nutrition_query", "ingredient_check", "chitchat", "video_summary"] = Field(
        description="用户意图类别：recipe_search(食谱搜索), nutrition_query(营养查询), "
                    "ingredient_check(食材搭配检查), chitchat(闲聊), video_summary(B站视频总结)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分类置信度，0.0 到 1.0 之间"
    )
    extracted_params: dict = Field(
        default_factory=dict,
        description="从查询中提取的参数，如 ingredients, time_limit, health_goal 等"
    )

    @field_validator("extracted_params", mode="before")
    @classmethod
    def _coerce_extracted_params(cls, value: Any) -> dict:
        if value is None:
            return {}
        return value


class ClarificationDecision(BaseModel):
    clarification_needed: bool = Field(
        description="当前是否必须先追问一个关键问题，才能避免给出低可执行性的推荐"
    )
    missing_slots: list[Literal[
        "available_ingredients",
        "max_cooking_time",
        "health_goal",
        "dietary_restrictions",
        "meal_type",
    ]] = Field(
        default_factory=list,
        description="若需要追问，当前最影响可执行性的缺失槽位"
    )
    question: str = Field(
        default="",
        description="若需要追问，应返回一个单轮澄清问题；否则为空"
    )


class Plan(BaseModel):
    """执行计划
    
    用于 Planner 节点的 Structured Output，生成有序步骤列表。
    
    Attributes:
        steps: 执行步骤列表
        reasoning: 规划理由
    """
    steps: list[str] = Field(
        description="执行步骤列表，如：['检索食谱', '过滤时间限制', '检查搭配', '排序推荐']"
    )
    reasoning: str = Field(
        description="规划理由，说明为什么选择这些步骤"
    )


class EvaluationResult(BaseModel):
    """质量评估结果
    
    用于 Evaluator 节点的 Structured Output，评估生成回复的质量。
    
    Attributes:
        is_satisfactory: 回复是否合格
        issues: 存在的问题列表
        suggestion: 改进建议
    """
    is_satisfactory: bool = Field(
        description="回复是否满足质量标准"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="存在的问题列表，如 ['回复未基于检索文档', '缺少营养信息']"
    )
    suggestion: str = Field(
        default="",
        description="改进建议，指导 Generator 重新生成"
    )


# ── Phase 1 新增 Schema ────────────────────────────────────────────────────────


class RetrievalJudgement(BaseModel):
    """检索必要性判断

    用于 Self-RAG 第一层门控，判断当前查询是否需要外部检索。

    Attributes:
        need_retrieval: 是否需要检索
        reason: 判断理由
    """
    need_retrieval: bool = Field(
        description="是否需要检索：闲聊/简单问答为 False，事实性/食谱搜索为 True"
    )
    reason: str = Field(
        default="",
        description="判断理由"
    )


class RelevanceJudgement(BaseModel):
    """文档相关性判断

    用于 Self-RAG 第二层门控，评估单个文档与查询的相关程度。

    Attributes:
        is_relevant: 文档是否与查询相关
        relevance_score: 相关性分数 (0.0 ~ 1.0)
    """
    is_relevant: bool = Field(
        description="文档是否与查询相关"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="相关性分数，0.0 到 1.0 之间"
    )


class HallucinationJudgement(BaseModel):
    """幻觉检测

    用于 Self-RAG 第三层门控，检测回复中不被检索文档支持的声明。

    Attributes:
        has_hallucination: 回复是否存在幻觉
        hallucinated_claims: 幻觉声明列表
    """
    has_hallucination: bool = Field(
        description="回复是否存在幻觉（不被文档支持的声明）"
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="幻觉声明列表，列出具体不被文档支持的内容"
    )


class UsefulnessJudgement(BaseModel):
    """有用性判断

    用于 Self-RAG 第四层门控，判断回复是否真正解决了用户问题。

    Attributes:
        is_useful: 回复是否有用
        missing_info: 缺失的信息列表
    """
    is_useful: bool = Field(
        description="回复是否真正解决了用户问题"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="缺失的信息列表，如 ['烹饪时间', '卡路里信息']"
    )


class SelfRAGLiteJudgement(BaseModel):
    """Self-RAG 精简联合判断

    用于在一次 Structured Output 调用中同时判断：
    - 回复是否存在幻觉
    - 回复是否真正有用
    """

    has_hallucination: bool = Field(
        description="回复是否存在幻觉（不被文档支持的声明）"
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="检测出的幻觉声明列表"
    )
    is_useful: bool = Field(
        description="回复是否真正解决了用户问题"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="若不够有用，缺失的信息列表"
    )


class QueryComplexity(BaseModel):
    """查询复杂度分类

    用于 Adaptive RAG 的查询分析，根据复杂度选择检索策略。

    Attributes:
        level: 复杂度等级
        reasoning: 分类理由
        suggested_strategy: 建议检索策略
    """
    level: Literal["simple", "complex", "ambiguous"] = Field(
        description="查询复杂度: simple(简单直接), complex(多条件), ambiguous(模糊)"
    )
    reasoning: str = Field(
        default="",
        description="分类理由"
    )
    suggested_strategy: str = Field(
        default="standard",
        description="建议策略: standard / multi_query / hyde / step_back"
    )
