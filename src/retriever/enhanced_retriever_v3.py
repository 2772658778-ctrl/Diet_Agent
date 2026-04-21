"""
增强检索器 V3

整合以下功能：
1. 向量检索（ChromaDB）
2. 关系推理（知识图谱）
3. 用户偏好（PostgreSQL）
4. Cross-Encoder 精排
5. RRF 融合
6. MMR 多样性
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime
from langchain_community.vectorstores import Chroma

from ..vectorstore.hybrid_retriever import HybridRetriever
from ..database.postgres_client import PostgresClient
from ..graph.schemas import normalize_extracted_params
from ..reranker.cross_encoder_reranker import CrossEncoderReranker
from ..utils.logger import get_logger

logger = get_logger(__name__)

TIME_FILTER_DISABLED_POLICIES = {"nutrition_evidence", "pairing_evidence"}
RERANK_BIAS_WEIGHTS = {
    "inventory_match": 0.10,
    "goal_fit": 0.10,
    "time_fit": 0.05,
    "feedback_preference": 0.05,
}


def _normalize_text_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = value.replace("，", ",").split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    normalized: set[str] = set()
    for item in raw_items:
        text = str(item).strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _normalize_inventory_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ingredient_name = str(item.get("ingredient_name") or "").strip()
        if not ingredient_name:
            continue
        normalized_items.append({**item, "ingredient_name": ingredient_name})
    return normalized_items


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace("，", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]

    normalized_items: list[str] = []
    seen_items: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip().lower()
        if text and text not in seen_items:
            seen_items.add(text)
            normalized_items.append(text)
    return normalized_items


def _get_retrieval_profile(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _resolve_hard_filter_policy(context: Dict[str, Any]) -> str:
    retrieval_profile = _get_retrieval_profile(context.get("retrieval_profile"))
    return str(
        retrieval_profile.get("hard_filter_policy")
        or context.get("hard_filter_policy")
        or ""
    ).strip()


def _resolve_rerank_bias(context: Dict[str, Any]) -> list[str]:
    retrieval_profile = _get_retrieval_profile(context.get("retrieval_profile"))
    raw_bias = retrieval_profile.get("rerank_bias")
    if raw_bias is None:
        raw_bias = context.get("rerank_bias")
    return _normalize_string_list(raw_bias)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _calculate_expiry_urgency(days_to_expiry: Optional[int]) -> float:
    if days_to_expiry is None:
        return 0.0
    if days_to_expiry <= 1:
        return 1.0
    if days_to_expiry <= 3:
        return 0.8
    if days_to_expiry <= 5:
        return 0.6
    if days_to_expiry <= 7:
        return 0.4
    return 0.1


def _build_doc_goal_blob(doc: Dict[str, Any]) -> str:
    blob_parts = [
        str(doc.get("name") or ""),
        str(doc.get("description") or ""),
        str(doc.get("text") or ""),
        str(doc.get("tags") or ""),
        str(doc.get("goal_tags") or ""),
        str(doc.get("health_goals") or ""),
    ]
    return " ".join(part for part in blob_parts if part).lower()


def _normalize_health_goal(health_goal: str) -> str:
    normalized_goal = str(health_goal or "").strip().lower()
    if not normalized_goal:
        return ""

    goal_aliases = (
        ("降血脂", "控脂"),
        ("血脂高", "控脂"),
        ("控脂", "控脂"),
        ("低脂", "控脂"),
        ("减肥", "减脂"),
        ("瘦身", "减脂"),
        ("减脂", "减脂"),
        ("高蛋白", "高蛋白"),
        ("增肌", "增肌"),
        ("控糖", "控糖"),
        ("低糖", "控糖"),
        ("养胃", "养胃"),
        ("清淡", "清淡口味"),
        ("低油", "低油"),
        ("少油", "低油"),
    )
    for alias, canonical in goal_aliases:
        if alias in normalized_goal:
            return canonical
    return normalized_goal


def _compute_goal_fit_score(doc: Dict[str, Any], health_goal: str) -> float:
    normalized_goal = _normalize_health_goal(health_goal)
    if not normalized_goal:
        return 0.0

    blob = _build_doc_goal_blob(doc)
    calories = _safe_float(doc.get("calories"), default=-1.0)
    protein = _safe_float(doc.get("protein"), default=-1.0)
    positive_markers = {
        "减脂": ("减脂", "减肥", "低脂", "低卡", "低热量", "清淡", "鸡胸肉", "沙拉", "时蔬"),
        "控脂": ("控脂", "降脂", "低脂", "低油", "清淡", "蒸", "鱼", "时蔬"),
        "高蛋白": ("高蛋白", "鸡胸肉", "鸡蛋", "蛋羹", "鱼", "瘦肉", "牛肉", "豆腐"),
        "控糖": ("控糖", "低糖", "粗粮", "高纤", "无糖"),
        "养胃": ("养胃", "易消化", "清淡", "粥", "汤", "蒸"),
        "清淡口味": ("清淡", "蒸", "汤", "粥", "凉拌", "时蔬", "鱼"),
        "低油": ("低油", "少油", "清淡", "蒸", "凉拌", "沙拉"),
    }
    if any(marker in blob for marker in positive_markers.get(normalized_goal, ())):
        return 1.0

    if normalized_goal in {"减脂", "控脂", "清淡口味", "低油"} and 0 <= calories <= 300:
        return 0.5
    if normalized_goal == "高蛋白" and protein >= 15:
        return 0.5
    return 0.0


def _compute_goal_conflict_score(doc: Dict[str, Any], health_goal: str) -> float:
    normalized_goal = _normalize_health_goal(health_goal)
    if not normalized_goal:
        return 0.0

    blob = _build_doc_goal_blob(doc)
    calories = _safe_float(doc.get("calories"), default=-1.0)
    penalty = 0.0
    negative_markers = {
        "减脂": ("红烧肉", "五花肉", "油炸", "肥而不腻", "烧烤", "干锅"),
        "控脂": ("红烧肉", "五花肉", "油炸", "肥而不腻", "高脂", "烧烤", "干锅"),
        "控糖": ("糖醋", "甜品", "奶茶", "蛋糕", "冰糖", "高糖"),
        "养胃": ("麻辣", "辛辣", "油炸", "烧烤", "干锅"),
        "清淡口味": ("麻辣", "重口", "油炸", "干锅", "红烧"),
        "低油": ("油炸", "红烧", "干锅", "肥肉", "五花肉"),
    }
    if any(marker in blob for marker in negative_markers.get(normalized_goal, ())):
        penalty = max(penalty, 0.9)

    if normalized_goal in {"减脂", "控脂", "清淡口味", "低油"}:
        if calories >= 600:
            penalty = max(penalty, 1.0)
        elif calories >= 450:
            penalty = max(penalty, 0.7)
    elif normalized_goal in {"控糖", "养胃"} and calories >= 500:
        penalty = max(penalty, 0.6)

    return round(penalty, 4)


def _build_feedback_signals(feedbacks: Any) -> Dict[str, Any]:
    liked_recipe_ids: list[str] = []
    disliked_recipe_ids: list[str] = []
    summary_parts: list[str] = []

    if not isinstance(feedbacks, list):
        return {
            "liked_recipe_ids": liked_recipe_ids,
            "disliked_recipe_ids": disliked_recipe_ids,
            "summary": "",
        }

    for item in feedbacks:
        if not isinstance(item, dict):
            continue
        recipe_id = str(item.get("recipe_id") or "").strip()
        rating = _safe_int(item.get("rating")) or 0
        liked = item.get("liked")
        if liked is None:
            liked = rating >= 4
        if recipe_id:
            if liked:
                liked_recipe_ids.append(recipe_id)
            elif rating > 0:
                disliked_recipe_ids.append(recipe_id)
        comment = str(item.get("comment") or "").strip()
        if recipe_id and comment:
            summary_parts.append(f"{recipe_id}:{comment}")

    return {
        "liked_recipe_ids": liked_recipe_ids,
        "disliked_recipe_ids": disliked_recipe_ids,
        "summary": "；".join(summary_parts[:3]),
    }


class EnhancedRetrieverV3:
    """增强检索器 V3
    
    完整的检索流程：
    1. 从 PostgreSQL 获取用户偏好
    2. 使用混合检索器（向量 + 关系）获取候选结果
    3. 使用 Cross-Encoder 精排
    4. 应用 MMR 多样性
    5. 记录交互到 PostgreSQL
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        postgres_client: Optional[PostgresClient] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        enable_reranking: bool = True,
        enable_user_preferences: bool = True
    ):
        """
        初始化增强检索器
        
        Args:
            vectorstore: ChromaDB 向量存储
            postgres_client: PostgreSQL 客户端（可选）
            reranker: Cross-Encoder 重排序器（可选）
            enable_reranking: 是否启用重排序
            enable_user_preferences: 是否启用用户偏好
        """
        self.hybrid_retriever = HybridRetriever(vectorstore)
        self.postgres_client = postgres_client
        self.reranker = reranker
        self.enable_reranking = enable_reranking and reranker is not None
        self.enable_user_preferences = enable_user_preferences and postgres_client is not None
        
        logger.info(
            f"EnhancedRetrieverV3 初始化完成: "
            f"reranking={self.enable_reranking}, "
            f"user_preferences={self.enable_user_preferences}"
        )
    
    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        lambda_param: float = 0.7,
        rrf_k: int = 60
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        增强检索主函数
        
        执行流程：
        1. 获取用户偏好（如果启用）
        2. 合并用户上下文
        3. 混合检索（向量 + 关系）
        4. Cross-Encoder 精排（如果启用）
        5. MMR 多样性
        6. 记录交互（如果启用）
        
        Args:
            query: 搜索查询
            user_id: 用户 ID（可选）
            user_context: 用户上下文（可选）
            top_k: 返回结果数量
            lambda_param: MMR 参数
            rrf_k: RRF 参数
        
        Returns:
            (检索结果列表, 检索统计)
        """
        logger.info(f"开始 V3 增强检索: query='{query}', user_id={user_id}, top_k={top_k}")
        
        # 1. 获取用户偏好
        merged_context = self._get_user_context(user_id, user_context)
        retrieval_stats = self._build_default_retrieval_stats(merged_context)
        
        # 2. 混合检索（向量 + 关系）
        # 获取更多候选结果用于精排
        candidate_k = top_k * 3 if self.enable_reranking else top_k
        
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            user_context=merged_context,
            top_k=candidate_k,
            lambda_param=lambda_param,
            rrf_k=rrf_k
        )
        retrieval_stats.update(self.hybrid_retriever.last_fusion_metadata)
        
        logger.info(f"混合检索返回 {len(candidates)} 条候选结果")
        retrieval_stats["raw_candidate_count"] = len(candidates)
        
        # 3. Cross-Encoder 精排
        if self.enable_reranking and candidates:
            logger.info("开始 Cross-Encoder 精排...")
            candidates = self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=top_k * 2  # 精排后保留更多结果用于 MMR
            )
            logger.info(f"精排后保留 {len(candidates)} 条结果")
        retrieval_stats["post_rerank_candidate_count"] = len(candidates)

        # 4. 约束后处理：硬过滤 + 可解释线性重排
        processed_candidates, post_stats = self._apply_constraint_postprocessing(
            candidates=candidates,
            context=merged_context,
        )
        retrieval_stats.update(post_stats)

        # 5. 最终结果（已经在 hybrid_retriever 中应用了 MMR）
        final_results = processed_candidates[:top_k]
        retrieval_stats["returned_doc_count"] = len(final_results)

        # 6. 记录交互
        if self.enable_user_preferences and user_id and merged_context.get("write_interaction_during_retrieval"):
            self._log_interaction(user_id, query, final_results, merged_context)
        
        logger.info(f"V3 增强检索完成，返回 {len(final_results)} 条结果")
        return final_results, retrieval_stats

    def _build_default_retrieval_stats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        retrieval_profile = _get_retrieval_profile(context.get("retrieval_profile"))
        return {
            "constraint_count": self._count_active_constraints(context),
            "filtered_doc_count": 0,
            "hard_filter_reasons": {},
            "inventory_match_ratio": 0.0,
            "goal_fit_score": 0.0,
            "expiry_urgency_score": 0.0,
            "bias_bonus_score": 0.0,
            "feedback_signal_count": 0,
            "fallback_triggered": False,
            "raw_candidate_count": 0,
            "post_rerank_candidate_count": 0,
            "returned_doc_count": 0,
            "fusion_mode": "",
            "w_struct": 0.0,
            "w_sem": 0.0,
            "vector_candidate_count": 0,
            "relation_candidate_count": 0,
            "retrieval_profile": retrieval_profile,
            "hard_filter_policy": _resolve_hard_filter_policy(context),
            "rerank_bias": _resolve_rerank_bias(context),
        }

    def _count_active_constraints(self, context: Dict[str, Any]) -> int:
        count = 0
        if context.get("available_ingredients"):
            count += 1
        if context.get("allergies"):
            count += 1
        if context.get("disliked_ingredients"):
            count += 1
        if context.get("max_cooking_time"):
            count += 1
        if context.get("health_goal"):
            count += 1
        if context.get("meal_type"):
            count += 1
        return count

    def _apply_constraint_postprocessing(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        retrieval_profile = _get_retrieval_profile(context.get("retrieval_profile"))
        hard_filter_policy = _resolve_hard_filter_policy(context)
        rerank_bias = _resolve_rerank_bias(context)
        allergies = _normalize_text_set(context.get("allergies"))
        disliked = _normalize_text_set(context.get("disliked_ingredients"))
        available_ingredients = _normalize_text_set(context.get("available_ingredients"))
        inventory_items = _normalize_inventory_items(context.get("inventory_items"))
        health_goal = str(context.get("health_goal") or "").strip().lower()
        max_cooking_time = context.get("max_cooking_time")
        prefer_inventory_first = bool(context.get("prefer_inventory_first", False))
        recent_feedback_signals = context.get("recent_feedback_signals", {}) or {}

        filtered_candidates: List[Dict[str, Any]] = []
        filtered_count = 0
        hard_filter_reasons: Dict[str, int] = {}
        inventory_scores: List[float] = []
        goal_scores: List[float] = []
        expiry_scores: List[float] = []
        bias_bonus_scores: List[float] = []

        for doc in candidates:
            doc_copy = dict(doc)
            violation_reason = self._get_hard_filter_reason(
                doc=doc_copy,
                allergies=allergies,
                disliked_ingredients=disliked,
                max_cooking_time=max_cooking_time,
                hard_filter_policy=hard_filter_policy,
            )
            if violation_reason:
                filtered_count += 1
                hard_filter_reasons[violation_reason] = hard_filter_reasons.get(violation_reason, 0) + 1
                continue

            score_breakdown = self._build_score_breakdown(
                doc=doc_copy,
                available_ingredients=available_ingredients,
                inventory_items=inventory_items,
                health_goal=health_goal,
                max_cooking_time=max_cooking_time,
                prefer_inventory_first=prefer_inventory_first,
                recent_feedback_signals=recent_feedback_signals,
                rerank_bias=rerank_bias,
            )
            doc_copy["score_breakdown"] = score_breakdown
            doc_copy["final_score"] = score_breakdown["final_score"]
            filtered_candidates.append(doc_copy)
            inventory_scores.append(score_breakdown["inventory_match_ratio"])
            goal_scores.append(score_breakdown["goal_fit_score"])
            expiry_scores.append(score_breakdown["expiry_urgency_score"])
            bias_bonus_scores.append(score_breakdown["bias_bonus_score"])

        ranked_candidates = sorted(
            filtered_candidates,
            key=lambda item: item.get("final_score", 0.0),
            reverse=True,
        )

        stats = {
            "filtered_doc_count": filtered_count,
            "hard_filter_reasons": hard_filter_reasons,
            "inventory_match_ratio": round(sum(inventory_scores) / len(inventory_scores), 4) if inventory_scores else 0.0,
            "goal_fit_score": round(sum(goal_scores) / len(goal_scores), 4) if goal_scores else 0.0,
            "expiry_urgency_score": round(sum(expiry_scores) / len(expiry_scores), 4) if expiry_scores else 0.0,
            "bias_bonus_score": round(sum(bias_bonus_scores) / len(bias_bonus_scores), 4) if bias_bonus_scores else 0.0,
            "feedback_signal_count": len(recent_feedback_signals.get("liked_recipe_ids", [])) + len(recent_feedback_signals.get("disliked_recipe_ids", [])),
            "prefer_inventory_first": prefer_inventory_first,
            "fallback_triggered": bool(candidates and not ranked_candidates),
            "retrieval_profile": retrieval_profile,
            "hard_filter_policy": hard_filter_policy,
            "rerank_bias": list(rerank_bias),
        }
        return ranked_candidates, stats

    def _get_hard_filter_reason(
        self,
        doc: Dict[str, Any],
        allergies: set[str],
        disliked_ingredients: set[str],
        max_cooking_time: Optional[int],
        hard_filter_policy: str,
    ) -> Optional[str]:
        doc_ingredients = _normalize_text_set(doc.get("ingredient_names", doc.get("ingredients", [])))
        doc_allergens = _normalize_text_set(doc.get("allergens", []))
        combined_ingredients = doc_ingredients | doc_allergens

        if allergies and combined_ingredients.intersection(allergies):
            return "allergy_conflict"

        if disliked_ingredients and doc_ingredients.intersection(disliked_ingredients):
            return "disliked_ingredient_conflict"

        cook_time = _safe_int(doc.get("cook_time", doc.get("time")))
        if (
            hard_filter_policy not in TIME_FILTER_DISABLED_POLICIES
            and max_cooking_time
            and cook_time
            and cook_time > max_cooking_time * 1.5
        ):
            return "time_budget_exceeded"

        return None

    def _build_score_breakdown(
        self,
        doc: Dict[str, Any],
        available_ingredients: set[str],
        inventory_items: list[dict[str, Any]],
        health_goal: str,
        max_cooking_time: Optional[int],
        prefer_inventory_first: bool,
        recent_feedback_signals: Dict[str, Any],
        rerank_bias: List[str],
    ) -> Dict[str, Any]:
        retrieval_score = _safe_float(
            doc.get("rerank_score", doc.get("score", doc.get("similarity_score", 0.0))),
            default=0.0,
        )
        ingredient_names = _normalize_text_set(doc.get("ingredient_names", doc.get("ingredients", [])))
        goal_tags = _normalize_text_set(doc.get("goal_tags", doc.get("health_goals", doc.get("tags", []))))
        cook_time = _safe_int(doc.get("cook_time", doc.get("time")))

        matched_ingredients = sorted(ingredient_names.intersection(available_ingredients))
        missing_ingredients = sorted(ingredient_names.difference(available_ingredients))
        ingredient_match_score = 0.0
        if ingredient_names and available_ingredients:
            ingredient_match_score = len(matched_ingredients) / len(ingredient_names)

        ingredient_coverage = 0.0
        if ingredient_names:
            ingredient_coverage = len(matched_ingredients) / len(ingredient_names)

        matched_inventory_items = []
        expiring_soon_ingredients = []
        expiry_urgency_values = []
        if inventory_items and matched_ingredients:
            matched_ingredient_set = set(matched_ingredients)
            for item in inventory_items:
                ingredient_name = str(item.get("ingredient_name") or "").strip().lower()
                if ingredient_name in matched_ingredient_set:
                    normalized_item = dict(item)
                    expiry_value = _parse_date(item.get("expiry_date"))
                    expiry_days = None
                    if expiry_value is not None:
                        expiry_days = (expiry_value - date.today()).days
                    urgency_score = _calculate_expiry_urgency(expiry_days)
                    normalized_item["expiry_days"] = expiry_days
                    normalized_item["expiry_urgency_score"] = round(urgency_score, 4)
                    matched_inventory_items.append(normalized_item)
                    if urgency_score > 0:
                        expiry_urgency_values.append(urgency_score)
                    if urgency_score >= 0.6:
                        expiring_soon_ingredients.append(str(item.get("ingredient_name") or "").strip())

        expiry_urgency_score = round(
            sum(expiry_urgency_values) / len(expiry_urgency_values), 4
        ) if expiry_urgency_values else 0.0
        inventory_priority_score = round(
            0.7 * ingredient_coverage + 0.3 * expiry_urgency_score,
            4,
        )
        feedback_preference_score = 0.0
        recipe_id = str(doc.get("id") or "").strip()
        liked_recipe_ids = set(recent_feedback_signals.get("liked_recipe_ids", []))
        disliked_recipe_ids = set(recent_feedback_signals.get("disliked_recipe_ids", []))
        if recipe_id and recipe_id in liked_recipe_ids:
            feedback_preference_score = 1.0
        elif recipe_id and recipe_id in disliked_recipe_ids:
            feedback_preference_score = -1.0

        normalized_health_goal = _normalize_health_goal(health_goal)
        goal_fit_score = _compute_goal_fit_score(doc, normalized_health_goal)
        if not goal_fit_score and normalized_health_goal and normalized_health_goal in goal_tags:
            goal_fit_score = 1.0
        goal_conflict_score = _compute_goal_conflict_score(doc, normalized_health_goal)
        time_fit_score = 0.0
        if max_cooking_time and cook_time:
            if cook_time <= max_cooking_time:
                time_fit_score = 1.0
            else:
                time_fit_score = max(0.0, 1 - ((cook_time - max_cooking_time) / max(max_cooking_time, 1)))

        bias_bonus_breakdown: Dict[str, float] = {}
        inventory_bias_signal = inventory_priority_score if prefer_inventory_first else ingredient_match_score
        if "inventory_match" in rerank_bias:
            bias_bonus_breakdown["inventory_match"] = round(
                RERANK_BIAS_WEIGHTS["inventory_match"] * inventory_bias_signal,
                4,
            )
        if "goal_fit" in rerank_bias:
            bias_bonus_breakdown["goal_fit"] = round(
                RERANK_BIAS_WEIGHTS["goal_fit"] * goal_fit_score,
                4,
            )
        if "time_fit" in rerank_bias:
            bias_bonus_breakdown["time_fit"] = round(
                RERANK_BIAS_WEIGHTS["time_fit"] * time_fit_score,
                4,
            )
        if "feedback_preference" in rerank_bias and feedback_preference_score > 0:
            bias_bonus_breakdown["feedback_preference"] = round(
                RERANK_BIAS_WEIGHTS["feedback_preference"] * feedback_preference_score,
                4,
            )
        bias_bonus_score = round(sum(bias_bonus_breakdown.values()), 4)

        if prefer_inventory_first:
            final_score = (
                0.30 * retrieval_score
                + 0.40 * inventory_priority_score
                + 0.05 * max(feedback_preference_score, 0.0)
                + 0.20 * goal_fit_score
                + 0.05 * time_fit_score
                - 0.25 * goal_conflict_score
                + bias_bonus_score
            )
        else:
            final_score = (
                0.45 * retrieval_score
                + 0.25 * ingredient_match_score
                + 0.10 * feedback_preference_score
                + 0.20 * goal_fit_score
                + 0.10 * time_fit_score
                - 0.25 * goal_conflict_score
                + bias_bonus_score
            )

        return {
            "retrieval_score": round(retrieval_score, 4),
            "ingredient_match_score": round(ingredient_match_score, 4),
            "ingredient_coverage": round(ingredient_coverage, 4),
            "inventory_priority_score": inventory_priority_score,
            "expiry_urgency_score": expiry_urgency_score,
            "feedback_preference_score": round(feedback_preference_score, 4),
            "goal_fit_score": round(goal_fit_score, 4),
            "goal_conflict_score": goal_conflict_score,
            "time_fit_score": round(time_fit_score, 4),
            "inventory_match_ratio": round(ingredient_match_score, 4),
            "matched_ingredients": matched_ingredients,
            "missing_ingredients": missing_ingredients,
            "matched_inventory_count": len(matched_ingredients),
            "required_ingredient_count": len(ingredient_names),
            "matched_inventory_items": matched_inventory_items,
            "expiring_soon_ingredients": expiring_soon_ingredients,
            "prefer_inventory_first": prefer_inventory_first,
            "applied_rerank_bias": list(rerank_bias),
            "bias_bonus_breakdown": bias_bonus_breakdown,
            "bias_bonus_score": bias_bonus_score,
            "normalized_health_goal": normalized_health_goal,
            "feedback_summary": recent_feedback_signals.get("summary", ""),
            "final_score": round(final_score, 4),
        }

    def _get_user_context(
        self,
        user_id: Optional[str],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        获取并合并用户上下文
        
        从 PostgreSQL 获取用户偏好，与传入的上下文合并
        
        Args:
            user_id: 用户 ID
            user_context: 传入的用户上下文
        
        Returns:
            合并后的用户上下文
        """
        raw_context = user_context or {}
        merged_context = normalize_extracted_params(raw_context)
        passthrough_context = {
            key: value
            for key, value in raw_context.items()
            if key not in merged_context
        }

        if not self.enable_user_preferences or not user_id:
            return {**merged_context, **passthrough_context}

        try:
            # 获取用户偏好
            preferences = self.postgres_client.get_user_preferences(user_id)
            
            if preferences:
                logger.info(f"从数据库加载用户偏好: user_id={user_id}")
                
                # 合并偏好到上下文（传入的上下文优先级更高）
                if not merged_context.get("health_goal") and preferences.get("health_goal"):
                    merged_context["health_goal"] = preferences["health_goal"]
                
                if "difficulty" not in merged_context and preferences.get("preferred_difficulty"):
                    merged_context["difficulty"] = preferences["preferred_difficulty"]
                
                if not merged_context.get("max_cooking_time") and preferences.get("max_cooking_time"):
                    merged_context["max_cooking_time"] = preferences["max_cooking_time"]
                
                # 添加饮食限制和不喜欢的食材
                if not merged_context.get("allergies"):
                    merged_context["allergies"] = preferences.get("allergies", [])
                if not merged_context.get("disliked_ingredients"):
                    merged_context["disliked_ingredients"] = preferences.get("disliked_ingredients", [])
            
            # 获取用户的食材库存
            inventory = self.postgres_client.get_user_inventory(user_id)
            if inventory:
                passthrough_context["inventory_items"] = inventory
            if inventory and not merged_context.get("available_ingredients"):
                # 提取食材名称
                ingredient_names = [item["ingredient_name"] for item in inventory]
                merged_context["available_ingredients"] = ingredient_names
                logger.info(f"从库存加载 {len(ingredient_names)} 种食材")

            feedbacks = self.postgres_client.get_user_feedbacks(user_id, limit=5)
            if feedbacks:
                passthrough_context["recent_feedback_signals"] = _build_feedback_signals(feedbacks)
        
        except Exception as e:
            logger.error(f"获取用户上下文失败: {e}", exc_info=True)
        
        return {
            **normalize_extracted_params(merged_context),
            **passthrough_context,
        }
    
    def _log_interaction(
        self,
        user_id: str,
        query: str,
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ):
        """
        记录交互到数据库
        
        Args:
            user_id: 用户 ID
            query: 查询文本
            results: 推荐结果
            context: 用户上下文
        """
        try:
            recommended_recipes = [
                {
                    "id": r.get("id", ""),
                    "name": r.get("name", ""),
                    "final_score": r.get("final_score", r.get("score", 0.0)),
                }
                for r in results
            ]
            
            self.postgres_client.log_interaction(
                user_id=user_id,
                session_id=str(context.get("session_id") or "retrieval"),
                user_input=query,
                agent_response="",
                recommended_recipes=recommended_recipes,
                selected_recipe_id=recommended_recipes[0].get("id") if recommended_recipes else None,
                context=context
            )
            
            logger.debug(f"交互已记录: user_id={user_id}, recipes={len(recommended_recipes)}")
        
        except Exception as e:
            logger.error(f"记录交互失败: {e}", exc_info=True)
    
    def get_user_feedback_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户反馈统计
        
        Args:
            user_id: 用户 ID
        
        Returns:
            反馈统计信息
        """
        if not self.enable_user_preferences:
            return {}
        
        try:
            feedbacks = self.postgres_client.get_user_feedbacks(user_id)
            
            if not feedbacks:
                return {"total": 0, "average_rating": 0.0}
            
            total = len(feedbacks)
            avg_rating = sum(f["rating"] for f in feedbacks) / total
            
            return {
                "total": total,
                "average_rating": round(avg_rating, 2),
                "feedbacks": feedbacks
            }
        
        except Exception as e:
            logger.error(f"获取反馈统计失败: {e}", exc_info=True)
            return {}
