"""
检索节点

封装 EnhancedRetrieverV3 + CrossEncoderReranker，支持通过 config.rag_strategy
在 'standard'（原有逻辑）和 'adaptive'（AdaptiveRAG）两种策略间切换。

策略选择：
- standard: 直接调用 EnhancedRetrieverV3（Phase 0 逻辑，向后兼容）
- adaptive: 先分类查询复杂度，再选择 simple/complex/ambiguous 策略检索

复用模块：
- src/retriever/enhanced_retriever_v3.py::EnhancedRetrieverV3
- src/reranker/cross_encoder_reranker.py::CrossEncoderReranker
- src/rag/adaptive_rag.py::AdaptiveRAG  (Phase 1 新增)
- src/vectorstore/chroma_client.py::get_vectorstore()
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

import re
from typing import Optional, TYPE_CHECKING

from diet_agent.runtime import get_skill_runtime_policy, select_skill_name
from ..state import DietAgentState, apply_stable_preferences, normalize_followup_contract, normalize_stable_user_preferences
from ...config import get_settings
from ...rag.query_features import extract_query_features
from ...utils.logger import get_logger

if TYPE_CHECKING:
    from ...retriever.enhanced_retriever_v3 import EnhancedRetrieverV3
    from ...reranker.cross_encoder_reranker import CrossEncoderReranker
    from ...rag.adaptive_rag import AdaptiveRAG


logger = get_logger(__name__)

# 模块级缓存
_retriever = None
_reranker = None
_adaptive_rag = None

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

TUTORIAL_FOLLOWUP_PHRASES = (
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
    "这个怎么做",
    "这道菜怎么做",
    "它怎么做",
)

TUTORIAL_QUERY_STRIP_MARKERS = (
    "怎么做",
    "做法",
    "步骤",
    "分步骤",
    "制作",
    "教程",
    "如何做",
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
    "这个",
    "这道菜",
    "这道",
    "它",
    "那个",
    "上一道",
    "上一个",
    "介绍一下",
    "简单介绍一下",
    "想吃",
    "请分步骤说明",
)


def _looks_like_followup_recipe_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(phrase in normalized_query for phrase in FOLLOWUP_RECOMMENDATION_PHRASES)


def _looks_like_tutorial_followup_request(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(phrase in normalized_query for phrase in TUTORIAL_FOLLOWUP_PHRASES)


def _extract_tutorial_topic_candidate(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return ""

    cleaned_text = normalized_text
    for phrase in sorted(TUTORIAL_QUERY_STRIP_MARKERS, key=len, reverse=True):
        cleaned_text = cleaned_text.replace(phrase, " ")

    candidates = [
        candidate.strip()
        for candidate in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", cleaned_text)
        if candidate.strip()
    ]
    for candidate in sorted(candidates, key=len, reverse=True):
        if len(candidate) < 2:
            continue
        if candidate not in {"这个", "这道菜", "这道", "它", "那个", "上一道", "上一个", "再"}:
            return candidate
    return ""


def _resolve_tutorial_topic_anchor(messages: list, user_query: str, extracted_params: dict) -> str:
    explicit_anchor = str(extracted_params.get("tutorial_topic_anchor") or "").strip()
    if explicit_anchor:
        return explicit_anchor

    direct_candidate = _extract_tutorial_topic_candidate(user_query)
    if direct_candidate:
        return direct_candidate

    prior_user_messages = []
    for message in messages[:-1]:
        message_type = getattr(message, "type", "")
        if message_type == "human":
            prior_user_messages.append(message)

    for message in reversed(prior_user_messages):
        candidate = _extract_tutorial_topic_candidate(
            message.content if hasattr(message, "content") else str(message)
        )
        if candidate:
            return candidate
    return ""


def _augment_followup_query(user_query: str, retrieval_context: dict) -> str:
    followup_mode = str(retrieval_context.get("followup_mode") or "").strip()
    if followup_mode not in {"anchored_followup", "generic_followup"} and not _looks_like_followup_recipe_request(user_query):
        return user_query

    anchor_names = [
        str(name).strip()
        for name in retrieval_context.get("followup_anchor_names", []) or []
        if str(name).strip()
    ]
    if not anchor_names:
        recent_recommended_recipes = retrieval_context.get("recent_recommended_recipes", []) or []
        anchor_names = [
            str(item.get("name") or "").strip()
            for item in recent_recommended_recipes[:3]
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
    if not anchor_names:
        return user_query

    style_instruction = "优先找相似方向但不要完全重复。"
    if followup_mode == "anchored_followup":
        style_instruction = "这是延续上一轮偏好的 follow-up 推荐；优先保持相近风格、时间和口味方向，但不要完全重复。"
    return f"{user_query}；参考上一轮推荐：{'、'.join(anchor_names)}；{style_instruction}."


def _augment_tutorial_query(user_query: str, retrieval_context: dict) -> str:
    if str(retrieval_context.get("active_skill") or "").strip() != "recipe_tutorial":
        return user_query

    tutorial_topic_anchor = str(retrieval_context.get("tutorial_topic_anchor") or "").strip()
    if not tutorial_topic_anchor or tutorial_topic_anchor in user_query:
        return user_query

    if _looks_like_tutorial_followup_request(user_query) or not _extract_tutorial_topic_candidate(user_query):
        return f"{user_query}；当前讨论的菜名：{tutorial_topic_anchor}；请围绕该菜的教程与步骤进行检索。"
    return user_query


def _resolve_active_skill(state: DietAgentState, intent: str, user_query: str, extracted_params: dict) -> str:
    active_skill = str(state.get("active_skill") or "").strip()
    if active_skill:
        return active_skill

    followup_mode = str(state.get("followup_mode") or "").strip()
    followup_anchor_names = [
        str(name).strip()
        for name in state.get("followup_anchor_names", []) or []
        if str(name).strip()
    ]
    followup_mode, followup_anchor_names = normalize_followup_contract(
        followup_mode,
        followup_anchor_names,
    )

    recent_feedback_signals = state.get("recent_feedback_signals", {}) or {}
    recent_recommended_recipes = state.get("recent_recommended_recipes", []) or []
    if not followup_mode and intent == "recipe_search" and _looks_like_followup_recipe_request(user_query):
        if recent_recommended_recipes or recent_feedback_signals.get("liked_recipe_ids") or recent_feedback_signals.get("disliked_recipe_ids"):
            followup_mode = "anchored_followup"
        else:
            followup_mode = "generic_followup"

    return select_skill_name(
        intent=intent,
        params={
            **extracted_params,
            "query": user_query,
            "tutorial_topic_anchor": str(extracted_params.get("tutorial_topic_anchor") or "").strip(),
            "followup_mode": followup_mode,
            "recent_feedback_signals": recent_feedback_signals,
            "recent_recommended_recipes": recent_recommended_recipes,
        },
    )


def _inject_runtime_retrieval_profile(retrieval_context: dict, active_skill: str) -> dict:
    if active_skill:
        retrieval_context["active_skill"] = active_skill

    runtime_policy = get_skill_runtime_policy(active_skill)
    retrieval_profile = dict(runtime_policy.retrieval_profile)
    if not retrieval_profile:
        return {}

    retrieval_context["retrieval_profile"] = retrieval_profile
    hard_filter_policy = str(retrieval_profile.get("hard_filter_policy") or "").strip()
    if hard_filter_policy:
        retrieval_context["hard_filter_policy"] = hard_filter_policy

    rerank_bias = retrieval_profile.get("rerank_bias")
    if isinstance(rerank_bias, (list, tuple)):
        normalized_rerank_bias = [
            str(item or "").strip()
            for item in rerank_bias
            if str(item or "").strip()
        ]
        if normalized_rerank_bias:
            retrieval_context["rerank_bias"] = normalized_rerank_bias

    return retrieval_profile


def _normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized_value = value.strip()
        return [normalized_value] if normalized_value else []
    if not isinstance(value, (list, tuple, set)):
        return []

    normalized_items: list[str] = []
    seen_items: set[str] = set()
    for item in value:
        normalized_item = str(item or "").strip()
        if not normalized_item or normalized_item in seen_items:
            continue
        seen_items.add(normalized_item)
        normalized_items.append(normalized_item)
    return normalized_items


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_primary_source(retrieval_profile: dict) -> str:
    explicit_source = str(retrieval_profile.get("primary_source") or "").strip().lower()
    if explicit_source in {"recipe", "tutorial"}:
        return explicit_source
    if str(retrieval_profile.get("profile") or "").strip() == "tutorial_search":
        return "tutorial"
    return "recipe"


def _resolve_secondary_sources(retrieval_profile: dict, primary_source: str | None = None) -> list[str]:
    resolved_primary_source = str(primary_source or _resolve_primary_source(retrieval_profile)).strip().lower()
    normalized_sources: list[str] = []
    for raw_source in _normalize_string_list(retrieval_profile.get("secondary_sources")):
        normalized_source = raw_source.lower()
        if normalized_source not in {"recipe", "tutorial"}:
            continue
        if normalized_source == resolved_primary_source or normalized_source in normalized_sources:
            continue
        normalized_sources.append(normalized_source)
    return normalized_sources


def _should_try_secondary_sources(primary_results: list[dict], primary_stats: dict, retrieval_profile: dict) -> bool:
    if not bool(retrieval_profile.get("allow_cross_source_fallback")):
        return False
    if not _resolve_secondary_sources(retrieval_profile):
        return False
    if primary_stats.get("fallback_triggered"):
        return True

    minimum_primary_doc_count = _safe_int(retrieval_profile.get("secondary_trigger_min_doc_count"), 0)
    if minimum_primary_doc_count <= 0:
        return not bool(primary_results)
    return len(primary_results) < minimum_primary_doc_count


def _resolve_secondary_max_docs(retrieval_profile: dict, default_count: int) -> int:
    configured_max_docs = _safe_int(retrieval_profile.get("secondary_max_docs"), 0)
    if configured_max_docs > 0:
        return configured_max_docs
    resolved_default = _safe_int(default_count, 0)
    if resolved_default > 0:
        return max(1, min(3, resolved_default))
    return 2


def _resolve_doc_entity_key(doc: dict) -> str:
    for field_name in ("name", "recipe_name", "tutorial_title", "id"):
        normalized_value = re.sub(r"\s+", "", str(doc.get(field_name) or "").strip()).lower()
        if normalized_value:
            return normalized_value
    return ""


def _annotate_retrieved_doc(doc: dict, *, source_name: str, source_role: str, retrieval_strategy: str) -> dict:
    normalized_source_name = str(source_name or "").strip().lower()
    doc_copy = dict(doc)
    matched_sources = _normalize_string_list(doc_copy.get("matched_sources"))
    if normalized_source_name and normalized_source_name not in matched_sources:
        matched_sources.append(normalized_source_name)
    if matched_sources:
        doc_copy["matched_sources"] = matched_sources
    if normalized_source_name and not str(doc_copy.get("knowledge_source") or "").strip():
        doc_copy["knowledge_source"] = normalized_source_name
    doc_copy["source_role"] = source_role
    if retrieval_strategy and not str(doc_copy.get("retrieval_strategy") or "").strip():
        doc_copy["retrieval_strategy"] = retrieval_strategy
    return doc_copy


def _merge_retrieved_docs(primary_doc: dict, supplemental_doc: dict) -> dict:
    merged_doc = dict(primary_doc)
    merged_doc["matched_sources"] = _normalize_string_list(
        _normalize_string_list(primary_doc.get("matched_sources"))
        + _normalize_string_list(supplemental_doc.get("matched_sources"))
    )

    primary_text = str(merged_doc.get("text") or "").strip()
    supplemental_text = str(supplemental_doc.get("text") or "").strip()
    if not primary_text and supplemental_text:
        merged_doc["text"] = supplemental_text
    elif supplemental_text and supplemental_text not in primary_text:
        merged_doc["text"] = f"{primary_text}\n{supplemental_text}".strip()

    primary_steps = [str(step).strip() for step in primary_doc.get("steps") or [] if str(step).strip()]
    supplemental_steps = [str(step).strip() for step in supplemental_doc.get("steps") or [] if str(step).strip()]
    combined_steps: list[str] = []
    seen_steps: set[str] = set()
    for step in primary_steps + supplemental_steps:
        if step in seen_steps:
            continue
        seen_steps.add(step)
        combined_steps.append(step)
    if combined_steps:
        merged_doc["steps"] = combined_steps[:8]

    for field_name in (
        "description",
        "difficulty",
        "time",
        "calories",
        "cuisine",
        "tags",
        "health_goals",
        "scenarios",
        "recipe_name",
        "tutorial_title",
        "collection_name",
        "source_type",
    ):
        if merged_doc.get(field_name) not in (None, "", [], ()): 
            continue
        supplemental_value = supplemental_doc.get(field_name)
        if supplemental_value not in (None, "", [], ()): 
            merged_doc[field_name] = supplemental_value

    if not merged_doc.get("description") and supplemental_doc.get("description"):
        merged_doc["description"] = supplemental_doc.get("description")
    if not merged_doc.get("score_breakdown") and supplemental_doc.get("score_breakdown"):
        merged_doc["score_breakdown"] = supplemental_doc.get("score_breakdown")
    if not merged_doc.get("final_score") and supplemental_doc.get("final_score"):
        merged_doc["final_score"] = supplemental_doc.get("final_score")

    merged_doc["matched_collection_names"] = _normalize_string_list(
        _normalize_string_list(primary_doc.get("matched_collection_names"))
        + _normalize_string_list(supplemental_doc.get("matched_collection_names"))
    )
    if str(merged_doc.get("source_role") or "").strip() == "primary" and str(supplemental_doc.get("source_role") or "").strip() == "secondary":
        merged_doc["source_role"] = "primary_with_secondary_support"
    return merged_doc


def _combine_cross_source_results(
    primary_results: list[dict],
    secondary_results: list[dict],
    *,
    primary_source: str,
    primary_strategy: str,
    secondary_source: str,
    secondary_strategy: str,
    secondary_max_docs: int,
) -> tuple[list[dict], int, int]:
    combined_results: list[dict] = []
    key_to_index: dict[str, int] = {}

    for doc in primary_results:
        annotated_doc = _annotate_retrieved_doc(
            doc,
            source_name=primary_source,
            source_role="primary",
            retrieval_strategy=primary_strategy,
        )
        combined_results.append(annotated_doc)
        entity_key = _resolve_doc_entity_key(annotated_doc)
        if entity_key and entity_key not in key_to_index:
            key_to_index[entity_key] = len(combined_results) - 1

    added_secondary_count = 0
    merged_overlap_count = 0
    for doc in secondary_results[:secondary_max_docs]:
        annotated_doc = _annotate_retrieved_doc(
            doc,
            source_name=secondary_source,
            source_role="secondary",
            retrieval_strategy=secondary_strategy,
        )
        entity_key = _resolve_doc_entity_key(annotated_doc)
        if entity_key and entity_key in key_to_index:
            combined_results[key_to_index[entity_key]] = _merge_retrieved_docs(
                combined_results[key_to_index[entity_key]],
                annotated_doc,
            )
            merged_overlap_count += 1
            continue
        combined_results.append(annotated_doc)
        if entity_key:
            key_to_index[entity_key] = len(combined_results) - 1
        added_secondary_count += 1

    return combined_results, added_secondary_count, merged_overlap_count


def _unpack_retrieval_payload(results) -> tuple[list[dict], dict]:
    if isinstance(results, tuple):
        payload_results = list(results[0] or [])
        payload_stats = dict(results[1] or {}) if len(results) > 1 else {}
        return payload_results, payload_stats
    return list(results or []), {}


def _build_secondary_recipe_context(retrieval_context: dict) -> dict:
    secondary_context = dict(retrieval_context)
    secondary_context["retrieval_profile"] = {
        "profile": "recipe_cross_source_fallback",
        "hard_filter_policy": "",
    }
    secondary_context.pop("hard_filter_policy", None)
    secondary_context.pop("rerank_bias", None)
    return secondary_context


def _maybe_apply_cross_source_fallback(
    primary_results: list[dict],
    primary_stats: dict,
    primary_strategy: str,
    retrieval_profile: dict,
    retrieval_query: str,
    retrieval_context: dict,
    settings,
    user_id: str,
) -> tuple[list[dict], dict, str]:
    primary_source = _resolve_primary_source(retrieval_profile)
    secondary_sources = _resolve_secondary_sources(retrieval_profile, primary_source)
    combined_results = [
        _annotate_retrieved_doc(
            doc,
            source_name=primary_source,
            source_role="primary",
            retrieval_strategy=primary_strategy,
        )
        for doc in primary_results
    ]
    combined_stats = dict(primary_stats or {})
    combined_stats["primary_source"] = primary_source
    combined_stats["secondary_sources"] = secondary_sources
    combined_stats["secondary_usage"] = str(retrieval_profile.get("secondary_usage") or "").strip()
    combined_stats["cross_source_fallback_attempted"] = False
    combined_stats["cross_source_fallback_used"] = False
    combined_stats["primary_returned_doc_count"] = len(primary_results)
    combined_stats["secondary_returned_doc_count"] = 0
    combined_stats["cross_source_added_doc_count"] = 0
    combined_stats["cross_source_merged_overlap_count"] = 0

    if not _should_try_secondary_sources(primary_results, combined_stats, retrieval_profile):
        combined_stats["fallback_triggered"] = not bool(combined_results)
        combined_stats["returned_doc_count"] = len(combined_results)
        return combined_results, combined_stats, primary_strategy

    combined_stats["cross_source_fallback_attempted"] = True
    secondary_reports: list[dict] = []
    tutorial_collection_names = _normalize_string_list(combined_stats.get("tutorial_collection_names"))
    matched_tutorial_collection_names = _normalize_string_list(combined_stats.get("matched_tutorial_collection_names"))
    evidence_direct_hit_count = _safe_int(combined_stats.get("evidence_direct_hit_count"), 0)
    tutorial_low_evidence = bool(combined_stats.get("tutorial_low_evidence", False))
    retrieval_strategies = [primary_strategy]
    secondary_max_docs = _resolve_secondary_max_docs(retrieval_profile, settings.graph_retriever_top_k)

    for secondary_source in secondary_sources:
        if secondary_source == "tutorial":
            secondary_results, secondary_stats, secondary_strategy = _retrieve_tutorial_docs(
                retrieval_query,
                settings,
                retrieval_profile,
                retrieval_context,
                direct_topic_match_required=False,
            )
        elif secondary_source == "recipe":
            secondary_payload, secondary_strategy = _fallback_standard(
                retrieval_query,
                user_id,
                _build_secondary_recipe_context(retrieval_context),
                settings,
            )
            secondary_results, secondary_stats = _unpack_retrieval_payload(secondary_payload)
        else:
            continue

        secondary_results = list(secondary_results or [])
        secondary_stats = dict(secondary_stats or {})
        secondary_reports.append(
            {
                "source": secondary_source,
                "strategy": secondary_strategy,
                "returned_doc_count": len(secondary_results),
                "fallback_triggered": bool(secondary_stats.get("fallback_triggered", False)),
            }
        )

        if secondary_source == "tutorial":
            tutorial_collection_names = _normalize_string_list(
                tutorial_collection_names + _normalize_string_list(secondary_stats.get("tutorial_collection_names"))
            )
            matched_tutorial_collection_names = _normalize_string_list(
                matched_tutorial_collection_names + _normalize_string_list(secondary_stats.get("matched_tutorial_collection_names"))
            )
            evidence_direct_hit_count += _safe_int(secondary_stats.get("evidence_direct_hit_count"), 0)
            tutorial_low_evidence = tutorial_low_evidence or bool(secondary_stats.get("tutorial_low_evidence", False))

        if not secondary_results:
            continue

        combined_results, added_secondary_count, merged_overlap_count = _combine_cross_source_results(
            combined_results,
            secondary_results,
            primary_source=primary_source,
            primary_strategy=primary_strategy,
            secondary_source=secondary_source,
            secondary_strategy=secondary_strategy,
            secondary_max_docs=secondary_max_docs,
        )
        combined_stats["cross_source_fallback_used"] = True
        combined_stats["secondary_returned_doc_count"] += len(secondary_results)
        combined_stats["cross_source_added_doc_count"] += added_secondary_count
        combined_stats["cross_source_merged_overlap_count"] += merged_overlap_count
        retrieval_strategies.append(secondary_strategy)

    combined_stats["secondary_retrievals"] = secondary_reports
    if tutorial_collection_names:
        combined_stats["tutorial_collection_names"] = tutorial_collection_names
    if matched_tutorial_collection_names:
        combined_stats["matched_tutorial_collection_names"] = matched_tutorial_collection_names
    if evidence_direct_hit_count:
        combined_stats["evidence_direct_hit_count"] = evidence_direct_hit_count
    if primary_source == "tutorial" or any(report.get("source") == "tutorial" for report in secondary_reports):
        combined_stats["tutorial_low_evidence"] = tutorial_low_evidence
    combined_stats["fallback_triggered"] = not bool(combined_results)
    combined_stats["returned_doc_count"] = len(combined_results)
    combined_strategy = "+".join(
        strategy_name
        for strategy_name in dict.fromkeys(str(item).strip() for item in retrieval_strategies if str(item).strip())
    )
    return combined_results, combined_stats, combined_strategy or primary_strategy


def _is_tutorial_retrieval_profile(retrieval_profile: dict) -> bool:
    return str(retrieval_profile.get("profile") or "").strip() == "tutorial_search"


def _retrieve_tutorial_docs(
    retrieval_query: str,
    settings,
    retrieval_profile: dict,
    retrieval_context: dict | None = None,
    *,
    direct_topic_match_required: bool = True,
) -> tuple[list[dict], dict, str]:
    from ...vectorstore.tutorial_store import search_tutorial_documents

    retrieval_context = retrieval_context or {}
    collection_names = [
        str(getattr(settings, "tutorial_collection_name", "") or "").strip(),
        str(getattr(settings, "bilibili_summary_collection_name", "") or "").strip(),
    ]
    normalized_collection_names: list[str] = []
    seen_collection_names: set[str] = set()
    for collection_name in collection_names:
        if not collection_name or collection_name in seen_collection_names:
            continue
        seen_collection_names.add(collection_name)
        normalized_collection_names.append(collection_name)

    results = search_tutorial_documents(
        query=retrieval_query,
        top_k=settings.graph_retriever_top_k,
        collection_names=normalized_collection_names,
        direct_topic_match_required=direct_topic_match_required,
    )
    matched_collection_names = list(
        dict.fromkeys(
            collection_name
            for doc in results
            for collection_name in doc.get("matched_collection_names", []) or []
            if str(collection_name).strip()
        )
    )
    evidence_direct_hit_count = sum(1 for doc in results if doc.get("direct_topic_match"))
    retrieval_stats = {
        "raw_candidate_count": len(results),
        "post_rerank_candidate_count": len(results),
        "returned_doc_count": len(results),
        "filtered_doc_count": 0,
        "hard_filter_reasons": {},
        "fallback_triggered": not bool(results),
        "tutorial_low_evidence": not bool(results),
        "tutorial_direct_topic_match_required": direct_topic_match_required,
        "tutorial_topic_anchor": str(retrieval_context.get("tutorial_topic_anchor") or "").strip(),
        "retrieval_profile": dict(retrieval_profile),
        "hard_filter_policy": str(retrieval_profile.get("hard_filter_policy") or ""),
        "rerank_bias": [],
        "tutorial_collection_name": getattr(settings, "tutorial_collection_name", ""),
        "tutorial_collection_names": normalized_collection_names,
        "matched_tutorial_collection_names": matched_collection_names,
        "evidence_direct_hit_count": evidence_direct_hit_count,
    }
    return results, retrieval_stats, "tutorial_vectorstore"


def _get_retriever():
    """获取检索器实例（单例模式）

    Returns:
        EnhancedRetrieverV3 实例，若向量库未初始化则返回 None
    """
    global _retriever
    if _retriever is not None:
        return _retriever

    from ...vectorstore.chroma_client import get_or_connect_vectorstore
    from ...database.postgres_client import get_postgres_client
    from ...retriever.enhanced_retriever_v3 import EnhancedRetrieverV3

    vectorstore = get_or_connect_vectorstore()
    if vectorstore is None:
        logger.warning("向量存储未初始化，无法创建检索器")
        return None

    reranker = _get_reranker()

    _retriever = EnhancedRetrieverV3(
        vectorstore=vectorstore,
        postgres_client=get_postgres_client(),
        reranker=reranker,
        enable_reranking=reranker is not None,
        enable_user_preferences=True,
    )
    logger.info("Graph 检索器初始化完成")
    return _retriever


def _get_reranker():
    """获取精排器实例（单例模式）

    Returns:
        CrossEncoderReranker 实例
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        from ...reranker.cross_encoder_reranker import CrossEncoderReranker

        settings = get_settings()
        _reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        logger.info(f"Graph 精排器初始化完成: {settings.reranker_model}")
        return _reranker
    except Exception as e:
        logger.error(f"精排器初始化失败: {e}", exc_info=True)
        return None


def _get_adaptive_rag():
    """获取 AdaptiveRAG 实例（单例模式，Phase 1 新增）

    Returns:
        AdaptiveRAG 实例，若检索器未初始化则返回 None
    """
    global _adaptive_rag
    if _adaptive_rag is not None:
        return _adaptive_rag

    retriever = _get_retriever()
    if retriever is None:
        logger.warning("检索器未初始化，无法创建 AdaptiveRAG")
        return None

    try:
        from ...rag.adaptive_rag import AdaptiveRAG

        reranker = _get_reranker()
        _adaptive_rag = AdaptiveRAG(
            retriever=retriever,
            reranker=reranker,
        )
        logger.info("AdaptiveRAG 初始化完成")
        return _adaptive_rag
    except Exception as e:
        logger.error(f"AdaptiveRAG 初始化失败: {e}", exc_info=True)
        return None


def retriever_node(state: DietAgentState) -> dict:
    """检索节点

    从 state 获取查询和参数，根据 config.rag_strategy 选择检索策略：
    - 'standard': 调用 EnhancedRetrieverV3（Phase 0 向后兼容）
    - 'adaptive': 调用 AdaptiveRAG（Phase 1 新增）

    Self-RAG 检索必要性判断和相关性过滤也在此节点内执行（若配置启用）。

    Args:
        state: LangGraph 全局状态

    Returns:
        包含 retrieved_docs, reranked_docs, query_complexity,
        retrieval_strategy, self_rag_judgements 的字典
    """
    logger.info("Retriever 节点开始执行")

    settings = get_settings()

    # 构建查询文本
    messages = state.get("messages", [])
    user_query = ""
    if messages:
        last_message = messages[-1]
        user_query = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

    stable_user_preferences = normalize_stable_user_preferences(
        state.get("stable_user_preferences", {})
    )
    extracted_params, applied_stable_preference_keys = apply_stable_preferences(
        state.get("extracted_params", {}),
        stable_user_preferences,
    )
    tutorial_topic_anchor = _resolve_tutorial_topic_anchor(messages, user_query, extracted_params)
    if tutorial_topic_anchor:
        extracted_params = {
            **extracted_params,
            "tutorial_topic_anchor": tutorial_topic_anchor,
        }
    intent = str(state.get("intent") or "").strip()
    active_skill = _resolve_active_skill(state, intent, user_query, extracted_params)
    query_features = extract_query_features(user_query, extracted_params)
    retrieval_context = {**extracted_params, "query_features": query_features}
    retrieval_profile = _inject_runtime_retrieval_profile(retrieval_context, active_skill)
    if stable_user_preferences:
        retrieval_context["stable_user_preferences"] = stable_user_preferences
    followup_mode = str(state.get("followup_mode") or "").strip()
    followup_anchor_names = [
        str(name).strip()
        for name in state.get("followup_anchor_names", []) or []
        if str(name).strip()
    ]
    followup_mode, followup_anchor_names = normalize_followup_contract(
        followup_mode,
        followup_anchor_names,
    )
    recent_feedback_signals = state.get("recent_feedback_signals", {}) or {}
    recent_recommended_recipes = state.get("recent_recommended_recipes", []) or []
    if not followup_mode and active_skill == "followup_recommendation" and _looks_like_followup_recipe_request(user_query):
        if recent_recommended_recipes or recent_feedback_signals.get("liked_recipe_ids") or recent_feedback_signals.get("disliked_recipe_ids"):
            followup_mode = "anchored_followup"
        else:
            followup_mode = "generic_followup"
    if followup_mode == "anchored_followup" and not followup_anchor_names:
        followup_anchor_names = [
            str(item.get("name") or "").strip()
            for item in recent_recommended_recipes[:3]
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
    followup_mode, followup_anchor_names = normalize_followup_contract(
        followup_mode,
        followup_anchor_names,
    )
    if followup_mode:
        retrieval_context["followup_mode"] = followup_mode
    if followup_anchor_names:
        retrieval_context["followup_anchor_names"] = followup_anchor_names
    if recent_feedback_signals:
        retrieval_context["recent_feedback_signals"] = recent_feedback_signals
    if recent_recommended_recipes:
        retrieval_context["recent_recommended_recipes"] = recent_recommended_recipes
    retrieval_query = _augment_followup_query(user_query, retrieval_context)
    retrieval_query = _augment_tutorial_query(retrieval_query, retrieval_context)

    if not user_query:
        logger.warning("查询为空，返回空结果")
        return {
            "retrieved_docs": [],
            "reranked_docs": [],
            "retrieval_stats": {},
            "query_complexity": "",
            "query_features": {},
            "retrieval_strategy": "standard",
            "self_rag_judgements": {},
            "active_skill": active_skill,
        }

    logger.info(
        f"检索查询: '{retrieval_query[:100]}', strategy={settings.rag_strategy}, active_skill={active_skill}"
    )
    logger.info(f"query_features={query_features}")

    self_rag_judgements: dict = {}
    skip_graph_eval = bool(state.get("skip_graph_eval", False))
    user_id = state.get("user_id", "")

    # ── Self-RAG: 检索必要性判断 ─────────────────────────────────────────────
    if _is_tutorial_retrieval_profile(retrieval_profile):
        tutorial_results, tutorial_stats, retrieval_strategy = _retrieve_tutorial_docs(
            retrieval_query,
            settings,
            retrieval_profile,
            retrieval_context,
        )
        tutorial_results, tutorial_stats, retrieval_strategy = _maybe_apply_cross_source_fallback(
            tutorial_results,
            tutorial_stats,
            retrieval_strategy,
            retrieval_profile,
            retrieval_query,
            retrieval_context,
            settings,
            user_id,
        )
        tutorial_stats["query_features"] = query_features
        tutorial_stats["entity_constraint_density"] = query_features.get("entity_constraint_density", 0.0)
        tutorial_stats["semantic_abstraction_score"] = query_features.get("semantic_abstraction_score", 0.0)
        tutorial_stats["inventory_signal"] = query_features.get("inventory_signal", 0.0)
        tutorial_stats["followup_mode"] = followup_mode
        tutorial_stats["followup_anchor_names"] = followup_anchor_names
        tutorial_stats["tutorial_topic_anchor"] = tutorial_topic_anchor
        tutorial_stats["stable_preference_keys"] = sorted(stable_user_preferences.keys())
        tutorial_stats["applied_stable_preference_keys"] = applied_stable_preference_keys
        tutorial_stats["active_skill"] = active_skill
        reranked = tutorial_results[:settings.graph_reranker_top_k]
        return {
            "retrieved_docs": tutorial_results,
            "reranked_docs": reranked,
            "retrieval_stats": tutorial_stats,
            "query_complexity": "",
            "query_features": query_features,
            "retrieval_strategy": retrieval_strategy,
            "self_rag_judgements": self_rag_judgements,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "active_skill": active_skill,
        }

    if settings.rag_strategy == "adaptive" and not skip_graph_eval:
        try:
            from ...rag.self_rag import get_self_rag_judge
            judge = get_self_rag_judge()

            history = state.get("messages", [])
            need_judgement = judge.judge_need_retrieval(user_query, history)
            self_rag_judgements["need_retrieval"] = {
                "need_retrieval": need_judgement.need_retrieval,
                "reason": need_judgement.reason,
            }
            if not need_judgement.need_retrieval:
                logger.info("Self-RAG: 无需检索，跳过检索步骤")
                return {
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "retrieval_stats": {},
                    "query_complexity": "simple",
                    "query_features": query_features,
                    "retrieval_strategy": "skip",
                    "self_rag_judgements": self_rag_judgements,
                    "active_skill": active_skill,
                }
        except Exception as e:
            logger.error(f"Self-RAG 检索必要性判断失败，继续检索: {e}", exc_info=True)

    # ── 执行检索 ─────────────────────────────────────────────────────────────
    query_complexity = ""
    retrieval_strategy = "standard"

    try:
        if settings.rag_strategy == "adaptive" and not skip_graph_eval:
            adaptive_rag = _get_adaptive_rag()
            if adaptive_rag is not None:
                results, retrieval_strategy = adaptive_rag.retrieve(
                    query=retrieval_query,
                    user_id=user_id,
                    user_context=retrieval_context,
                    top_k=settings.graph_retriever_top_k,
                )
                # 从 adaptive_rag 的 classify 结果读取 complexity
                # (strategy 名中含有 level 信息)
                if retrieval_strategy.startswith("complex"):
                    query_complexity = "complex"
                elif retrieval_strategy.startswith("ambiguous"):
                    query_complexity = "ambiguous"
                elif retrieval_strategy == "skip":
                    query_complexity = "simple"
                else:
                    query_complexity = "simple"
            else:
                logger.warning("AdaptiveRAG 未初始化，降级为 standard 策略")
                results, retrieval_strategy = _fallback_standard(
                    retrieval_query, user_id, retrieval_context, settings
                )
        else:
            if settings.rag_strategy == "adaptive" and skip_graph_eval:
                logger.info("命中 skip_graph_eval，retriever 走 standard fast smoke path")
            results, retrieval_strategy = _fallback_standard(
                retrieval_query, user_id, retrieval_context, settings
            )

        logger.info(f"检索完成: strategy={retrieval_strategy}, results={len(results)}")

        # ── Self-RAG: 相关性过滤 ──────────────────────────────────────────
        if settings.rag_strategy == "adaptive" and results:
            try:
                from ...rag.self_rag import get_self_rag_judge
                judge = get_self_rag_judge()

                if settings.fast_relevance_enabled:
                    # Phase 5: embedding 快速过滤
                    filtered = judge.judge_relevance_fast(
                        user_query, results, settings.fast_relevance_threshold
                    )
                else:
                    # Phase 1 原始逻辑: LLM 逐篇过滤（保留兼容）
                    filtered = judge.judge_relevance(user_query, results)

                self_rag_judgements["relevance_filter"] = {
                    "before": len(results),
                    "after": len(filtered),
                    "mode": "fast_embedding" if settings.fast_relevance_enabled else "llm",
                }
                if filtered:
                    results = filtered
                    logger.info(
                        f"Self-RAG 相关性过滤: {len(filtered)}/{len(results)} 通过 "
                        f"(mode={'fast' if settings.fast_relevance_enabled else 'llm'})"
                    )
            except Exception as e:
                logger.error(f"Self-RAG 相关性过滤失败，使用原始结果: {e}", exc_info=True)

        retrieval_stats = {}
        if isinstance(results, tuple):
            results, retrieval_stats = results

        retrieval_stats = dict(retrieval_stats or {})
        results, retrieval_stats, retrieval_strategy = _maybe_apply_cross_source_fallback(
            list(results or []),
            retrieval_stats,
            retrieval_strategy,
            retrieval_profile,
            retrieval_query,
            retrieval_context,
            settings,
            user_id,
        )
        retrieval_stats["query_features"] = query_features
        retrieval_stats["entity_constraint_density"] = query_features.get("entity_constraint_density", 0.0)
        retrieval_stats["semantic_abstraction_score"] = query_features.get("semantic_abstraction_score", 0.0)
        retrieval_stats["inventory_signal"] = query_features.get("inventory_signal", 0.0)
        retrieval_stats["followup_mode"] = followup_mode
        retrieval_stats["followup_anchor_names"] = followup_anchor_names
        retrieval_stats["stable_preference_keys"] = sorted(stable_user_preferences.keys())
        retrieval_stats["applied_stable_preference_keys"] = applied_stable_preference_keys
        retrieval_stats["active_skill"] = active_skill
        retrieval_stats["retrieval_profile"] = dict(retrieval_profile)
        retrieval_stats["hard_filter_policy"] = str(
            retrieval_stats.get("hard_filter_policy")
            or retrieval_profile.get("hard_filter_policy")
            or ""
        )
        retrieval_stats["rerank_bias"] = list(
            retrieval_stats.get("rerank_bias")
            or retrieval_context.get("rerank_bias")
            or []
        )

        reranked = results[:settings.graph_reranker_top_k]

        return {
            "retrieved_docs": results,
            "reranked_docs": reranked,
            "retrieval_stats": retrieval_stats,
            "query_complexity": query_complexity,
            "query_features": query_features,
            "retrieval_strategy": retrieval_strategy,
            "self_rag_judgements": self_rag_judgements,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "active_skill": active_skill,
        }

    except Exception as e:
        logger.error(f"Retriever 节点执行失败: {e}", exc_info=True)
        return {
            "retrieved_docs": [],
            "reranked_docs": [],
            "retrieval_stats": {},
            "query_complexity": "",
            "query_features": query_features,
            "retrieval_strategy": "error",
            "self_rag_judgements": self_rag_judgements,
            "extracted_params": extracted_params,
            "stable_user_preferences": stable_user_preferences,
            "applied_stable_preference_keys": applied_stable_preference_keys,
            "active_skill": active_skill,
        }


def _fallback_standard(
    user_query: str,
    user_id: str,
    extracted_params: dict,
    settings,
) -> tuple[list[dict], str]:
    """standard 策略降级实现，复用 EnhancedRetrieverV3。

    Args:
        user_query: 查询文本
        user_id: 用户 ID
        extracted_params: 提取的参数
        settings: 配置实例

    Returns:
        (检索结果列表, 策略名称)
    """
    retriever = _get_retriever()
    if retriever is None:
        logger.error("检索器未初始化，返回空结果")
        return [], "standard"

    results = retriever.retrieve(
        query=user_query,
        user_id=user_id,
        user_context=extracted_params,
        top_k=settings.graph_retriever_top_k,
    )
    return results, "standard"
