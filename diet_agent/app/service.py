"""App service helpers to keep the FastAPI layer thin."""

from __future__ import annotations

from typing import Any

from diet_agent.integrations.database import get_postgres_client
from diet_agent.runtime import (
    build_skill_quality_signals,
    get_skill_assets,
    get_skill_capability_status,
    get_skill_spec,
)
from src.graph.state import normalize_followup_contract

from diet_agent.user import build_recommended_recipe_summaries, summarize_memory_readback


def collect_request_memory_context(
    session_store: Any,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "history_messages": session_store.get_history(user_id, session_id=session_id),
        "recent_feedback_signals": session_store.get_recent_feedback_signals(user_id),
        "recent_recommended_recipes": session_store.get_recent_recommended_recipes(user_id),
    }


def _normalize_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


def build_recent_session_summaries(
    session_store: Any,
    user_id: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not user_id:
        return []
    session_map: dict[str, dict[str, Any]] = {}
    ordered_session_ids: list[str] = []

    for item in session_store.list_sessions(user_id, limit=limit):
        session_id = str(item.get("session_id") or "")
        session_map[session_id] = {
            "session_id": session_id,
            "message_count": int(item.get("message_count") or 0),
            "interaction_count": int(item.get("interaction_count") or 0),
            "preview": str(item.get("preview") or ""),
            "updated_at": str(item.get("updated_at") or ""),
            "last_interaction_at": str(item.get("last_interaction_at") or ""),
            "source": str(item.get("source") or "memory"),
        }
        ordered_session_ids.append(session_id)

    try:
        postgres_client = get_postgres_client()
    except Exception:
        postgres_client = None

    if postgres_client is not None:
        try:
            db_sessions = postgres_client.list_user_sessions(user_id, limit=limit)
        except Exception:
            db_sessions = []
        for item in db_sessions:
            session_id = str(item.get("session_id") or "")
            if session_id not in session_map:
                session_map[session_id] = {
                    "session_id": session_id,
                    "message_count": 0,
                    "interaction_count": 0,
                    "preview": "",
                    "updated_at": "",
                    "last_interaction_at": "",
                    "source": "postgres",
                }
                ordered_session_ids.append(session_id)
            session_summary = session_map[session_id]
            session_summary["interaction_count"] = int(
                item.get("interaction_count") or session_summary["interaction_count"] or 0
            )
            if not session_summary["preview"]:
                session_summary["preview"] = str(
                    item.get("last_query") or item.get("last_response_preview") or ""
                )[:80]
            last_interaction_at = _normalize_timestamp(item.get("last_interaction_at"))
            if last_interaction_at:
                session_summary["last_interaction_at"] = last_interaction_at
                if not session_summary["updated_at"]:
                    session_summary["updated_at"] = last_interaction_at
            if session_summary["source"] == "memory":
                session_summary["source"] = "hybrid"
            elif not session_summary["source"]:
                session_summary["source"] = "postgres"
    return [session_map[session_id] for session_id in ordered_session_ids[:limit]]


def build_session_history_view(
    session_store: Any,
    user_id: str,
    session_id: str,
    *,
    message_limit: int = 20,
    interaction_limit: int = 20,
) -> dict[str, Any]:
    messages = session_store.export_history(
        user_id,
        session_id=session_id,
        limit=message_limit,
    )
    used_memory_messages = bool(messages)

    try:
        postgres_client = get_postgres_client()
    except Exception:
        postgres_client = None

    interactions: list[dict[str, Any]] = []
    if postgres_client is not None and user_id and session_id:
        try:
            raw_interactions = postgres_client.get_session_interactions(
                user_id,
                session_id,
                limit=interaction_limit,
            )
        except Exception:
            raw_interactions = []
        interactions = [
            {
                "interaction_id": str(item.get("interaction_id") or ""),
                "session_id": str(item.get("session_id") or session_id),
                "user_input": str(item.get("user_input") or ""),
                "agent_response": str(item.get("agent_response") or ""),
                "recommended_recipes": list(item.get("recommended_recipes") or []),
                "context": dict(item.get("context") or {}),
                "created_at": _normalize_timestamp(item.get("created_at")),
            }
            for item in raw_interactions
        ]

    if not messages and interactions:
        rebuilt_messages: list[dict[str, str]] = []
        for item in reversed(interactions):
            if item["user_input"]:
                rebuilt_messages.append({"role": "user", "content": item["user_input"]})
            if item["agent_response"]:
                rebuilt_messages.append({"role": "assistant", "content": item["agent_response"]})
        messages = rebuilt_messages[-message_limit:]

    history_source = "empty"
    if used_memory_messages and interactions:
        history_source = "hybrid"
    elif messages:
        history_source = "memory" if used_memory_messages else "postgres"
    elif interactions:
        history_source = "postgres"

    return {
        "user_id": user_id,
        "session_id": session_id,
        "history_source": history_source,
        "messages": messages,
        "interactions": interactions,
    }


def hydrate_final_state(
    final_state: dict[str, Any],
    *,
    recent_feedback_signals: dict[str, Any],
    recent_recommended_recipes: list[dict[str, Any]],
    stable_user_preferences: dict[str, Any],
) -> dict[str, Any]:
    if recent_feedback_signals and not final_state.get("recent_feedback_signals"):
        final_state["recent_feedback_signals"] = recent_feedback_signals
    if recent_recommended_recipes and not final_state.get("recent_recommended_recipes"):
        final_state["recent_recommended_recipes"] = recent_recommended_recipes
    if stable_user_preferences and not final_state.get("stable_user_preferences"):
        final_state["stable_user_preferences"] = dict(stable_user_preferences)
    return final_state


def sync_memory_artifacts(
    final_state: dict[str, Any],
    *,
    user_id: str,
    session_store: Any,
    request_feedback: Any,
    recommended_recipes: list[dict[str, Any]],
    stable_user_preferences: dict[str, Any],
) -> dict[str, Any]:
    if user_id and recommended_recipes:
        session_store.remember_recommended_recipes(user_id, recommended_recipes)

    effective_feedback_signals = final_state.get("recent_feedback_signals", {}) or {}
    if request_feedback is not None and user_id:
        effective_feedback_signals = session_store.get_recent_feedback_signals(user_id)
    elif not effective_feedback_signals:
        effective_feedback_signals = session_store.get_recent_feedback_signals(user_id)

    effective_recommendation_anchors = (
        recommended_recipes
        or final_state.get("recent_recommended_recipes", [])
        or session_store.get_recent_recommended_recipes(user_id)
    )

    if effective_feedback_signals:
        final_state["recent_feedback_signals"] = dict(effective_feedback_signals)
    elif "recent_feedback_signals" not in final_state:
        final_state["recent_feedback_signals"] = {}

    if effective_recommendation_anchors:
        final_state["recent_recommended_recipes"] = [dict(item) for item in effective_recommendation_anchors]
    elif "recent_recommended_recipes" not in final_state:
        final_state["recent_recommended_recipes"] = []

    followup_mode, followup_anchor_names = normalize_followup_contract(
        final_state.get("followup_mode", ""),
        final_state.get("followup_anchor_names", []),
    )
    final_state["followup_mode"] = followup_mode
    final_state["followup_anchor_names"] = followup_anchor_names

    return {
        "effective_feedback_signals": effective_feedback_signals if user_id else {},
        "effective_recommendation_anchors": effective_recommendation_anchors if user_id else [],
        "followup_mode": followup_mode,
        "followup_anchor_names": followup_anchor_names,
        "recommended_recipes": recommended_recipes,
    }


def build_chat_metadata(
    final_state: dict[str, Any],
    *,
    prev_messages: list[Any],
    retrieval_stats: dict[str, Any],
    evaluation: dict[str, Any],
    missing_slots: list[str],
    interaction_id: str,
    feedback_logged: bool,
    recommended_recipes: list[dict[str, Any]],
    effective_feedback_signals: dict[str, Any],
    effective_recommendation_anchors: list[dict[str, Any]],
    stable_user_preferences: dict[str, Any],
) -> dict[str, Any]:
    next_expected_slot = missing_slots[0] if missing_slots else ""
    fallback_triggered = bool(
        final_state.get("response_type") == "fallback"
        or retrieval_stats.get("fallback_triggered", False)
    )
    active_skill = str(final_state.get("active_skill") or "")
    skill_contract = {}
    skill_capability = {}
    quality_signals = build_skill_quality_signals(
        active_skill,
        response=str(final_state.get("response") or ""),
        response_type=str(final_state.get("response_type") or "recommendation"),
        retrieval_stats=retrieval_stats,
        evaluation=evaluation if isinstance(evaluation, dict) else {},
    )
    if active_skill:
        skill_spec = get_skill_spec(active_skill)
        asset_view = get_skill_assets(active_skill)
        capability_status = get_skill_capability_status(active_skill)
        skill_capability = {
            "required_tools": list(capability_status.required_tools),
            "available_tools": list(capability_status.available_tools),
            "missing_tools": list(capability_status.missing_tools),
            "is_ready": capability_status.is_ready,
        }
        if skill_spec is not None:
            skill_contract = {
                "skill_id": skill_spec.skill_id,
                "intent_scope": list(skill_spec.intent_scope),
                "required_tools": list(skill_spec.required_tools),
                "required_slots": list(skill_spec.required_slots),
                "retrieval_profile": dict(skill_spec.retrieval_profile),
                "response_contract": dict(skill_spec.response_contract),
                "evidence_policy": dict(skill_spec.evidence_policy),
                "fallback_policy": dict(skill_spec.fallback_policy),
                "quality_rubric": dict(asset_view.quality_rubric),
                "ui_metadata": dict(asset_view.ui_metadata),
                "has_prompt_template": asset_view.has_prompt_template,
                "few_shot_example_count": asset_view.few_shot_example_count,
                "requires_evidence_boundary": bool(
                    quality_signals.get("requires_evidence_boundary", False)
                ),
                "capability_status": dict(skill_capability),
            }
    applied_stable_preference_keys = final_state.get("applied_stable_preference_keys", []) or []
    memory_metadata = summarize_memory_readback(
        feedback_signals=effective_feedback_signals,
        recommended_recipes=effective_recommendation_anchors,
        stable_user_preferences=final_state.get("stable_user_preferences", {}) or stable_user_preferences,
        applied_stable_preference_keys=applied_stable_preference_keys,
    )
    retrieved = final_state.get("retrieved_docs", [])
    reranked = final_state.get("reranked_docs", [])
    return {
        "extracted_params": final_state.get("extracted_params", {}),
        "current_step": final_state.get("current_step", 0),
        "retry_count": final_state.get("retry_count", 0),
        "goal_type": final_state.get("goal_type", ""),
        "planner_next_action": final_state.get("planner_next_action", "retrieve"),
        "inherit_followup_direction": final_state.get("inherit_followup_direction", False),
        "response_type": final_state.get("response_type", "recommendation"),
        "clarification_needed": final_state.get("clarification_needed", False),
        "clarification_question": final_state.get("clarification_question", ""),
        "missing_slots": missing_slots,
        "next_expected_slot": next_expected_slot,
        "fallback_triggered": fallback_triggered,
        "followup_mode": final_state.get("followup_mode", ""),
        "followup_anchor_names": final_state.get("followup_anchor_names", []),
        "retrieval_stats": retrieval_stats,
        "active_skill": active_skill,
        "skill_contract": skill_contract,
        "skill_capability": skill_capability,
        "history_message_count": len(prev_messages),
        "interaction_id": interaction_id,
        "feedback_logged": feedback_logged,
        "recommended_recipes": recommended_recipes,
        "retrieval": {
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "constraint_count": retrieval_stats.get("constraint_count", 0),
            "filtered_doc_count": retrieval_stats.get("filtered_doc_count", 0),
            "hard_filter_reasons": retrieval_stats.get("hard_filter_reasons", {}),
            "inventory_match_ratio": retrieval_stats.get("inventory_match_ratio", 0.0),
            "goal_fit_score": retrieval_stats.get("goal_fit_score", 0.0),
            "expiry_urgency_score": retrieval_stats.get("expiry_urgency_score", 0.0),
            "fallback_triggered": fallback_triggered,
        },
        "memory": memory_metadata,
        "evaluation": evaluation if isinstance(evaluation, dict) else {},
        "quality_signals": quality_signals,
    }


__all__ = [
    "build_recent_session_summaries",
    "build_session_history_view",
    "build_chat_metadata",
    "build_recommended_recipe_summaries",
    "collect_request_memory_context",
    "hydrate_final_state",
    "sync_memory_artifacts",
]
