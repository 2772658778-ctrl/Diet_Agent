"""Persistent memory loop helpers for the reference app."""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_recipe_id(doc: dict[str, Any]) -> str:
    explicit_id = str(doc.get("id") or "").strip()
    if explicit_id:
        return explicit_id

    name = str(doc.get("name") or "").strip()
    if not name:
        return ""

    normalized_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"recipe_{normalized_name}"


def build_recommended_recipe_summaries(final_state: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    docs = list(final_state.get("reranked_docs", []) or final_state.get("retrieved_docs", []))[:limit]
    return [
        {
            "id": _resolve_recipe_id(doc),
            "name": doc.get("name", ""),
            "final_score": doc.get("final_score", doc.get("score", 0.0)),
        }
        for doc in docs
    ]


def write_memory_loopback(
    request: Any,
    request_id: str,
    response_text: str,
    final_state: dict[str, Any],
) -> tuple[str, bool]:
    user_id = getattr(request, "user_id", "") or ""
    if not user_id:
        return "", False

    try:
        from diet_agent.integrations.database import get_postgres_client

        postgres_client = get_postgres_client()
        if postgres_client is None:
            return "", False
        recommended_recipes = build_recommended_recipe_summaries(final_state)
        interaction_id = postgres_client.log_interaction(
            user_id=user_id,
            session_id=getattr(request, "session_id", None) or "",
            user_input=getattr(request, "query", ""),
            agent_response=response_text,
            recommended_recipes=recommended_recipes,
            selected_recipe_id="",
            context={
                "request_id": request_id,
                "intent": final_state.get("intent", ""),
                "response_type": final_state.get("response_type", "recommendation"),
                "planner_next_action": final_state.get("planner_next_action", "retrieve"),
                "followup_mode": final_state.get("followup_mode", ""),
            },
        ) or ""

        feedback_logged = False
        feedback = getattr(request, "feedback", None)
        if feedback is not None:
            liked = feedback.liked
            if liked is None:
                liked = feedback.rating >= 4
            postgres_client.add_feedback(
                user_id=user_id,
                interaction_id=interaction_id,
                recipe_id=feedback.recipe_id,
                rating=feedback.rating,
                liked=liked,
                taste_rating=feedback.taste_rating,
                difficulty_rating=feedback.difficulty_rating,
                time_accurate=feedback.time_accurate,
                comment=feedback.comment,
                tags=feedback.tags,
            )
            feedback_logged = True

        return interaction_id, feedback_logged
    except Exception as exc:
        logger.warning(f"写回交互/反馈失败，降级为内存闭环: {exc}")
        return "", False


def summarize_memory_readback(
    feedback_signals: dict[str, Any],
    recommended_recipes: list[dict[str, Any]],
    stable_user_preferences: dict[str, Any],
    applied_stable_preference_keys: list[str],
) -> dict[str, Any]:
    liked_recipe_ids = list(feedback_signals.get("liked_recipe_ids", []) or [])
    disliked_recipe_ids = list(feedback_signals.get("disliked_recipe_ids", []) or [])
    recommendation_anchor_count = len(recommended_recipes or [])
    feedback_signal_count = len(liked_recipe_ids) + len(disliked_recipe_ids)
    stable_preference_keys = sorted(str(key) for key in (stable_user_preferences or {}).keys())
    return {
        "feedback_signal_count": feedback_signal_count,
        "recommendation_anchor_count": recommendation_anchor_count,
        "feedback_summary": str(feedback_signals.get("summary") or ""),
        "memory_readback_ok": bool(feedback_signal_count or recommendation_anchor_count or stable_preference_keys),
        "stable_preference_count": len(stable_preference_keys),
        "stable_preference_keys": stable_preference_keys,
        "applied_stable_preference_keys": [
            str(key) for key in (applied_stable_preference_keys or []) if str(key)
        ],
    }


__all__ = ["build_recommended_recipe_summaries", "write_memory_loopback", "summarize_memory_readback"]
