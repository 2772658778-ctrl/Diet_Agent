"""Session and lightweight user-state helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.graph.state import normalize_stable_user_preferences
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._session_history: dict[str, list[Any]] = {}
        self._session_history_by_session: dict[str, list[Any]] = {}
        self._session_index: dict[str, list[dict[str, Any]]] = {}
        self._feedback_memory: dict[str, list[dict[str, Any]]] = {}
        self._recommendation_memory: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def _build_feedback_signals(feedbacks: list[dict[str, Any]] | None) -> dict[str, Any]:
        liked_recipe_ids: list[str] = []
        disliked_recipe_ids: list[str] = []
        summary_parts: list[str] = []

        for item in feedbacks or []:
            if not isinstance(item, dict):
                continue
            recipe_id = str(item.get("recipe_id") or "").strip()
            rating = int(item.get("rating") or 0)
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

    @staticmethod
    def _build_session_key(user_id: str, session_id: str | None) -> str:
        normalized_user_id = str(user_id or "").strip()
        normalized_session_id = str(session_id or "").strip()
        if not normalized_user_id or not normalized_session_id:
            return ""
        return f"{normalized_user_id}:{normalized_session_id}"

    @staticmethod
    def _message_role(message: Any) -> str:
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "").strip().lower()
        else:
            role = str(getattr(message, "type", "") or "").strip().lower()
        if role == "human":
            return "user"
        if role == "ai":
            return "assistant"
        if role:
            return role
        return "assistant"

    @staticmethod
    def _message_content(message: Any) -> str:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content or "")

    def _update_session_index(self, user_id: str, session_id: str, messages: list[Any]) -> None:
        if not user_id or not session_id:
            return
        preview = ""
        for message in reversed(messages):
            content = self._message_content(message).strip()
            if content:
                preview = content[:80]
                break
        session_summary = {
            "session_id": session_id,
            "message_count": len(messages),
            "preview": preview,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "memory",
        }
        previous = [
            item
            for item in self._session_index.get(user_id, [])
            if str(item.get("session_id") or "") != session_id
        ]
        self._session_index[user_id] = [session_summary] + previous[:19]

    def get_history(self, user_id: str, session_id: str | None = None) -> list[Any]:
        if not user_id:
            return []
        session_key = self._build_session_key(user_id, session_id)
        if session_key:
            scoped_history = self._session_history_by_session.get(session_key)
            if scoped_history is not None:
                return list(scoped_history)
        return list(self._session_history.get(user_id, []))

    def set_history(
        self,
        user_id: str,
        messages: list[Any],
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> None:
        if not user_id:
            return
        trimmed_messages = list(messages or [])[-limit:]
        self._session_history[user_id] = trimmed_messages
        session_key = self._build_session_key(user_id, session_id)
        if session_key:
            self._session_history_by_session[session_key] = trimmed_messages
            self._update_session_index(user_id, str(session_id or "").strip(), trimmed_messages)

    def list_sessions(self, user_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if not user_id:
            return []
        sessions = list(self._session_index.get(user_id, []))
        if sessions:
            return sessions[:limit]
        history = self._session_history.get(user_id, [])
        if not history:
            return []
        preview = ""
        for message in reversed(history):
            content = self._message_content(message).strip()
            if content:
                preview = content[:80]
                break
        return [{
            "session_id": "",
            "message_count": len(history),
            "preview": preview,
            "updated_at": "",
            "source": "memory",
        }]

    def export_history(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        history = self.get_history(user_id, session_id=session_id)
        return [
            {
                "role": self._message_role(message),
                "content": self._message_content(message),
            }
            for message in history[-limit:]
        ]

    def remember_feedback_event(self, user_id: str, feedback: Any) -> None:
        if not user_id or feedback is None:
            return
        history = list(self._feedback_memory.get(user_id, []))
        liked = feedback.liked if feedback.liked is not None else feedback.rating >= 4
        history.append({
            "recipe_id": feedback.recipe_id,
            "rating": feedback.rating,
            "liked": liked,
            "comment": feedback.comment or "",
        })
        self._feedback_memory[user_id] = history[-10:]

    def get_recent_feedback_signals(self, user_id: str) -> dict[str, Any]:
        if not user_id:
            return {}
        return self._build_feedback_signals(self._feedback_memory.get(user_id, []))

    def remember_recommended_recipes(self, user_id: str, recipes: list[dict[str, Any]]) -> None:
        if not user_id or not recipes:
            return

        cleaned_recipes: list[dict[str, Any]] = []
        for item in recipes[:5]:
            if not isinstance(item, dict):
                continue
            recipe_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not recipe_id and not name:
                continue
            cleaned_recipes.append({
                "id": recipe_id,
                "name": name,
                "final_score": item.get("final_score", 0.0),
            })
        if cleaned_recipes:
            self._recommendation_memory[user_id] = cleaned_recipes

    def get_recent_recommended_recipes(self, user_id: str) -> list[dict[str, Any]]:
        if not user_id:
            return []
        return list(self._recommendation_memory.get(user_id, []))


_SESSION_STORE = InMemorySessionStore()


def get_session_store() -> InMemorySessionStore:
    return _SESSION_STORE


def load_stable_user_preferences(user_id: str) -> dict[str, Any]:
    if not user_id:
        return {}
    try:
        from diet_agent.integrations.database import get_postgres_client

        postgres_client = get_postgres_client()
        if postgres_client is None:
            return {}
        preferences = postgres_client.get_user_preferences(user_id) or {}
        return normalize_stable_user_preferences(preferences)
    except Exception as exc:
        logger.warning(f"加载稳定用户偏好失败: {exc}")
        return {}


__all__ = ["InMemorySessionStore", "get_session_store", "load_stable_user_preferences"]
