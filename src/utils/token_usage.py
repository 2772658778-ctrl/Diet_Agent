from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Iterator

import tiktoken

_TOKEN_USAGE_CONTEXT: ContextVar[dict[str, int] | None] = ContextVar(
    "token_usage_context",
    default=None,
)
_PROMPT_KEYS = (
    "prompt",
    "prompt_tokens",
    "input",
    "input_tokens",
    "inputTokenCount",
)
_COMPLETION_KEYS = (
    "completion",
    "completion_tokens",
    "output",
    "output_tokens",
    "outputTokenCount",
)
_TOTAL_KEYS = (
    "total",
    "total_tokens",
    "totalTokenCount",
)
_CALL_KEYS = ("llm_calls", "calls")
_NESTED_KEYS = (
    "usage",
    "token_usage",
    "usage_metadata",
    "response_metadata",
    "metadata",
)


@lru_cache(maxsize=32)
def _get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else 0
    return 0


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                payload = method()
            except TypeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in mapping:
            value = _coerce_int(mapping.get(key))
            if value:
                return value
    return 0


def _first_attr_int(value: Any, keys: tuple[str, ...]) -> int:
    for key in keys:
        if hasattr(value, key):
            result = _coerce_int(getattr(value, key))
            if result:
                return result
    return 0


def _build_usage(prompt: int, completion: int, total: int, llm_calls: int = 0) -> dict[str, int]:
    prompt_value = max(prompt, 0)
    completion_value = max(completion, 0)
    total_value = max(total, 0)
    if total_value <= 0 and (prompt_value or completion_value):
        total_value = prompt_value + completion_value
    if not (prompt_value or completion_value or total_value or llm_calls):
        return {}
    usage = {
        "prompt": prompt_value,
        "completion": completion_value,
        "total": total_value,
    }
    if llm_calls:
        usage["llm_calls"] = max(llm_calls, 0)
    return usage


def count_text_tokens(text: Any, model: str = "cl100k_base") -> int:
    if text in (None, ""):
        return 0
    return len(_get_encoding(str(model or "cl100k_base")).encode(str(text)))


def count_message_tokens(messages: Any, model: str = "cl100k_base") -> int:
    if not messages:
        return 0
    total = 0
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", message)
        total += 4
        total += count_text_tokens(content, model)
    return total + 2


def estimate_text_token_usage(prompt_text: Any, completion_text: Any, model: str = "cl100k_base") -> dict[str, int]:
    return merge_token_usage(
        {
            "prompt": count_text_tokens(prompt_text, model),
            "completion": count_text_tokens(completion_text, model),
        }
    )


def estimate_chat_token_usage(messages: Any, completion_text: Any, model: str = "cl100k_base") -> dict[str, int]:
    return merge_token_usage(
        {
            "prompt": count_message_tokens(messages, model),
            "completion": count_text_tokens(completion_text, model),
        }
    )


def extract_token_usage(value: Any) -> dict[str, int]:
    seen: set[int] = set()

    def _extract(obj: Any) -> dict[str, int]:
        if obj is None:
            return {}
        obj_id = id(obj)
        if obj_id in seen:
            return {}
        seen.add(obj_id)

        mapping = _to_mapping(obj)
        if mapping:
            prompt = _first_int(mapping, _PROMPT_KEYS)
            completion = _first_int(mapping, _COMPLETION_KEYS)
            total = _first_int(mapping, _TOTAL_KEYS)
            direct = _build_usage(prompt, completion, total)
            if direct:
                return direct
            for key in _NESTED_KEYS:
                if key in mapping:
                    nested_usage = _extract(mapping.get(key))
                    if nested_usage:
                        return nested_usage

        prompt = _first_attr_int(obj, _PROMPT_KEYS)
        completion = _first_attr_int(obj, _COMPLETION_KEYS)
        total = _first_attr_int(obj, _TOTAL_KEYS)
        direct_attrs = _build_usage(prompt, completion, total)
        if direct_attrs:
            return direct_attrs

        for attr_name in _NESTED_KEYS:
            if hasattr(obj, attr_name):
                nested_usage = _extract(getattr(obj, attr_name))
                if nested_usage:
                    return nested_usage

        return {}

    return _extract(value)


def normalize_token_usage(value: Any) -> dict[str, int]:
    usage = extract_token_usage(value)
    mapping = _to_mapping(value)
    llm_calls = _first_int(mapping, _CALL_KEYS) if mapping else 0
    if not usage and not llm_calls:
        return {}
    return _build_usage(
        usage.get("prompt", 0),
        usage.get("completion", 0),
        usage.get("total", 0),
        llm_calls=llm_calls,
    )


def merge_token_usage(*values: Any) -> dict[str, int]:
    prompt = 0
    completion = 0
    total = 0
    llm_calls = 0
    has_any = False
    for value in values:
        usage = normalize_token_usage(value)
        if not usage:
            continue
        has_any = True
        prompt += usage.get("prompt", 0)
        completion += usage.get("completion", 0)
        total += usage.get("total", 0)
        llm_calls += usage.get("llm_calls", 0)
    if not has_any:
        return {}
    return _build_usage(prompt, completion, total, llm_calls=llm_calls)


@contextmanager
def token_usage_scope(seed: Any = None) -> Iterator[dict[str, int]]:
    seeded = normalize_token_usage(seed)
    state = {
        "prompt": seeded.get("prompt", 0),
        "completion": seeded.get("completion", 0),
        "total": seeded.get("total", 0),
        "llm_calls": seeded.get("llm_calls", 0),
    }
    token = _TOKEN_USAGE_CONTEXT.set(state)
    try:
        yield state
    finally:
        _TOKEN_USAGE_CONTEXT.reset(token)


def add_token_usage(value: Any) -> dict[str, int]:
    usage = extract_token_usage(value)
    if not usage:
        return {}
    state = _TOKEN_USAGE_CONTEXT.get()
    if state is None:
        return _build_usage(
            usage.get("prompt", 0),
            usage.get("completion", 0),
            usage.get("total", 0),
            llm_calls=1,
        )
    state["prompt"] += usage.get("prompt", 0)
    state["completion"] += usage.get("completion", 0)
    state["total"] += usage.get("total", 0)
    state["llm_calls"] += 1
    return dict(state)


def get_current_token_usage() -> dict[str, int]:
    state = _TOKEN_USAGE_CONTEXT.get()
    if state is None:
        return {}
    return _build_usage(
        state.get("prompt", 0),
        state.get("completion", 0),
        state.get("total", 0),
        llm_calls=state.get("llm_calls", 0),
    )
