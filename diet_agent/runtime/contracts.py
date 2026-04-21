from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillSpec:
    skill_id: str
    intent_scope: tuple[str, ...]
    description: str = ""
    required_tools: tuple[str, ...] = ()
    required_slots: tuple[str, ...] = ()
    prompt_template: str = ""
    few_shot_examples: tuple[dict[str, str], ...] = ()
    clarification_policy: dict[str, Any] = field(default_factory=dict)
    planner_policy: dict[str, Any] = field(default_factory=dict)
    retrieval_profile: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    response_contract: dict[str, Any] = field(default_factory=dict)
    fallback_policy: dict[str, Any] = field(default_factory=dict)
    quality_rubric: dict[str, Any] = field(default_factory=dict)
    ui_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_intent(self) -> str:
        return self.intent_scope[0] if self.intent_scope else ""

    def supports_intent(self, intent: str) -> bool:
        return intent in self.intent_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "intent": self.primary_intent,
            "supported_intents": list(self.intent_scope),
            "intent_scope": list(self.intent_scope),
            "description": self.description,
            "required_tools": list(self.required_tools),
            "required_slots": list(self.required_slots),
            "prompt_template": self.prompt_template,
            "few_shot_examples": [dict(item) for item in self.few_shot_examples],
            "clarification_policy": dict(self.clarification_policy),
            "planner_policy": dict(self.planner_policy),
            "retrieval_profile": dict(self.retrieval_profile),
            "evidence_policy": dict(self.evidence_policy),
            "response_contract": dict(self.response_contract),
            "fallback_policy": dict(self.fallback_policy),
            "quality_rubric": dict(self.quality_rubric),
            "ui_metadata": dict(self.ui_metadata),
        }


def normalize_skill_spec(skill_id: str, skill_spec: SkillSpec | Mapping[str, Any]) -> SkillSpec:
    if isinstance(skill_spec, SkillSpec):
        if skill_spec.skill_id == skill_id:
            return skill_spec
        return normalize_skill_spec(skill_id, skill_spec.to_dict())

    if not isinstance(skill_spec, Mapping):
        raise TypeError("skill_spec must be a mapping or SkillSpec")

    intent_scope = _normalize_intent_scope(skill_spec)
    return SkillSpec(
        skill_id=skill_id,
        intent_scope=intent_scope,
        description=str(skill_spec.get("description") or ""),
        required_tools=_normalize_string_tuple(skill_spec.get("required_tools")),
        required_slots=_normalize_string_tuple(skill_spec.get("required_slots")),
        prompt_template=str(skill_spec.get("prompt_template") or ""),
        few_shot_examples=_normalize_examples(skill_spec.get("few_shot_examples")),
        clarification_policy=_normalize_dict(skill_spec.get("clarification_policy")),
        planner_policy=_normalize_dict(skill_spec.get("planner_policy")),
        retrieval_profile=_normalize_dict(skill_spec.get("retrieval_profile")),
        evidence_policy=_normalize_dict(skill_spec.get("evidence_policy")),
        response_contract=_normalize_dict(skill_spec.get("response_contract")),
        fallback_policy=_normalize_dict(skill_spec.get("fallback_policy")),
        quality_rubric=_normalize_dict(skill_spec.get("quality_rubric")),
        ui_metadata=_normalize_dict(skill_spec.get("ui_metadata")),
    )


def _normalize_intent_scope(skill_spec: Mapping[str, Any]) -> tuple[str, ...]:
    explicit_scope = skill_spec.get("intent_scope") or skill_spec.get("supported_intents")
    scope = _normalize_string_tuple(explicit_scope)
    if scope:
        return scope
    intent = str(skill_spec.get("intent") or "").strip()
    return (intent,) if intent else ()


def _normalize_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if not isinstance(value, (list, tuple, set)):
        return ()

    items: list[str] = []
    for item in value:
        normalized = str(item or "").strip()
        if normalized:
            items.append(normalized)
    return tuple(items)


def _normalize_examples(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    normalized_examples: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        normalized_examples.append(
            {
                "user": str(item.get("user") or ""),
                "assistant": str(item.get("assistant") or ""),
            }
        )
    return tuple(normalized_examples)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
