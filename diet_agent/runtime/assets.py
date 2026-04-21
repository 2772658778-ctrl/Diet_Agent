from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .skills import get_skill_spec, list_skill_specs


@dataclass(slots=True)
class SkillAssetView:
    skill_name: str = ""
    prompt_template: str = ""
    few_shot_examples: tuple[dict[str, str], ...] = ()
    quality_rubric: dict[str, Any] = field(default_factory=dict)
    ui_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_prompt_template(self) -> bool:
        return bool(self.prompt_template.strip())

    @property
    def few_shot_example_count(self) -> int:
        return len(self.few_shot_examples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_name,
            "prompt_template": self.prompt_template,
            "few_shot_examples": [dict(item) for item in self.few_shot_examples],
            "quality_rubric": dict(self.quality_rubric),
            "ui_metadata": dict(self.ui_metadata),
            "has_prompt_template": self.has_prompt_template,
            "few_shot_example_count": self.few_shot_example_count,
        }


def get_skill_assets(name: str | None) -> SkillAssetView:
    skill_name = str(name or "").strip()
    if not skill_name:
        return SkillAssetView()

    skill_spec = get_skill_spec(skill_name)
    if skill_spec is None:
        return SkillAssetView(skill_name=skill_name)

    return SkillAssetView(
        skill_name=skill_name,
        prompt_template=str(skill_spec.prompt_template or ""),
        few_shot_examples=tuple(dict(item) for item in skill_spec.few_shot_examples),
        quality_rubric=dict(skill_spec.quality_rubric),
        ui_metadata=dict(skill_spec.ui_metadata),
    )


def list_skill_assets() -> dict[str, SkillAssetView]:
    return {
        name: get_skill_assets(name)
        for name in list_skill_specs().keys()
    }


__all__ = ["SkillAssetView", "get_skill_assets", "list_skill_assets"]
