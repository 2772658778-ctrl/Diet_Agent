from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assets import get_skill_assets
from .contracts import SkillSpec
from .skills import get_skill_spec, select_skill_name

_DEFAULT_BUDGET_WEIGHTS: dict[str, float] = {
    "system": 0.15,
    "user_profile": 0.06,
    "history": 0.30,
    "retrieved_docs": 0.60,
    "query": 0.06,
}

_DEFAULT_TOKEN_CAPS: dict[str, int] = {
    "system": 500,
    "user_profile": 200,
    "history": 1000,
    "retrieved_docs": 2000,
    "query": 200,
}


@dataclass(slots=True)
class ContextAssemblyPolicy:
    budget_weights: dict[str, float] = field(default_factory=dict)
    token_caps: dict[str, int] = field(default_factory=dict)
    system_suffix_lines: tuple[str, ...] = ()
    docs_heading: str = "参考文档"
    query_heading: str = "用户查询"
    few_shot_examples: tuple[dict[str, str], ...] = ()
    few_shot_budget: int = 0


@dataclass(slots=True)
class SkillRuntimePolicyView:
    skill_name: str = ""
    skill_spec: SkillSpec | None = None
    required_tools: tuple[str, ...] = ()
    few_shot_examples: tuple[dict[str, str], ...] = ()
    clarification_policy: dict[str, Any] = field(default_factory=dict)
    planner_policy: dict[str, Any] = field(default_factory=dict)
    retrieval_profile: dict[str, Any] = field(default_factory=dict)
    response_contract: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    fallback_policy: dict[str, Any] = field(default_factory=dict)
    quality_rubric: dict[str, Any] = field(default_factory=dict)
    context_assembly: ContextAssemblyPolicy = field(default_factory=ContextAssemblyPolicy)


@dataclass(slots=True)
class SkillCapabilityStatus:
    skill_name: str = ""
    required_tools: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    missing_tools: tuple[str, ...] = ()
    is_ready: bool = True


@dataclass(slots=True)
class ResolvedSkillRuntime:
    skill_name: str = ""
    skill_spec: SkillSpec | None = None
    runtime_policy: SkillRuntimePolicyView = field(default_factory=SkillRuntimePolicyView)
    capability_status: SkillCapabilityStatus = field(default_factory=SkillCapabilityStatus)


def get_skill_runtime_policy(name: str | None) -> SkillRuntimePolicyView:
    skill_name = str(name or "").strip()
    skill_spec = get_skill_spec(skill_name) if skill_name else None
    if skill_spec is None:
        return SkillRuntimePolicyView(skill_name=skill_name)

    asset_view = get_skill_assets(skill_name)
    few_shot_examples = tuple(dict(item) for item in asset_view.few_shot_examples)
    return SkillRuntimePolicyView(
        skill_name=skill_name,
        skill_spec=skill_spec,
        required_tools=tuple(skill_spec.required_tools),
        few_shot_examples=few_shot_examples,
        clarification_policy=dict(skill_spec.clarification_policy),
        planner_policy=dict(skill_spec.planner_policy),
        retrieval_profile=dict(skill_spec.retrieval_profile),
        response_contract=dict(skill_spec.response_contract),
        evidence_policy=dict(skill_spec.evidence_policy),
        fallback_policy=dict(skill_spec.fallback_policy),
        quality_rubric=dict(asset_view.quality_rubric),
        context_assembly=build_context_assembly_policy(
            skill_spec,
            few_shot_examples=few_shot_examples,
        ),
    )


def list_available_tool_names() -> tuple[str, ...]:
    try:
        from src import tools as tools_module
    except Exception:
        tools_init_path = Path(__file__).resolve().parents[2] / "src" / "tools" / "__init__.py"
        try:
            module_ast = ast.parse(tools_init_path.read_text(encoding="utf-8"))
        except Exception:
            return ()

        for node in module_ast.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                return ()
            normalized_names: list[str] = []
            for element in node.value.elts:
                if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                    continue
                normalized_name = element.value.strip()
                if normalized_name and normalized_name not in normalized_names:
                    normalized_names.append(normalized_name)
            return tuple(normalized_names)
        return ()

    exported_names = getattr(tools_module, "__all__", ()) or ()
    normalized_names: list[str] = []
    for tool_name in exported_names:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name or normalized_name in normalized_names:
            continue
        if hasattr(tools_module, normalized_name):
            normalized_names.append(normalized_name)
    return tuple(normalized_names)


def get_skill_capability_status(
    name: str | None,
    *,
    available_tool_names: tuple[str, ...] | None = None,
) -> SkillCapabilityStatus:
    skill_name = str(name or "").strip()
    normalized_available_tools = (
        tuple(available_tool_names)
        if available_tool_names is not None
        else list_available_tool_names()
    )
    skill_spec = get_skill_spec(skill_name) if skill_name else None
    if skill_spec is None:
        return SkillCapabilityStatus(
            skill_name=skill_name,
            available_tools=normalized_available_tools,
        )

    required_tools = tuple(skill_spec.required_tools)
    missing_tools = tuple(
        tool_name for tool_name in required_tools if tool_name not in normalized_available_tools
    )
    return SkillCapabilityStatus(
        skill_name=skill_name,
        required_tools=required_tools,
        available_tools=normalized_available_tools,
        missing_tools=missing_tools,
        is_ready=not bool(missing_tools),
    )


def resolve_skill_runtime(intent: str, params: dict[str, Any] | None = None) -> ResolvedSkillRuntime:
    skill_name = select_skill_name(intent, params)
    runtime_policy = get_skill_runtime_policy(skill_name)
    return ResolvedSkillRuntime(
        skill_name=skill_name,
        skill_spec=runtime_policy.skill_spec,
        runtime_policy=runtime_policy,
        capability_status=get_skill_capability_status(skill_name),
    )


def requires_evidence_boundary(response_contract: dict | None, evidence_policy: dict | None) -> bool:
    response_contract = response_contract or {}
    evidence_policy = evidence_policy or {}
    return bool(
        response_contract.get("require_evidence_boundary")
        or evidence_policy.get("separate_evidence_from_general_advice")
    )


def detect_evidence_boundary_observed(
    response: str,
    *,
    response_type: str,
    requires_evidence_boundary: bool,
    separate_evidence_from_general_advice: bool,
) -> bool | None:
    if not requires_evidence_boundary:
        return None
    normalized_response = str(response or "").strip()
    if not normalized_response or response_type == "clarification":
        return False

    evidence_markers = (
        "当前证据",
        "证据不足",
        "证据边界",
        "参考文档",
        "文档支持",
        "证据结论",
        "直接支持",
    )
    general_advice_markers = (
        "通用建议",
        "一般建议",
        "额外建议",
    )
    has_evidence_marker = any(marker in normalized_response for marker in evidence_markers)
    has_general_advice_marker = any(marker in normalized_response for marker in general_advice_markers)
    if separate_evidence_from_general_advice:
        return has_evidence_marker and has_general_advice_marker
    return has_evidence_marker or has_general_advice_marker


def build_quality_rubric_observations(
    priorities: list[str],
    *,
    generation_metrics: dict | None = None,
    ragas_metrics: dict | None = None,
    constraint_hit_rate: float | None = None,
    task_completed: bool | None = None,
    fallback_triggered: bool = False,
    evidence_boundary_observed: bool | None = None,
    clarification_decision_correct: bool | None = None,
    hard_constraint_violation: bool = False,
    evaluation_pass: bool | None = None,
) -> dict[str, float]:
    generation_metrics = generation_metrics or {}
    ragas_metrics = ragas_metrics or {}
    observations: dict[str, float] = {}
    for priority in priorities:
        if priority == "faithfulness":
            faithfulness_score = ragas_metrics.get("faithfulness")
            if faithfulness_score is None:
                faithfulness_score = generation_metrics.get("faithfulness")
            if faithfulness_score is not None:
                observations[priority] = round(float(faithfulness_score), 4)
        elif priority == "answer_relevancy":
            answer_relevancy = generation_metrics.get("answer_relevancy")
            if answer_relevancy is None:
                answer_relevancy = ragas_metrics.get("answer_relevancy")
            if answer_relevancy is not None:
                observations[priority] = round(float(answer_relevancy), 4)
        elif priority == "evidence_boundary":
            if evidence_boundary_observed is not None:
                observations[priority] = 1.0 if evidence_boundary_observed else 0.0
        elif priority == "constraint_hit":
            if constraint_hit_rate is not None:
                observations[priority] = round(float(constraint_hit_rate), 4)
        elif priority == "grounded_recommendation":
            if task_completed is not None:
                observations[priority] = 1.0 if task_completed and not fallback_triggered else 0.0
        elif priority == "safety":
            observations[priority] = 0.0 if hard_constraint_violation else 1.0
        elif priority == "goal_fit":
            if constraint_hit_rate is not None:
                observations[priority] = round(float(constraint_hit_rate), 4)
        elif priority == "consistency":
            if clarification_decision_correct is not None:
                observations[priority] = 1.0 if clarification_decision_correct else 0.0
            elif evaluation_pass is not None:
                observations[priority] = 1.0 if evaluation_pass else 0.0
    return observations


def build_skill_quality_signals(
    skill_name: str | None,
    *,
    response: str = "",
    response_type: str = "recommendation",
    retrieval_stats: dict | None = None,
    evaluation: dict | None = None,
    generation_metrics: dict | None = None,
    ragas_metrics: dict | None = None,
    constraint_hit_rate: float | None = None,
    task_completed: bool | None = None,
    clarification_decision_correct: bool | None = None,
    hard_constraint_violation: bool = False,
) -> dict[str, Any]:
    normalized_skill_name = str(skill_name or "").strip()
    runtime_policy = get_skill_runtime_policy(normalized_skill_name)
    if runtime_policy.skill_spec is None and not normalized_skill_name:
        return {}

    retrieval_stats = retrieval_stats or {}
    evaluation = evaluation or {}
    priorities = [
        str(item).strip()
        for item in runtime_policy.quality_rubric.get("prioritize", []) or []
        if str(item).strip()
    ]
    evaluation_pass = evaluation.get("passed")
    if evaluation_pass is None:
        is_satisfactory = evaluation.get("is_satisfactory")
        if isinstance(is_satisfactory, bool):
            evaluation_pass = is_satisfactory

    fallback_triggered = bool(
        response_type == "fallback" or retrieval_stats.get("fallback_triggered", False)
    )
    clarification_triggered = response_type == "clarification"
    requires_boundary = requires_evidence_boundary(
        runtime_policy.response_contract,
        runtime_policy.evidence_policy,
    )
    evidence_boundary_observed = detect_evidence_boundary_observed(
        response,
        response_type=response_type,
        requires_evidence_boundary=requires_boundary,
        separate_evidence_from_general_advice=bool(
            runtime_policy.evidence_policy.get("separate_evidence_from_general_advice")
        ),
    )
    observations = build_quality_rubric_observations(
        priorities,
        generation_metrics=generation_metrics,
        ragas_metrics=ragas_metrics,
        constraint_hit_rate=constraint_hit_rate,
        task_completed=task_completed,
        fallback_triggered=fallback_triggered,
        evidence_boundary_observed=evidence_boundary_observed,
        clarification_decision_correct=clarification_decision_correct,
        hard_constraint_violation=hard_constraint_violation,
        evaluation_pass=evaluation_pass if isinstance(evaluation_pass, bool) else None,
    )
    quality_rubric_score = None
    if observations:
        quality_rubric_score = round(
            sum(observations.values()) / len(observations),
            4,
        )

    return {
        "active_skill": runtime_policy.skill_name,
        "requires_evidence_boundary": requires_boundary,
        "evidence_boundary_observed": evidence_boundary_observed,
        "fallback_triggered": fallback_triggered,
        "clarification_triggered": clarification_triggered,
        "quality_rubric_priorities": priorities,
        "quality_rubric_observations": observations,
        "quality_rubric_score": quality_rubric_score,
        "evaluation_pass": evaluation_pass,
    }


def build_context_assembly_policy(
    skill_spec: SkillSpec,
    *,
    few_shot_examples: tuple[dict[str, str], ...] = (),
) -> ContextAssemblyPolicy:
    response_contract = dict(skill_spec.response_contract)
    evidence_policy = dict(skill_spec.evidence_policy)
    fallback_policy = dict(skill_spec.fallback_policy)
    planner_policy = dict(skill_spec.planner_policy)

    budget_weights = dict(_DEFAULT_BUDGET_WEIGHTS)
    token_caps = dict(_DEFAULT_TOKEN_CAPS)
    system_suffix_lines: list[str] = []
    docs_heading = "参考文档"
    query_heading = "用户查询"

    require_evidence_boundary = bool(response_contract.get("require_evidence_boundary"))
    separate_evidence_from_general_advice = bool(
        evidence_policy.get("separate_evidence_from_general_advice")
    )
    low_evidence_mode = str(fallback_policy.get("on_low_evidence") or "").strip()
    is_subgraph_candidate = bool(response_contract.get("allow_subgraph")) or (
        str(planner_policy.get("default_next_action") or "").strip() == "subgraph_candidate"
    )

    if require_evidence_boundary or separate_evidence_from_general_advice:
        budget_weights.update(
            {
                "system": 0.18,
                "user_profile": 0.05,
                "history": 0.18,
                "retrieved_docs": 0.72,
                "query": 0.10,
            }
        )
        token_caps.update(
            {
                "system": 650,
                "user_profile": 160,
                "history": 700,
                "retrieved_docs": 2600,
                "query": 260,
            }
        )
        docs_heading = "证据参考文档"
        query_heading = "当前问题"
        system_suffix_lines.extend(
            [
                "回答时优先区分“参考文档直接支持的结论”和“通用建议”。",
                "如果证据不足，必须明确说明边界，不要把通用经验写成已被当前参考文档直接支持。",
            ]
        )
        if low_evidence_mode == "general_advice_only":
            system_suffix_lines.append(
                "当证据不足时，只能提供明确标注边界的通用建议，不要给出伪精确结论。"
            )
    elif is_subgraph_candidate or low_evidence_mode == "outline_only":
        budget_weights.update(
            {
                "system": 0.17,
                "user_profile": 0.07,
                "history": 0.22,
                "retrieved_docs": 0.66,
                "query": 0.10,
            }
        )
        token_caps.update(
            {
                "system": 620,
                "user_profile": 240,
                "history": 800,
                "retrieved_docs": 2300,
                "query": 260,
            }
        )
        system_suffix_lines.append(
            "如果当前证据不足以支撑完整计划，优先给出结构化提纲，而不是伪精确的完整细节。"
        )
    elif skill_spec.primary_intent == "recipe_search":
        budget_weights.update(
            {
                "system": 0.15,
                "user_profile": 0.07,
                "history": 0.24,
                "retrieved_docs": 0.64,
                "query": 0.10,
            }
        )
        token_caps.update(
            {
                "user_profile": 220,
                "history": 900,
                "retrieved_docs": 2200,
                "query": 240,
            }
        )

    few_shot_budget = 0
    if few_shot_examples:
        few_shot_budget = min(360, max(80, 120 * len(few_shot_examples)))

    return ContextAssemblyPolicy(
        budget_weights=budget_weights,
        token_caps=token_caps,
        system_suffix_lines=tuple(system_suffix_lines),
        docs_heading=docs_heading,
        query_heading=query_heading,
        few_shot_examples=few_shot_examples,
        few_shot_budget=few_shot_budget,
    )


__all__ = [
    "ContextAssemblyPolicy",
    "ResolvedSkillRuntime",
    "SkillCapabilityStatus",
    "SkillRuntimePolicyView",
    "build_quality_rubric_observations",
    "build_skill_quality_signals",
    "build_context_assembly_policy",
    "detect_evidence_boundary_observed",
    "get_skill_capability_status",
    "get_skill_runtime_policy",
    "list_available_tool_names",
    "requires_evidence_boundary",
    "resolve_skill_runtime",
]
