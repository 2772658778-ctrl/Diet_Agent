import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Literal

from diet_agent.runtime import build_skill_quality_signals, run_diet_agent

from pydantic import BaseModel, Field

from ..config import get_settings
from ..graph.schemas import normalize_extracted_params
from ..utils.logger import get_logger
from ..utils.token_usage import get_current_token_usage, merge_token_usage, token_usage_scope
from .generation_metrics import GenerationMetrics
from .ragas_eval import RAGASEvaluator
from .retrieval_metrics import RetrievalMetrics

logger = get_logger(__name__)

IntentName = Literal["recipe_search", "nutrition_query", "ingredient_check", "chitchat"]


class QuantBenchmarkCase(BaseModel):
    case_id: str
    query: str
    expected_intent: IntentName
    scenario_type: str = "generic"
    expected_constraints: dict = Field(default_factory=dict)
    expected_active_skill: str = ""
    expected_planner_next_action: str = ""
    expected_response_type: str = ""
    should_clarify: bool = False
    expected_missing_slots: list[str] = Field(default_factory=list)
    ground_truth_doc_ids: list[str] = Field(default_factory=list)
    ground_truth_answer_keywords: list[str] = Field(default_factory=list)
    ground_truth_answer: str = ""
    notes: str = ""


class QuantBenchmarkResult(BaseModel):
    case_id: str
    query: str
    scenario_type: str
    expected_intent: str
    actual_intent: str = ""
    response: str = ""
    response_type: str = ""
    active_skill: str = ""
    planner_next_action: str = ""
    extracted_params: dict = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieval_metrics: dict = Field(default_factory=dict)
    generation_metrics: dict = Field(default_factory=dict)
    ragas_metrics: dict = Field(default_factory=dict)
    token_usage: dict = Field(default_factory=dict)
    evaluation_token_usage: dict = Field(default_factory=dict)
    overall_token_usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    agent_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    fallback_triggered: bool = False
    clarification_triggered: bool = False
    hard_constraint_violation: bool = False
    system_success: bool = False
    task_completed: bool = False
    intent_correct: bool = False
    active_skill_correct: bool | None = None
    planner_action_correct: bool | None = None
    clarification_decision_correct: bool = False
    requires_evidence_boundary: bool = False
    evidence_boundary_observed: bool | None = None
    quality_rubric_priorities: list[str] = Field(default_factory=list)
    quality_rubric_observations: dict = Field(default_factory=dict)
    quality_rubric_score: float | None = None
    constraint_hit_rate: float = 0.0
    tool_call_accuracy: float = 0.0
    hallucination_detected: bool | None = None
    error: str = ""


class QuantitativeBenchmark:
    def __init__(
        self,
        dataset_path: str,
        enable_generation_metrics: bool = True,
        enable_ragas_metrics: bool = True,
    ) -> None:
        self._settings = get_settings()
        self._dataset_path = dataset_path
        self._gen_metrics = None
        self._ragas = None
        if enable_generation_metrics:
            try:
                self._gen_metrics = GenerationMetrics()
            except Exception as exc:
                logger.warning(f"GenerationMetrics 初始化失败，将跳过生成质量指标: {exc}")
        if enable_ragas_metrics:
            try:
                self._ragas = RAGASEvaluator()
            except Exception as exc:
                logger.warning(f"RAGASEvaluator 初始化失败，将跳过 RAGAS 指标: {exc}")
        self._dataset: list[QuantBenchmarkCase] = []
        if os.path.exists(dataset_path):
            self._dataset = self.load_dataset(dataset_path)

    def load_dataset(self, path: str) -> list[QuantBenchmarkCase]:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        dataset: list[QuantBenchmarkCase] = []
        for index, item in enumerate(raw_data, 1):
            payload = dict(item)
            if "case_id" not in payload:
                payload["case_id"] = str(payload.get("id") or f"case_{index:03d}")
            dataset.append(QuantBenchmarkCase(**payload))
        return dataset

    def _resolve_cases(self, case_ids: list[str] | None = None) -> list[QuantBenchmarkCase]:
        if not case_ids:
            return list(self._dataset)

        requested_case_ids = [str(case_id).strip() for case_id in case_ids if str(case_id).strip()]
        if not requested_case_ids:
            return list(self._dataset)

        requested_case_id_set = set(requested_case_ids)
        selected_cases = [case for case in self._dataset if case.case_id in requested_case_id_set]
        selected_case_id_set = {case.case_id for case in selected_cases}
        missing_case_ids = [case_id for case_id in requested_case_ids if case_id not in selected_case_id_set]
        if missing_case_ids:
            logger.warning(f"以下 case_id 未在数据集中找到，将被忽略: {missing_case_ids}")
        if not selected_cases:
            raise ValueError(f"未找到任何匹配的 case_id: {requested_case_ids}")
        return selected_cases

    @staticmethod
    def _normalize_text_set(value: object) -> set[str]:
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
            if isinstance(item, dict):
                text = str(item.get("name") or item.get("ingredient") or "").strip().lower()
            else:
                text = str(item).strip().lower()
            if text:
                normalized.add(text)
        return normalized

    @staticmethod
    def _safe_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return int(digits) if digits else None

    @staticmethod
    def _active_expected_constraint_items(expected_constraints: dict) -> list[tuple[str, object]]:
        active_items: list[tuple[str, object]] = []
        for key, value in expected_constraints.items():
            if isinstance(value, list) and value:
                active_items.append((key, value))
            elif value not in (None, "", False, []):
                active_items.append((key, value))
        return active_items

    @classmethod
    def _constraint_matches(cls, actual: dict, key: str, expected_value: object) -> bool:
        actual_value = actual.get(key)
        if isinstance(expected_value, list):
            actual_list = [str(item).strip().lower() for item in (actual_value or [])]
            return all(str(item).strip().lower() in actual_list for item in expected_value)
        if isinstance(expected_value, bool):
            return bool(actual_value) == expected_value
        if expected_value is None:
            return actual_value is None
        return str(actual_value).strip().lower() == str(expected_value).strip().lower()

    @classmethod
    def _is_hard_constraint_violation(cls, top_doc: dict, expected_constraints: dict) -> bool:
        if not top_doc:
            return False
        allergies = cls._normalize_text_set(expected_constraints.get("allergies"))
        disliked = cls._normalize_text_set(expected_constraints.get("disliked_ingredients"))
        ingredient_names = cls._normalize_text_set(top_doc.get("ingredient_names", top_doc.get("ingredients", [])))
        allergens = cls._normalize_text_set(top_doc.get("allergens", []))
        combined = ingredient_names | allergens
        if allergies and combined.intersection(allergies):
            return True
        if disliked and ingredient_names.intersection(disliked):
            return True
        max_cooking_time = expected_constraints.get("max_cooking_time")
        cook_time = cls._safe_int(top_doc.get("cook_time", top_doc.get("time")))
        if max_cooking_time and cook_time and cook_time > max_cooking_time * 1.5:
            return True
        return False

    @staticmethod
    def _get_top_doc(state: dict) -> dict:
        reranked = state.get("reranked_docs") or []
        retrieved = state.get("retrieved_docs") or []
        docs = reranked or retrieved
        return dict(docs[0]) if docs else {}

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(len(ordered) * p / 100), len(ordered) - 1)
        return ordered[index]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _rate(values: list[bool]) -> float:
        return sum(1 for value in values if value) / len(values) if values else 0.0

    @staticmethod
    def _get_token_value(token_usage: dict, *keys: str) -> int:
        for key in keys:
            value = token_usage.get(key)
            if isinstance(value, int):
                return value
        return 0

    @classmethod
    def _token_series(cls, token_usages: list[dict], *keys: str) -> list[int]:
        return [
            cls._get_token_value(token_usage, *keys)
            for token_usage in token_usages
            if token_usage
        ]

    @classmethod
    def _build_token_summary(cls, token_usages: list[dict]) -> dict[str, float]:
        normalized_usages = [merge_token_usage(token_usage) for token_usage in token_usages if token_usage]
        prompt_values = cls._token_series(normalized_usages, "prompt", "prompt_tokens")
        completion_values = cls._token_series(normalized_usages, "completion", "completion_tokens")
        total_values = cls._token_series(normalized_usages, "total", "total_tokens")
        llm_call_values = cls._token_series(normalized_usages, "llm_calls", "calls")
        return {
            "prompt_tokens_mean": round(cls._mean(prompt_values), 2),
            "completion_tokens_mean": round(cls._mean(completion_values), 2),
            "total_tokens_mean": round(cls._mean(total_values), 2),
            "llm_calls_mean": round(cls._mean(llm_call_values), 2),
            "prompt_tokens_sum": int(sum(prompt_values)),
            "completion_tokens_sum": int(sum(completion_values)),
            "total_tokens_sum": int(sum(total_values)),
            "llm_calls_sum": int(sum(llm_call_values)),
        }

    def _compute_tool_call_accuracy(
        self,
        intent_correct: bool,
        active_skill_correct: bool | None,
        planner_action_correct: bool | None,
        constraint_hit_rate: float | None,
        clarification_decision_correct: bool,
    ) -> float:
        weighted_values: list[tuple[float, float]] = [(1.0 if intent_correct else 0.0, 0.30)]
        if active_skill_correct is not None:
            weighted_values.append((1.0 if active_skill_correct else 0.0, 0.20))
        if planner_action_correct is not None:
            weighted_values.append((1.0 if planner_action_correct else 0.0, 0.20))
        if constraint_hit_rate is not None:
            weighted_values.append((constraint_hit_rate, 0.20))
        weighted_values.append((1.0 if clarification_decision_correct else 0.0, 0.10))
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight <= 0:
            return 0.0
        return sum(value * weight for value, weight in weighted_values) / total_weight

    @staticmethod
    def _requires_grounded_faithfulness(case: QuantBenchmarkCase) -> bool:
        return case.expected_intent != "chitchat"

    @staticmethod
    def _response_contains_expected_keywords(response: str, expected_keywords: list[str]) -> bool:
        normalized_response = str(response or "").strip().lower()
        if not normalized_response:
            return False
        normalized_keywords = [
            str(keyword).strip().lower()
            for keyword in expected_keywords or []
            if str(keyword).strip()
        ]
        if not normalized_keywords:
            return True
        return all(keyword in normalized_response for keyword in normalized_keywords)

    @staticmethod
    def _requires_evidence_boundary(response_contract: dict, evidence_policy: dict) -> bool:
        return bool(
            response_contract.get("require_evidence_boundary")
            or evidence_policy.get("separate_evidence_from_general_advice")
        )

    @staticmethod
    def _detect_evidence_boundary_observed(
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

    @staticmethod
    def _build_quality_rubric_observations(
        priorities: list[str],
        *,
        generation_metrics: dict,
        ragas_metrics: dict,
        constraint_hit_rate: float,
        task_completed: bool,
        fallback_triggered: bool,
        evidence_boundary_observed: bool | None,
        clarification_decision_correct: bool,
        hard_constraint_violation: bool,
    ) -> dict[str, float]:
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
                observations[priority] = round(float(constraint_hit_rate), 4)
            elif priority == "grounded_recommendation":
                observations[priority] = 1.0 if task_completed and not fallback_triggered else 0.0
            elif priority == "safety":
                observations[priority] = 0.0 if hard_constraint_violation else 1.0
            elif priority == "goal_fit":
                observations[priority] = round(float(constraint_hit_rate), 4)
            elif priority == "consistency":
                observations[priority] = 1.0 if clarification_decision_correct else 0.0
        return observations

    def _compute_task_completed(
        self,
        case: QuantBenchmarkCase,
        system_success: bool,
        response: str,
        response_type: str,
        intent_correct: bool,
        clarification_decision_correct: bool,
        constraint_hit_rate: float,
        hard_constraint_violation: bool,
        generation_metrics: dict,
        ragas_metrics: dict,
    ) -> bool:
        expected_response_type = str(case.expected_response_type or "").strip()
        if case.should_clarify:
            return clarification_decision_correct and response_type == "clarification"
        if expected_response_type and response_type != expected_response_type:
            return False
        if response_type == "clarification":
            return False
        if not intent_correct:
            return False
        if hard_constraint_violation:
            return False
        if constraint_hit_rate < 1.0:
            return False
        if response_type == "fallback":
            if expected_response_type != "fallback":
                return False
            return self._response_contains_expected_keywords(
                response,
                case.ground_truth_answer_keywords,
            )
        if not system_success:
            return False
        relevancy_score = generation_metrics.get("answer_relevancy")
        if relevancy_score is not None and relevancy_score < self._settings.eval_relevancy_threshold:
            return False
        if not self._requires_grounded_faithfulness(case):
            return True
        faithfulness_score = ragas_metrics.get("faithfulness")
        if faithfulness_score is None:
            faithfulness_score = generation_metrics.get("faithfulness")
        if faithfulness_score is not None and faithfulness_score < self._settings.eval_faithfulness_threshold:
            return False
        return True

    def run_single(
        self,
        case: QuantBenchmarkCase,
        skip_generation_metrics: bool = False,
        skip_ragas_metrics: bool = False,
        skip_graph_eval: bool = False,
    ) -> QuantBenchmarkResult:
        started_at = time.perf_counter()
        try:
            response, final_state = run_diet_agent(
                query=case.query,
                user_id=f"quant_benchmark_{case.case_id}",
                return_state=True,
                skip_graph_eval=skip_graph_eval,
            )
        except Exception as exc:
            agent_latency_ms = (time.perf_counter() - started_at) * 1000
            return QuantBenchmarkResult(
                case_id=case.case_id,
                query=case.query,
                scenario_type=case.scenario_type,
                expected_intent=case.expected_intent,
                latency_ms=round(agent_latency_ms, 2),
                agent_latency_ms=round(agent_latency_ms, 2),
                evaluation_latency_ms=0.0,
                end_to_end_latency_ms=round(agent_latency_ms, 2),
                error=str(exc),
            )

        agent_latency_ms = (time.perf_counter() - started_at) * 1000
        expected_constraints = normalize_extracted_params(case.expected_constraints)
        actual_constraints = normalize_extracted_params(final_state.get("extracted_params") or {})
        active_expected_items = self._active_expected_constraint_items(expected_constraints)
        matched_constraints = sum(
            1 for key, value in active_expected_items if self._constraint_matches(actual_constraints, key, value)
        )
        constraint_hit_rate = (
            matched_constraints / len(active_expected_items)
            if active_expected_items
            else 1.0
        )

        actual_intent = str(final_state.get("intent") or "")
        response_type = str(final_state.get("response_type") or "recommendation")
        active_skill = str(final_state.get("active_skill") or "")
        planner_next_action = str(final_state.get("planner_next_action") or "")
        missing_slots = [str(item) for item in final_state.get("missing_slots") or []]
        retrieval_docs = list(final_state.get("reranked_docs") or final_state.get("retrieved_docs") or [])
        retrieved_doc_ids = [str(doc.get("id") or "") for doc in retrieval_docs if str(doc.get("id") or "")]
        token_usage = dict(final_state.get("token_usage") or {})
        retrieval_stats = dict(final_state.get("retrieval_stats") or {})
        fallback_triggered = bool(response_type == "fallback" or retrieval_stats.get("fallback_triggered", False))
        clarification_triggered = response_type == "clarification"
        intent_correct = actual_intent == case.expected_intent
        active_skill_correct = None
        if case.expected_active_skill:
            active_skill_correct = active_skill == case.expected_active_skill
        planner_action_correct = None
        if case.expected_planner_next_action:
            planner_action_correct = planner_next_action == case.expected_planner_next_action
        clarification_decision_correct = (response_type == "clarification") == case.should_clarify
        if case.expected_missing_slots and response_type == "clarification":
            clarification_decision_correct = clarification_decision_correct and bool(
                set(case.expected_missing_slots).intersection(set(missing_slots))
            )

        top_doc = self._get_top_doc(final_state)
        hard_constraint_violation = self._is_hard_constraint_violation(top_doc, expected_constraints)
        system_success = bool(str(response).strip()) and not fallback_triggered

        retrieval_metrics: dict[str, float] = {}
        if case.ground_truth_doc_ids:
            retrieval_metrics = RetrievalMetrics.compute_all(
                retrieved_ids=retrieved_doc_ids,
                relevant_ids=case.ground_truth_doc_ids,
                k=self._settings.eval_retrieval_k,
            )

        contexts = []
        for doc in retrieval_docs[:5]:
            text = doc.get("text") or doc.get("description") or doc.get("name") or str(doc)
            contexts.append(str(text)[:300])

        generation_metrics: dict[str, float] = {}
        ragas_metrics: dict[str, float] = {}
        evaluation_token_usage: dict[str, int] = {}
        evaluation_latency_ms = 0.0
        should_run_quality_eval = (
            not case.should_clarify
            and response_type != "clarification"
            and str(response).strip()
        )
        has_enabled_quality_eval = (
            (not skip_generation_metrics and self._gen_metrics is not None)
            or (not skip_ragas_metrics and bool(case.ground_truth_answer) and self._ragas is not None)
        )
        if should_run_quality_eval and has_enabled_quality_eval:
            evaluation_started_at = time.perf_counter()
            with token_usage_scope():
                if not skip_generation_metrics and self._gen_metrics is not None:
                    try:
                        generation_metrics = self._gen_metrics.evaluate_single(
                            query=case.query,
                            answer=response,
                            contexts=contexts,
                            ground_truth_keywords=case.ground_truth_answer_keywords,
                        )
                    except Exception as exc:
                        logger.warning(f"GenerationMetrics 评测失败，case_id={case.case_id}: {exc}")
                if not skip_ragas_metrics and case.ground_truth_answer and self._ragas is not None:
                    try:
                        ragas_metrics = self._ragas.evaluate_single(
                            query=case.query,
                            answer=response,
                            contexts=contexts,
                            ground_truth=case.ground_truth_answer,
                        )
                    except Exception as exc:
                        logger.warning(f"RAGAS 评测失败，case_id={case.case_id}: {exc}")
                evaluation_token_usage = get_current_token_usage()
            evaluation_latency_ms = (time.perf_counter() - evaluation_started_at) * 1000

        overall_token_usage = merge_token_usage(token_usage, evaluation_token_usage)
        end_to_end_latency_ms = (time.perf_counter() - started_at) * 1000

        tool_call_accuracy = self._compute_tool_call_accuracy(
            intent_correct=intent_correct,
            active_skill_correct=active_skill_correct,
            planner_action_correct=planner_action_correct,
            constraint_hit_rate=constraint_hit_rate,
            clarification_decision_correct=clarification_decision_correct,
        )

        hallucination_detected: bool | None = None
        if self._requires_grounded_faithfulness(case):
            faithfulness_score = ragas_metrics.get("faithfulness")
            if faithfulness_score is None:
                faithfulness_score = generation_metrics.get("faithfulness")
            if faithfulness_score is not None:
                hallucination_detected = faithfulness_score < self._settings.eval_faithfulness_threshold

        task_completed = self._compute_task_completed(
            case=case,
            system_success=system_success,
            response=response,
            response_type=response_type,
            intent_correct=intent_correct,
            clarification_decision_correct=clarification_decision_correct,
            constraint_hit_rate=constraint_hit_rate,
            hard_constraint_violation=hard_constraint_violation,
            generation_metrics=generation_metrics,
            ragas_metrics=ragas_metrics,
        )
        quality_signals = build_skill_quality_signals(
            active_skill,
            response=response,
            response_type=response_type,
            retrieval_stats=retrieval_stats,
            evaluation={},
            generation_metrics=generation_metrics,
            ragas_metrics=ragas_metrics,
            constraint_hit_rate=constraint_hit_rate,
            task_completed=task_completed,
            clarification_decision_correct=clarification_decision_correct,
            hard_constraint_violation=hard_constraint_violation,
        )
        requires_evidence_boundary = bool(quality_signals.get("requires_evidence_boundary", False))
        evidence_boundary_observed = quality_signals.get("evidence_boundary_observed")
        quality_rubric_priorities = list(quality_signals.get("quality_rubric_priorities", []))
        quality_rubric_observations = dict(quality_signals.get("quality_rubric_observations", {}))
        quality_rubric_score = quality_signals.get("quality_rubric_score")

        return QuantBenchmarkResult(
            case_id=case.case_id,
            query=case.query,
            scenario_type=case.scenario_type,
            expected_intent=case.expected_intent,
            actual_intent=actual_intent,
            response=response,
            response_type=response_type,
            active_skill=active_skill,
            planner_next_action=planner_next_action,
            extracted_params=actual_constraints,
            missing_slots=missing_slots,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            ragas_metrics=ragas_metrics,
            token_usage=token_usage,
            evaluation_token_usage=evaluation_token_usage,
            overall_token_usage=overall_token_usage,
            latency_ms=round(agent_latency_ms, 2),
            agent_latency_ms=round(agent_latency_ms, 2),
            evaluation_latency_ms=round(evaluation_latency_ms, 2),
            end_to_end_latency_ms=round(end_to_end_latency_ms, 2),
            fallback_triggered=fallback_triggered,
            clarification_triggered=clarification_triggered,
            hard_constraint_violation=hard_constraint_violation,
            system_success=system_success,
            task_completed=task_completed,
            intent_correct=intent_correct,
            active_skill_correct=active_skill_correct,
            planner_action_correct=planner_action_correct,
            clarification_decision_correct=clarification_decision_correct,
            requires_evidence_boundary=requires_evidence_boundary,
            evidence_boundary_observed=evidence_boundary_observed,
            quality_rubric_priorities=quality_rubric_priorities,
            quality_rubric_observations=quality_rubric_observations,
            quality_rubric_score=quality_rubric_score,
            constraint_hit_rate=round(constraint_hit_rate, 4),
            tool_call_accuracy=round(tool_call_accuracy, 4),
            hallucination_detected=hallucination_detected,
        )

    def run_all(
        self,
        show_progress: bool = False,
        case_ids: list[str] | None = None,
        skip_generation_metrics: bool = False,
        skip_ragas_metrics: bool = False,
        skip_graph_eval: bool = False,
    ) -> dict:
        selected_cases = self._resolve_cases(case_ids)
        details: list[QuantBenchmarkResult] = []
        total_cases = len(selected_cases)
        for index, case in enumerate(selected_cases, 1):
            if show_progress:
                print(
                    f"[{index}/{total_cases}] START case_id={case.case_id} query={case.query}",
                    flush=True,
                )
            started_at = time.perf_counter()
            detail = self.run_single(
                case,
                skip_generation_metrics=skip_generation_metrics,
                skip_ragas_metrics=skip_ragas_metrics,
                skip_graph_eval=skip_graph_eval,
            )
            details.append(detail)
            if show_progress:
                elapsed_seconds = time.perf_counter() - started_at
                status = "error" if detail.error else "ok"

                print(
                    f"[{index}/{total_cases}] END case_id={case.case_id} status={status} "
                    f"response_type={detail.response_type or 'unknown'} latency_ms={detail.latency_ms:.2f} "
                    f"elapsed_s={elapsed_seconds:.2f}",
                    flush=True,
                )

        result_dicts = [detail.model_dump() for detail in details]
        agent_latencies = [detail.agent_latency_ms or detail.latency_ms for detail in details]
        evaluation_latencies = [detail.evaluation_latency_ms for detail in details]
        end_to_end_latencies = [
            detail.end_to_end_latency_ms
            or (detail.agent_latency_ms or detail.latency_ms) + detail.evaluation_latency_ms
            for detail in details
        ]
        clarify_case_ids = {case.case_id for case in selected_cases if case.should_clarify}
        agent_token_summary = self._build_token_summary([detail.token_usage for detail in details])
        evaluation_token_summary = self._build_token_summary([detail.evaluation_token_usage for detail in details])
        overall_token_summary = self._build_token_summary([detail.overall_token_usage for detail in details])
        retrieval_cases = [detail for detail in details if detail.retrieval_metrics]
        generation_cases = [detail for detail in details if detail.generation_metrics]
        ragas_cases = [detail for detail in details if detail.ragas_metrics]
        grounded_generation_cases = [
            detail for detail in generation_cases if detail.expected_intent != "chitchat"
        ]
        grounded_ragas_cases = [
            detail for detail in ragas_cases if detail.expected_intent != "chitchat"
        ]
        hallucination_cases = [
            detail for detail in details if detail.hallucination_detected is not None
        ]

        avg_retrieval = {
            "recall_at_k": self._mean([detail.retrieval_metrics.get("recall_at_k", 0.0) for detail in retrieval_cases]),
            "mrr": self._mean([detail.retrieval_metrics.get("mrr", 0.0) for detail in retrieval_cases]),
            "ndcg": self._mean([detail.retrieval_metrics.get("ndcg", 0.0) for detail in retrieval_cases]),
        }
        avg_generation = {
            "faithfulness": self._mean([detail.generation_metrics.get("faithfulness", 0.0) for detail in grounded_generation_cases]),
            "answer_relevancy": self._mean([detail.generation_metrics.get("answer_relevancy", 0.0) for detail in generation_cases]),
            "completeness": self._mean([detail.generation_metrics.get("completeness", 0.0) for detail in generation_cases]),
        }
        avg_ragas = {
            "context_precision": self._mean([detail.ragas_metrics.get("context_precision", 0.0) for detail in grounded_ragas_cases]),
            "context_recall": self._mean([detail.ragas_metrics.get("context_recall", 0.0) for detail in grounded_ragas_cases]),
            "faithfulness": self._mean([detail.ragas_metrics.get("faithfulness", 0.0) for detail in grounded_ragas_cases]),
            "answer_relevancy": self._mean([detail.ragas_metrics.get("answer_relevancy", 0.0) for detail in grounded_ragas_cases]),
        }

        by_intent: dict[str, dict[str, float | int]] = {}
        for intent in ["recipe_search", "nutrition_query", "ingredient_check", "chitchat"]:
            intent_details = [detail for detail in details if detail.expected_intent == intent]
            if not intent_details:
                continue
            intent_overall_token_summary = self._build_token_summary(
                [detail.overall_token_usage for detail in intent_details]
            )
            by_intent[intent] = {
                "count": len(intent_details),
                "task_completion_rate": round(self._rate([detail.task_completed for detail in intent_details]), 4),
                "intent_accuracy": round(self._rate([detail.intent_correct for detail in intent_details]), 4),
                "latency_ms_mean": round(
                    self._mean([detail.agent_latency_ms or detail.latency_ms for detail in intent_details]),
                    2,
                ),
                "end_to_end_latency_ms_mean": round(
                    self._mean([detail.end_to_end_latency_ms for detail in intent_details]),
                    2,
                ),
                "overall_total_tokens_mean": intent_overall_token_summary.get("total_tokens_mean", 0.0),
            }

        by_skill: dict[str, dict[str, Any]] = {}
        active_skill_names = sorted({detail.active_skill for detail in details if detail.active_skill})
        for skill_name in active_skill_names:
            skill_details = [detail for detail in details if detail.active_skill == skill_name]
            if not skill_details:
                continue
            skill_overall_token_summary = self._build_token_summary(
                [detail.overall_token_usage for detail in skill_details]
            )
            evidence_boundary_cases = [
                detail for detail in skill_details if detail.evidence_boundary_observed is not None
            ]
            rubric_signal_names = sorted(
                {
                    signal_name
                    for detail in skill_details
                    for signal_name in detail.quality_rubric_observations.keys()
                }
            )
            rubric_signal_means = {
                signal_name: round(
                    self._mean(
                        [
                            float(detail.quality_rubric_observations[signal_name])
                            for detail in skill_details
                            if signal_name in detail.quality_rubric_observations
                        ]
                    ),
                    4,
                )
                for signal_name in rubric_signal_names
            }
            rubric_priorities = sorted(
                {
                    priority
                    for detail in skill_details
                    for priority in detail.quality_rubric_priorities
                }
            )
            by_skill[skill_name] = {
                "count": len(skill_details),
                "task_completion_rate": round(self._rate([detail.task_completed for detail in skill_details]), 4),
                "fallback_trigger_rate": round(self._rate([detail.fallback_triggered for detail in skill_details]), 4),
                "clarification_rate": round(self._rate([detail.clarification_triggered for detail in skill_details]), 4),
                "active_skill_accuracy": round(
                    self._rate([
                        detail.active_skill_correct
                        for detail in skill_details
                        if detail.active_skill_correct is not None
                    ]),
                    4,
                ),
                "evidence_boundary_rate": round(
                    self._rate([detail.evidence_boundary_observed for detail in evidence_boundary_cases]),
                    4,
                ),
                "quality_rubric_score_mean": round(
                    self._mean([
                        detail.quality_rubric_score
                        for detail in skill_details
                        if detail.quality_rubric_score is not None
                    ]),
                    4,
                ),
                "tool_call_accuracy_mean": round(
                    self._mean([detail.tool_call_accuracy for detail in skill_details]),
                    4,
                ),
                "overall_total_tokens_mean": skill_overall_token_summary.get("total_tokens_mean", 0.0),
                "rubric_priorities": rubric_priorities,
                "rubric_signal_means": rubric_signal_means,
            }

        summary = {
            "total_cases": len(details),
            "system_success_rate": round(self._rate([detail.system_success for detail in details]), 4),
            "task_completion_rate": round(self._rate([detail.task_completed for detail in details]), 4),
            "fallback_trigger_rate": round(self._rate([detail.fallback_triggered for detail in details]), 4),
            "clarification_rate": round(self._rate([detail.clarification_triggered for detail in details]), 4),
            "evidence_boundary_rate": round(
                self._rate([
                    detail.evidence_boundary_observed
                    for detail in details
                    if detail.evidence_boundary_observed is not None
                ]),
                4,
            ),
            "quality_rubric_score_mean": round(
                self._mean([
                    detail.quality_rubric_score
                    for detail in details
                    if detail.quality_rubric_score is not None
                ]),
                4,
            ),
            "clarification_success_rate": round(
                self._rate([detail.task_completed for detail in details if detail.case_id in clarify_case_ids]),
                4,
            ),
            "intent_accuracy": round(self._rate([detail.intent_correct for detail in details]), 4),
            "active_skill_accuracy": round(
                self._rate([detail.active_skill_correct for detail in details if detail.active_skill_correct is not None]),
                4,
            ),
            "planner_action_accuracy": round(
                self._rate([detail.planner_action_correct for detail in details if detail.planner_action_correct is not None]),
                4,
            ),
            "clarification_decision_accuracy": round(
                self._rate([detail.clarification_decision_correct for detail in details]),
                4,
            ),
            "constraint_hit_rate": round(self._mean([detail.constraint_hit_rate for detail in details]), 4),
            "tool_call_accuracy": round(self._mean([detail.tool_call_accuracy for detail in details]), 4),
            "hard_constraint_violation_rate": round(
                self._rate([detail.hard_constraint_violation for detail in details]),
                4,
            ),
            "hallucination_rate": round(
                self._rate([detail.hallucination_detected for detail in hallucination_cases]),
                4,
            ),
            "rag_faithfulness_avg": round(avg_ragas.get("faithfulness", 0.0), 4),
            "rag_faithfulness_pass_rate": round(
                self._rate([
                    detail.ragas_metrics.get("faithfulness", 0.0) >= self._settings.eval_faithfulness_threshold
                    for detail in grounded_ragas_cases
                ]),
                4,
            ),
            "avg_retrieval": avg_retrieval,
            "avg_generation": avg_generation,
            "avg_ragas": avg_ragas,
            "latency_ms_mean": round(self._mean(agent_latencies), 2),
            "latency_ms_p50": round(self._percentile(agent_latencies, 50), 2),
            "latency_ms_p90": round(self._percentile(agent_latencies, 90), 2),
            "agent_latency_ms_mean": round(self._mean(agent_latencies), 2),
            "agent_latency_ms_p50": round(self._percentile(agent_latencies, 50), 2),
            "agent_latency_ms_p90": round(self._percentile(agent_latencies, 90), 2),
            "evaluation_latency_ms_mean": round(self._mean(evaluation_latencies), 2),
            "evaluation_latency_ms_p50": round(self._percentile(evaluation_latencies, 50), 2),
            "evaluation_latency_ms_p90": round(self._percentile(evaluation_latencies, 90), 2),
            "end_to_end_latency_ms_mean": round(self._mean(end_to_end_latencies), 2),
            "end_to_end_latency_ms_p50": round(self._percentile(end_to_end_latencies, 50), 2),
            "end_to_end_latency_ms_p90": round(self._percentile(end_to_end_latencies, 90), 2),
            "prompt_tokens_mean": agent_token_summary.get("prompt_tokens_mean", 0.0),
            "completion_tokens_mean": agent_token_summary.get("completion_tokens_mean", 0.0),
            "total_tokens_mean": agent_token_summary.get("total_tokens_mean", 0.0),
            "llm_calls_mean": agent_token_summary.get("llm_calls_mean", 0.0),
            "prompt_tokens_sum": agent_token_summary.get("prompt_tokens_sum", 0),
            "completion_tokens_sum": agent_token_summary.get("completion_tokens_sum", 0),
            "total_tokens_sum": agent_token_summary.get("total_tokens_sum", 0),
            "llm_calls_sum": agent_token_summary.get("llm_calls_sum", 0),
            "agent_token_summary": agent_token_summary,
            "evaluation_token_summary": evaluation_token_summary,
            "overall_token_summary": overall_token_summary,
        }

        return {
            "dataset_path": self._dataset_path,
            "selected_case_ids": [case.case_id for case in selected_cases],
            "skip_generation_metrics": skip_generation_metrics,
            "skip_ragas_metrics": skip_ragas_metrics,
            "skip_graph_eval": skip_graph_eval,
            "total_cases": len(details),
            "summary": summary,
            "by_intent": by_intent,
            "by_skill": by_skill,
            "details": result_dicts,
        }

    def generate_report(self, results: dict, version_label: str = "quant-benchmark") -> str:
        if not results:
            return "# Quant Benchmark Report\n\n无评测数据。"
        summary = results.get("summary", {})
        avg_retrieval = summary.get("avg_retrieval", {})
        avg_generation = summary.get("avg_generation", {})
        avg_ragas = summary.get("avg_ragas", {})
        agent_token_summary = summary.get("agent_token_summary", {})
        evaluation_token_summary = summary.get("evaluation_token_summary", {})
        overall_token_summary = summary.get("overall_token_summary", {})
        lines = [
            f"# {version_label} Quant Benchmark Report",
            "",
            "## Overview",
            f"- total_cases: {results.get('total_cases', 0)}",
            f"- selected_case_ids: {', '.join(results.get('selected_case_ids', [])) or 'all'}",
            f"- skip_generation_metrics: {bool(results.get('skip_generation_metrics', False))}",
            f"- skip_ragas_metrics: {bool(results.get('skip_ragas_metrics', False))}",
            f"- skip_graph_eval: {bool(results.get('skip_graph_eval', False))}",
            f"- system_success_rate: {summary.get('system_success_rate', 0.0):.4f}",
            f"- task_completion_rate: {summary.get('task_completion_rate', 0.0):.4f}",
            f"- fallback_trigger_rate: {summary.get('fallback_trigger_rate', 0.0):.4f}",
            f"- clarification_rate: {summary.get('clarification_rate', 0.0):.4f}",
            f"- evidence_boundary_rate: {summary.get('evidence_boundary_rate', 0.0):.4f}",
            f"- tool_call_accuracy: {summary.get('tool_call_accuracy', 0.0):.4f}",
            f"- hallucination_rate: {summary.get('hallucination_rate', 0.0):.4f}",
            f"- agent_latency_ms_mean: {summary.get('agent_latency_ms_mean', summary.get('latency_ms_mean', 0.0)):.2f}",
            f"- end_to_end_latency_ms_mean: {summary.get('end_to_end_latency_ms_mean', 0.0):.2f}",
            "",
            "## Latency Breakdown",
            "",
            "| metric | value |",
            "|------|------|",
            f"| agent_latency_ms_mean | {summary.get('agent_latency_ms_mean', summary.get('latency_ms_mean', 0.0)):.2f} |",
            f"| agent_latency_ms_p50 | {summary.get('agent_latency_ms_p50', summary.get('latency_ms_p50', 0.0)):.2f} |",
            f"| agent_latency_ms_p90 | {summary.get('agent_latency_ms_p90', summary.get('latency_ms_p90', 0.0)):.2f} |",
            f"| evaluation_latency_ms_mean | {summary.get('evaluation_latency_ms_mean', 0.0):.2f} |",
            f"| evaluation_latency_ms_p50 | {summary.get('evaluation_latency_ms_p50', 0.0):.2f} |",
            f"| evaluation_latency_ms_p90 | {summary.get('evaluation_latency_ms_p90', 0.0):.2f} |",
            f"| end_to_end_latency_ms_mean | {summary.get('end_to_end_latency_ms_mean', 0.0):.2f} |",
            f"| end_to_end_latency_ms_p50 | {summary.get('end_to_end_latency_ms_p50', 0.0):.2f} |",
            f"| end_to_end_latency_ms_p90 | {summary.get('end_to_end_latency_ms_p90', 0.0):.2f} |",
            "",
            "## Core Metrics",
            "",
            "| metric | value |",
            "|------|------|",
            f"| intent_accuracy | {summary.get('intent_accuracy', 0.0):.4f} |",
            f"| active_skill_accuracy | {summary.get('active_skill_accuracy', 0.0):.4f} |",
            f"| planner_action_accuracy | {summary.get('planner_action_accuracy', 0.0):.4f} |",
            f"| clarification_decision_accuracy | {summary.get('clarification_decision_accuracy', 0.0):.4f} |",
            f"| constraint_hit_rate | {summary.get('constraint_hit_rate', 0.0):.4f} |",
            f"| fallback_trigger_rate | {summary.get('fallback_trigger_rate', 0.0):.4f} |",
            f"| clarification_rate | {summary.get('clarification_rate', 0.0):.4f} |",
            f"| evidence_boundary_rate | {summary.get('evidence_boundary_rate', 0.0):.4f} |",
            f"| quality_rubric_score_mean | {summary.get('quality_rubric_score_mean', 0.0):.4f} |",
            f"| rag_faithfulness_avg | {summary.get('rag_faithfulness_avg', 0.0):.4f} |",
            f"| rag_faithfulness_pass_rate | {summary.get('rag_faithfulness_pass_rate', 0.0):.4f} |",
            f"| hard_constraint_violation_rate | {summary.get('hard_constraint_violation_rate', 0.0):.4f} |",
            f"| prompt_tokens_mean | {summary.get('prompt_tokens_mean', 0.0):.2f} |",
            f"| completion_tokens_mean | {summary.get('completion_tokens_mean', 0.0):.2f} |",
            f"| total_tokens_mean | {summary.get('total_tokens_mean', 0.0):.2f} |",
            f"| llm_calls_mean | {summary.get('llm_calls_mean', 0.0):.2f} |",
            "",
            "## Token Usage Breakdown",
            "",
            "| scope | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | llm_calls_mean | total_tokens_sum |",
            "|------|------:|------:|------:|------:|------:|",
            f"| agent | {agent_token_summary.get('prompt_tokens_mean', 0.0):.2f} | {agent_token_summary.get('completion_tokens_mean', 0.0):.2f} | {agent_token_summary.get('total_tokens_mean', 0.0):.2f} | {agent_token_summary.get('llm_calls_mean', 0.0):.2f} | {agent_token_summary.get('total_tokens_sum', 0)} |",
            f"| evaluation | {evaluation_token_summary.get('prompt_tokens_mean', 0.0):.2f} | {evaluation_token_summary.get('completion_tokens_mean', 0.0):.2f} | {evaluation_token_summary.get('total_tokens_mean', 0.0):.2f} | {evaluation_token_summary.get('llm_calls_mean', 0.0):.2f} | {evaluation_token_summary.get('total_tokens_sum', 0)} |",
            f"| overall | {overall_token_summary.get('prompt_tokens_mean', 0.0):.2f} | {overall_token_summary.get('completion_tokens_mean', 0.0):.2f} | {overall_token_summary.get('total_tokens_mean', 0.0):.2f} | {overall_token_summary.get('llm_calls_mean', 0.0):.2f} | {overall_token_summary.get('total_tokens_sum', 0)} |",
            "",
            "## Retrieval Metrics",
            "",
            "| metric | value |",
            "|------|------|",
            f"| Recall@K | {avg_retrieval.get('recall_at_k', 0.0):.4f} |",
            f"| MRR | {avg_retrieval.get('mrr', 0.0):.4f} |",
            f"| nDCG | {avg_retrieval.get('ndcg', 0.0):.4f} |",
            "",
            "## Generation Metrics",
            "",
            "| metric | value |",
            "|------|------|",
            f"| Faithfulness | {avg_generation.get('faithfulness', 0.0):.4f} |",
            f"| Answer Relevancy | {avg_generation.get('answer_relevancy', 0.0):.4f} |",
            f"| Completeness | {avg_generation.get('completeness', 0.0):.4f} |",
            "",
            "## RAGAS Metrics",
            "",
            "| metric | value |",
            "|------|------|",
            f"| Context Precision | {avg_ragas.get('context_precision', 0.0):.4f} |",
            f"| Context Recall | {avg_ragas.get('context_recall', 0.0):.4f} |",
            f"| Faithfulness | {avg_ragas.get('faithfulness', 0.0):.4f} |",
            f"| Answer Relevancy | {avg_ragas.get('answer_relevancy', 0.0):.4f} |",
            "",
            "## By Intent",
            "",
            "| intent | count | task_completion_rate | intent_accuracy | agent_latency_ms_mean | end_to_end_latency_ms_mean | overall_total_tokens_mean |",
            "|------|------:|------:|------:|------:|------:|------:|",
        ]
        for intent, intent_summary in results.get("by_intent", {}).items():
            lines.append(
                f"| {intent} | {intent_summary.get('count', 0)} | {intent_summary.get('task_completion_rate', 0.0):.4f} | {intent_summary.get('intent_accuracy', 0.0):.4f} | {intent_summary.get('latency_ms_mean', 0.0):.2f} | {intent_summary.get('end_to_end_latency_ms_mean', 0.0):.2f} | {intent_summary.get('overall_total_tokens_mean', 0.0):.2f} |"
            )

        lines.extend([
            "",
            "## By Skill",
            "",
            "| skill | count | task_completion_rate | fallback_trigger_rate | clarification_rate | evidence_boundary_rate | quality_rubric_score_mean | overall_total_tokens_mean | rubric_priorities |",
            "|------|------:|------:|------:|------:|------:|------:|------:|------|",
        ])
        for skill_name, skill_summary in results.get("by_skill", {}).items():
            rubric_priorities = ", ".join(skill_summary.get("rubric_priorities", [])) or "-"
            lines.append(
                f"| {skill_name} | {skill_summary.get('count', 0)} | {skill_summary.get('task_completion_rate', 0.0):.4f} | {skill_summary.get('fallback_trigger_rate', 0.0):.4f} | {skill_summary.get('clarification_rate', 0.0):.4f} | {skill_summary.get('evidence_boundary_rate', 0.0):.4f} | {skill_summary.get('quality_rubric_score_mean', 0.0):.4f} | {skill_summary.get('overall_total_tokens_mean', 0.0):.2f} | {rubric_priorities} |"
            )
        return "\n".join(lines)


__all__ = [
    "QuantBenchmarkCase",
    "QuantBenchmarkResult",
    "QuantitativeBenchmark",
]
