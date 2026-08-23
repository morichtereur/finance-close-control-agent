"""Evaluation metrics.

What is measured, and why each one is here:

``risk_accuracy``
    Agreement with the expected risk rating. The headline number, and the least
    interesting one on its own.

``escalation_precision`` / ``escalation_recall``
    The metrics a Financial Controller actually cares about. Recall is the cost
    of a missed escalation — an item that should have reached a person and did
    not. Precision is the cost of noise — reviewer time spent on items that did
    not need it. They trade off, and a system that escalates everything scores
    perfect recall while being worthless.

``retrieval_recall``
    Did the *retriever* surface the governing policy document at all? Reported
    separately from citation accuracy so a citation failure can be attributed to
    the right layer.

``citation_accuracy``
    Did the decision cite the governing document, from grounded citations only?

``structured_output_success``
    Share of cases that produced a schema-valid decision at all.

``ungrounded_citation_rate`` / ``unsupported_recommendation_rate``
    The two honesty metrics. The first counts decisions that cited policy which
    was never retrieved. The second counts decisions that recommend an action
    while resting on no grounded policy evidence at all.

Failures are never silently dropped: a case that produced no valid decision
counts as a wrong risk rating and as an escalation (the gate forces one).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from pydantic import BaseModel, Field

from fcca.close.models import CaseFailure, CaseResult, LabelledCase
from fcca.close.workflow.grounding import _normalise_document


@dataclass
class CaseEvaluation:
    """Per-case comparison against the label."""

    exception_id: str
    scenario: str
    decided: bool
    expected_risk: str
    actual_risk: str | None
    expected_review: bool
    actual_review: bool
    expected_action: str
    actual_action: str | None
    expected_document: str
    retrieved_expected_document: bool
    cited_expected_document: bool
    ungrounded_citations: int
    grounded_citations: int
    latency_ms: int | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None

    @property
    def risk_correct(self) -> bool:
        return self.decided and self.actual_risk == self.expected_risk

    @property
    def action_correct(self) -> bool:
        return self.decided and self.actual_action == self.expected_action


class BenchmarkMetrics(BaseModel):
    """Aggregate metrics for one provider run."""

    n_cases: int
    n_decided: int
    n_failed: int

    risk_accuracy: float
    action_category_accuracy: float
    escalation_precision: float
    escalation_recall: float
    escalation_f1: float
    retrieval_recall: float
    citation_accuracy: float
    structured_output_success: float
    ungrounded_citation_rate: float
    unsupported_recommendation_rate: float
    auto_recommendation_rate: float

    median_latency_ms: float
    p95_latency_ms: float

    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None
    cost_per_case_usd: float | None = None

    confusion: dict[str, int] = Field(
        default_factory=dict,
        description="Escalation confusion matrix: tp, fp, tn, fn.",
    )
    per_scenario_risk_accuracy: dict[str, float] = Field(default_factory=dict)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate_case(result: CaseResult | CaseFailure, label: LabelledCase) -> CaseEvaluation:
    """Compare one case outcome against its label."""
    expected_document = _normalise_document(label.expected_policy_document)

    if isinstance(result, CaseFailure):
        return CaseEvaluation(
            exception_id=label.exception_id,
            scenario=label.scenario,
            decided=False,
            expected_risk=label.expected_risk_level,
            actual_risk=None,
            expected_review=label.expected_requires_human_review,
            actual_review=True,
            expected_action=label.expected_action_category,
            actual_action=None,
            expected_document=label.expected_policy_document,
            retrieved_expected_document=False,
            cited_expected_document=False,
            ungrounded_citations=0,
            grounded_citations=0,
            latency_ms=None,
            error=result.error,
        )

    retrieved_documents = {_normalise_document(e.document) for e in result.evidence}
    cited_documents = {_normalise_document(c.document) for c in result.decision.policy_citations}

    return CaseEvaluation(
        exception_id=label.exception_id,
        scenario=label.scenario,
        decided=True,
        expected_risk=label.expected_risk_level,
        actual_risk=result.decision.risk_level,
        expected_review=label.expected_requires_human_review,
        actual_review=result.final_requires_human_review,
        expected_action=label.expected_action_category,
        actual_action=result.decision.action_category,
        expected_document=label.expected_policy_document,
        retrieved_expected_document=expected_document in retrieved_documents,
        cited_expected_document=expected_document in cited_documents,
        ungrounded_citations=len(result.grounding.ungrounded_citations),
        grounded_citations=result.grounding.grounded_citations,
        latency_ms=result.run.latency_ms,
        input_tokens=result.run.input_tokens,
        output_tokens=result.run.output_tokens,
        cost_usd=result.run.estimated_cost_usd,
    )


def evaluate_cases(
    results: list[CaseResult | CaseFailure],
    labels: dict[str, LabelledCase],
) -> tuple[BenchmarkMetrics, list[CaseEvaluation]]:
    """Aggregate per-case evaluations into benchmark metrics."""
    evaluations = [
        evaluate_case(result, labels[_exception_id(result)])
        for result in results
        if _exception_id(result) in labels
    ]
    return aggregate(evaluations), evaluations


def _exception_id(result: CaseResult | CaseFailure) -> str:
    if isinstance(result, CaseFailure):
        return result.exception_id
    return result.exception.exception_id


def aggregate(evaluations: list[CaseEvaluation]) -> BenchmarkMetrics:
    """Compute aggregate metrics from per-case evaluations."""
    n = len(evaluations)
    decided = [e for e in evaluations if e.decided]

    tp = sum(1 for e in evaluations if e.actual_review and e.expected_review)
    fp = sum(1 for e in evaluations if e.actual_review and not e.expected_review)
    fn = sum(1 for e in evaluations if not e.actual_review and e.expected_review)
    tn = sum(1 for e in evaluations if not e.actual_review and not e.expected_review)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    latencies = [float(e.latency_ms) for e in decided if e.latency_ms is not None]
    input_tokens = [e.input_tokens for e in decided if e.input_tokens is not None]
    output_tokens = [e.output_tokens for e in decided if e.output_tokens is not None]
    costs = [e.cost_usd for e in decided if e.cost_usd is not None]

    unsupported = sum(
        1
        for e in decided
        if e.grounded_citations == 0 and e.actual_action not in (None, "no_action")
    )

    per_scenario: dict[str, list[bool]] = {}
    for evaluation in evaluations:
        per_scenario.setdefault(evaluation.scenario, []).append(evaluation.risk_correct)

    return BenchmarkMetrics(
        n_cases=n,
        n_decided=len(decided),
        n_failed=n - len(decided),
        risk_accuracy=_safe_ratio(sum(1 for e in evaluations if e.risk_correct), n),
        action_category_accuracy=_safe_ratio(sum(1 for e in evaluations if e.action_correct), n),
        escalation_precision=precision,
        escalation_recall=recall,
        escalation_f1=f1,
        retrieval_recall=_safe_ratio(
            sum(1 for e in evaluations if e.retrieved_expected_document), n
        ),
        citation_accuracy=_safe_ratio(sum(1 for e in evaluations if e.cited_expected_document), n),
        structured_output_success=_safe_ratio(len(decided), n),
        ungrounded_citation_rate=_safe_ratio(
            sum(1 for e in decided if e.ungrounded_citations > 0), len(decided)
        ),
        unsupported_recommendation_rate=_safe_ratio(unsupported, len(decided)),
        auto_recommendation_rate=_safe_ratio(sum(1 for e in evaluations if not e.actual_review), n),
        median_latency_ms=round(statistics.median(latencies), 1) if latencies else 0.0,
        p95_latency_ms=round(_percentile(latencies, 95), 1) if latencies else 0.0,
        total_input_tokens=sum(input_tokens) if input_tokens else None,
        total_output_tokens=sum(output_tokens) if output_tokens else None,
        total_cost_usd=round(sum(costs), 6) if costs else None,
        cost_per_case_usd=round(sum(costs) / len(decided), 6) if costs and decided else None,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        per_scenario_risk_accuracy={
            scenario: _safe_ratio(sum(flags), len(flags))
            for scenario, flags in per_scenario.items()
        },
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[index]
