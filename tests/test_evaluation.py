"""Evaluation metrics and the benchmark's not_run discipline.

The metrics are tested against deliberately wrong answers. A scoring function
that only ever sees correct input is not evidence of anything.
"""

from __future__ import annotations

from fcca.config import Settings
from fcca.evaluation.benchmark import CSV_COLUMNS, NOT_RUN, BenchmarkRun, run_benchmark
from fcca.evaluation.metrics import CaseEvaluation, aggregate


def _case(**overrides: object) -> CaseEvaluation:
    payload: dict[str, object] = {
        "exception_id": "EXC-0001",
        "scenario": "unsupported_manual_material",
        "decided": True,
        "expected_risk": "high",
        "actual_risk": "high",
        "expected_review": True,
        "actual_review": True,
        "expected_action": "escalate_to_financial_controller",
        "actual_action": "escalate_to_financial_controller",
        "expected_document": "Supporting Documentation Standard",
        "retrieved_expected_document": True,
        "cited_expected_document": True,
        "ungrounded_citations": 0,
        "grounded_citations": 2,
        "latency_ms": 100,
    }
    payload.update(overrides)
    return CaseEvaluation(**payload)  # type: ignore[arg-type]


def test_perfect_run_scores_one() -> None:
    metrics = aggregate([_case(), _case(expected_review=False, actual_review=False)])
    assert metrics.risk_accuracy == 1.0
    assert metrics.escalation_precision == 1.0
    assert metrics.escalation_recall == 1.0


def test_a_missed_escalation_lowers_recall() -> None:
    metrics = aggregate([_case(), _case(actual_review=False)])
    assert metrics.escalation_recall == 0.5
    assert metrics.escalation_precision == 1.0
    assert metrics.confusion == {"tp": 1, "fp": 0, "tn": 0, "fn": 1}


def test_over_escalation_lowers_precision() -> None:
    metrics = aggregate([_case(), _case(expected_review=False, actual_review=True)])
    assert metrics.escalation_precision == 0.5
    assert metrics.escalation_recall == 1.0


def test_wrong_risk_rating_lowers_accuracy() -> None:
    metrics = aggregate([_case(), _case(actual_risk="low")])
    assert metrics.risk_accuracy == 0.5


def test_a_failed_case_counts_as_wrong_and_as_an_escalation() -> None:
    metrics = aggregate([_case(decided=False, actual_risk=None, actual_review=True)])
    assert metrics.structured_output_success == 0.0
    assert metrics.risk_accuracy == 0.0
    assert metrics.n_failed == 1
    assert metrics.escalation_recall == 1.0


def test_recommendation_without_grounded_evidence_is_counted() -> None:
    metrics = aggregate([_case(grounded_citations=0, cited_expected_document=False)])
    assert metrics.unsupported_recommendation_rate == 1.0


def test_a_no_action_recommendation_without_citations_is_not_unsupported() -> None:
    metrics = aggregate(
        [_case(grounded_citations=0, actual_action="no_action", cited_expected_document=False)]
    )
    assert metrics.unsupported_recommendation_rate == 0.0


def test_ungrounded_citations_are_counted() -> None:
    metrics = aggregate([_case(), _case(ungrounded_citations=1)])
    assert metrics.ungrounded_citation_rate == 0.5


def test_retrieval_failure_is_attributed_separately_from_citation_failure() -> None:
    metrics = aggregate([_case(retrieved_expected_document=False, cited_expected_document=False)])
    assert metrics.retrieval_recall == 0.0
    assert metrics.citation_accuracy == 0.0


# ---------------------------------------------------------------------- runner
def test_mock_benchmark_runs_and_is_labelled_as_a_stub(settings: Settings) -> None:
    run = run_benchmark("mock", limit=6, settings=settings, with_audit=False)
    assert run.status == "ok"
    assert run.metrics is not None
    assert run.metrics.n_cases == 6
    assert "stub" in run.note


def test_a_provider_that_did_not_run_writes_no_numbers() -> None:
    run = BenchmarkRun(provider="bedrock", model="some-model", status="not_run", note="no creds")
    row = run.as_row()
    assert set(row) == set(CSV_COLUMNS)
    assert row["risk_accuracy"] == NOT_RUN
    assert row["median_latency_ms"] == NOT_RUN
    assert row["cost_per_case_usd"] == NOT_RUN


def test_uncredentialed_cloud_provider_is_reported_as_not_run(settings: Settings) -> None:
    """Never an exception, never a fabricated number."""
    without_project = settings.model_copy(update={"google_cloud_project": None})
    run = run_benchmark("vertex", limit=1, settings=without_project, with_audit=False)
    assert run.status == "not_run"
    assert run.metrics is None
    assert run.note
