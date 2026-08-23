"""Tests for the invoice-to-pay evaluation.

Two kinds of test live here.

The first kind checks the metric functions themselves, against deliberately
wrong answers as well as correct ones. A metric that only ever sees a perfect
run is not tested; it is merely exercised.

The second kind is
:meth:`TestTheSafetyProperty.test_false_auto_post_count_is_zero`, which is not a
metric at all. A false auto-post is an invoice the system cleared without a
person that ground truth says was an exception. There is no acceptable nonzero
value, so it is asserted rather than reported, and the build fails on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fcca.i2p.evaluate import run_evaluation, run_slug, write_results
from fcca.i2p.evaluation import (
    CLASSES,
    ClassMetrics,
    EvaluationReport,
    confusion_as_rows,
    evaluate,
    exception_rate_by_type,
    render_report,
)
from fcca.shared.config import Settings


@pytest.fixture(scope="module")
def report(sandbox: Settings) -> EvaluationReport:
    evaluation, _ = run_evaluation(provider="mock", settings=sandbox)
    return evaluation


@pytest.fixture(scope="module")
def labels(sandbox: Settings) -> dict[str, str]:
    return {
        row["invoice_id"]: row["expected_exception"]
        for row in json.loads(Path(sandbox.i2p_labels_path).read_text())
    }


# ===========================================================================
class TestTheSafetyProperty:
    """Not a metric. A required property, asserted."""

    def test_false_auto_post_count_is_zero(self, report: EvaluationReport) -> None:
        assert report.false_auto_post_count == 0, (
            "The system cleared invoices without a person that ground truth says were "
            f"exceptions: {report.false_auto_post_ids}. There is no acceptable nonzero "
            "value for this."
        )

    def test_the_report_knows_it_is_safe(self, report: EvaluationReport) -> None:
        assert report.is_safe is True

    def test_a_run_with_a_false_auto_post_is_reported_as_unsafe(self) -> None:
        """The detector must be capable of firing, or the assertion above proves nothing."""
        unsafe = EvaluationReport(
            provider="mock",
            model="stub",
            invoices=1,
            tier_counts={"auto_clear": 1},
            touchless_rate=1.0,
            false_auto_post_count=1,
            false_auto_post_ids=["INV-00001"],
            confusion={a: dict.fromkeys(CLASSES, 0) for a in CLASSES},
            per_class=[],
            actual_counts={},
            predicted_counts={},
            model_calls=0,
            invoices_with_findings=0,
            ungrounded_citation_invoices=0,
            mean_confidence=None,
            settings_snapshot={},
        )
        assert unsafe.is_safe is False

    def test_the_cli_exits_nonzero_when_unsafe(self) -> None:
        """CI must notice even if nobody reads the table."""
        import inspect

        from fcca.i2p import evaluate as module

        source = inspect.getsource(module.main)
        assert "if not report.is_safe" in source
        assert "return 1" in source


# ===========================================================================
class TestTouchlessRate:
    def test_it_is_the_auto_clear_share_and_nothing_else(self, report: EvaluationReport) -> None:
        expected = report.tier_counts.get("auto_clear", 0) / report.invoices
        assert report.touchless_rate == pytest.approx(expected)

    def test_the_tiers_account_for_every_invoice(self, report: EvaluationReport) -> None:
        assert sum(report.tier_counts.values()) == report.invoices

    def test_it_is_reported_next_to_the_exception_rate(self, report: EvaluationReport) -> None:
        """A system that flags nothing has a wonderful touchless rate."""
        text = render_report(report)
        assert "touchless rate" in text
        assert "Exception rate by type" in text


class TestModelUsage:
    def test_the_model_is_called_exactly_on_the_invoices_with_findings(
        self, report: EvaluationReport
    ) -> None:
        """The touchless rate is a property of the rules, not of a model."""
        assert report.model_calls == report.invoices_with_findings

    def test_no_invoice_cited_evidence_it_was_not_given(self, report: EvaluationReport) -> None:
        assert report.ungrounded_citation_invoices == 0


# ===========================================================================
class TestConfusionMatrix:
    def test_it_totals_to_the_population(self, report: EvaluationReport) -> None:
        total = sum(v for row in report.confusion.values() for v in row.values())
        assert total == report.invoices

    def test_the_axes_are_stable_even_for_absent_classes(self, report: EvaluationReport) -> None:
        assert set(report.confusion) == set(CLASSES)
        for row in report.confusion.values():
            assert set(row) == set(CLASSES)

    def test_row_totals_match_the_actual_counts(self, report: EvaluationReport) -> None:
        for exception_type in CLASSES:
            row_total = sum(report.confusion[exception_type].values())
            assert row_total == report.actual_counts.get(exception_type, 0)

    def test_long_form_rows_omit_empty_cells(self, report: EvaluationReport) -> None:
        rows = confusion_as_rows(report)
        assert rows
        assert all(row["count"] for row in rows)
        assert sum(int(row["count"]) for row in rows) == report.invoices  # type: ignore[arg-type]


class TestPerClassMetrics:
    def test_metrics_are_computed_for_every_class(self, report: EvaluationReport) -> None:
        assert {m.exception_type for m in report.per_class} == set(CLASSES)

    def test_a_perfect_class_scores_one(self) -> None:
        metrics = ClassMetrics(
            exception_type="price_variance",
            support=10,
            predicted=10,
            true_positives=10,
            false_positives=0,
            false_negatives=0,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_over_prediction_costs_precision_not_recall(self) -> None:
        metrics = ClassMetrics(
            exception_type="price_variance",
            support=10,
            predicted=20,
            true_positives=10,
            false_positives=10,
            false_negatives=0,
        )
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == 1.0

    def test_under_prediction_costs_recall_not_precision(self) -> None:
        metrics = ClassMetrics(
            exception_type="price_variance",
            support=10,
            predicted=5,
            true_positives=5,
            false_positives=0,
            false_negatives=5,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == pytest.approx(0.5)

    def test_a_class_that_never_occurs_is_absent_not_failed(self) -> None:
        """0.000 would read as a failure; the class simply never came up."""
        metrics = ClassMetrics(
            exception_type="quantity_variance",
            support=0,
            predicted=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        assert metrics.is_absent is True

    def test_a_class_that_was_predicted_but_never_occurred_is_not_absent(self) -> None:
        metrics = ClassMetrics(
            exception_type="quantity_variance",
            support=0,
            predicted=3,
            true_positives=0,
            false_positives=3,
            false_negatives=0,
        )
        assert metrics.is_absent is False
        assert metrics.precision == 0.0


class TestScoringAgainstWrongAnswers:
    """The metric functions must be able to report a bad run, or they measure nothing."""

    def test_a_deliberately_mislabelled_run_scores_badly(
        self, sandbox: Settings, labels: dict[str, str]
    ) -> None:
        _, resolved = run_evaluation(provider="mock", settings=sandbox, limit=40)
        # Claim every invoice was a duplicate. Nothing about the run changes;
        # only the ground truth it is scored against.
        wrong = {item.invoice_id: "duplicate_invoice" for item in resolved}
        scored = evaluate(resolved, wrong, "mock", "stub", {})
        assert scored.exact_agreement < 0.5
        duplicate = next(m for m in scored.per_class if m.exception_type == "duplicate_invoice")
        assert duplicate.recall < 1.0

    def test_scoring_a_correct_run_against_correct_labels_agrees(
        self, sandbox: Settings, labels: dict[str, str]
    ) -> None:
        _, resolved = run_evaluation(provider="mock", settings=sandbox, limit=40)
        scored = evaluate(
            resolved,
            {item.invoice_id: labels[item.invoice_id] for item in resolved},
            "mock",
            "stub",
            {},
        )
        assert scored.exact_agreement == pytest.approx(1.0)


# ===========================================================================
class TestReportOutput:
    def test_the_thresholds_in_force_are_recorded(self, report: EvaluationReport) -> None:
        """An old report must be rereadable against the config of the day it was made."""
        assert report.settings_snapshot["price_tolerance_pct"] > 0
        assert "auto_clear_max_value" in report.settings_snapshot
        assert "random_seed" in report.settings_snapshot

    def test_rendering_covers_every_required_metric(self, report: EvaluationReport) -> None:
        text = render_report(report)
        for heading in (
            "Routing",
            "Safety",
            "Exception rate by type",
            "Per-class precision / recall",
            "Confusion matrix",
            "Model usage",
        ):
            assert heading in text

    def test_exception_rates_sum_to_one(self, report: EvaluationReport) -> None:
        rates = exception_rate_by_type(report)
        assert sum(rates.values()) == pytest.approx(1.0)

    def test_results_are_written_where_the_readme_can_quote_them(
        self, sandbox: Settings, report: EvaluationReport
    ) -> None:
        _, resolved = run_evaluation(provider="mock", settings=sandbox, limit=20)
        report_path, detail_path = write_results(report, resolved, sandbox)
        assert report_path.exists()
        assert detail_path.exists()
        payload = json.loads(report_path.read_text())
        # Derived properties must be written, so a consumer need not reimplement
        # the arithmetic and risk disagreeing with the report.
        assert "exact_agreement" in payload
        assert payload["per_class"][0]["precision"] is not None

    def test_the_slug_is_filesystem_safe(self) -> None:
        assert run_slug("bedrock", "eu.anthropic.claude-sonnet-4-5:0") == (
            "bedrock__eu-anthropic-claude-sonnet-4-5-0"
        )
