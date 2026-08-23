"""Tests for the deterministic invoice-to-pay pipeline.

Three things are asserted here. That the pipeline runs the same twelve steps in
the same order on every invoice, because that fixed sequence is the control
design. That every step leaves a trace record attributed to a rule, because a
step that runs without a record is a step no reviewer can see. And that the
deterministic layer agrees with ground truth across the whole population,
including — the load-bearing one — that it never clears a real exception.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fcca.i2p.engine import STEP_RULES, InvoiceEngine
from fcca.i2p.repository import I2PRepository
from fcca.shared.config import Settings
from fcca.shared.trace import TraceWriter, read_trace

EXPECTED_STEPS = [
    "intake",
    # Second, and unconditional: it writes its record even on synthetic data
    # where there is nothing to gate, because a step that only appears when it
    # fires is a step nobody can audit.
    "extraction_confidence_gate",
    "classification",
    "duplicate_check",
    "master_data_resolution",
    "tax_code",
    "gl_derivation",
    "cost_center_derivation",
    "three_way_match",
    "quantity_check",
    "price_check",
    "tolerance_evaluation",
    "routing_decision",
]


@pytest.fixture(scope="module")
def repository(sandbox: Settings) -> I2PRepository:
    return I2PRepository(sandbox)


@pytest.fixture(scope="module")
def labels(sandbox: Settings) -> dict[str, dict[str, str]]:
    return {row["invoice_id"]: row for row in json.loads(Path(sandbox.i2p_labels_path).read_text())}


@pytest.fixture(scope="module")
def population(
    sandbox: Settings, repository: I2PRepository, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, str]:
    """Run every invoice once; the predicted primary exception per invoice."""
    trace = TraceWriter(tmp_path_factory.mktemp("i2p-run") / "trace.jsonl", module="i2p")
    engine = InvoiceEngine(repository, sandbox, trace=trace)
    return {
        invoice_id: engine.run(invoice_id).primary_exception for invoice_id in repository.invoices
    }


# ------------------------------------------------------------------ the trace
class TestTheTrace:
    def test_every_step_emits_exactly_one_record_in_a_fixed_order(
        self, sandbox: Settings, repository: I2PRepository, tmp_path: Path
    ) -> None:
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        invoice_id = next(iter(repository.invoices))
        engine.run(invoice_id)

        records = read_trace(trace.path, case_id=invoice_id)
        assert [record.step_name for record in records] == EXPECTED_STEPS

    def test_the_order_is_the_same_for_every_invoice(
        self, sandbox: Settings, repository: I2PRepository, tmp_path: Path
    ) -> None:
        """The population is only comparable if every item got the same checks."""
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        sample = list(repository.invoices)[:25]
        for invoice_id in sample:
            engine.run(invoice_id)

        for invoice_id in sample:
            steps = [r.step_name for r in read_trace(trace.path, case_id=invoice_id)]
            assert steps == EXPECTED_STEPS

    def test_every_record_is_attributed_to_a_rule_not_a_model(
        self, sandbox: Settings, repository: I2PRepository, tmp_path: Path
    ) -> None:
        """No model may touch matching, arithmetic or tolerance. The trace proves it."""
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        engine.run(next(iter(repository.invoices)))

        for record in read_trace(trace.path):
            assert record.actor == "rule"
            assert record.rule_id in STEP_RULES.values()
            assert record.model is None
            assert record.prompt_version is None

    def test_the_price_check_record_carries_both_figures(
        self,
        sandbox: Settings,
        repository: I2PRepository,
        labels: dict[str, dict[str, str]],
        tmp_path: Path,
    ) -> None:
        """A reviewer must be able to see what normalisation changed."""
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        target = next(
            invoice_id
            for invoice_id, row in labels.items()
            if row["scenario"] == "clean_mm_messy_pricing"
        )
        engine.run(target)
        record = next(
            r for r in read_trace(trace.path, case_id=target) if r.step_name == "price_check"
        )
        assert "normalised" in record.summary
        assert "naive comparison would show" in record.summary

    def test_a_rerun_appends_rather_than_replacing(
        self, sandbox: Settings, repository: I2PRepository, tmp_path: Path
    ) -> None:
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        invoice_id = next(iter(repository.invoices))
        engine.run(invoice_id)
        engine.run(invoice_id)
        assert len(read_trace(trace.path, case_id=invoice_id)) == 2 * len(EXPECTED_STEPS)


# ----------------------------------------------------------------- accuracy
class TestAgainstGroundTruth:
    def test_no_real_exception_is_cleared(
        self, population: dict[str, str], labels: dict[str, dict[str, str]]
    ) -> None:
        """The load-bearing assertion. A missed exception is a payment that should not have been made."""
        missed = [
            invoice_id
            for invoice_id, predicted in population.items()
            if labels[invoice_id]["expected_exception"] != "no_exception"
            and predicted == "no_exception"
        ]
        assert missed == []

    def test_no_clean_invoice_is_flagged(
        self, population: dict[str, str], labels: dict[str, dict[str, str]]
    ) -> None:
        """False exceptions are the cost side: every one is a reviewer's time."""
        spurious = [
            invoice_id
            for invoice_id, predicted in population.items()
            if labels[invoice_id]["expected_exception"] == "no_exception"
            and predicted != "no_exception"
        ]
        assert spurious == []

    def test_each_exception_is_classified_as_the_right_type(
        self, population: dict[str, str], labels: dict[str, dict[str, str]]
    ) -> None:
        wrong = {
            invoice_id: (labels[invoice_id]["expected_exception"], predicted)
            for invoice_id, predicted in population.items()
            if labels[invoice_id]["expected_exception"] != predicted
        }
        assert wrong == {}

    def test_every_exception_class_was_actually_exercised(self, population: dict[str, str]) -> None:
        """Guards against a suite that passes because nothing ran."""
        counts = Counter(population.values())
        for exception_type in (
            "bank_details_mismatch",
            "cost_center_missing",
            "duplicate_invoice",
            "gl_account_missing",
            "missing_or_delayed_goods_receipt",
            "price_variance",
        ):
            assert counts[exception_type] > 0, exception_type


# ------------------------------------------------------------------ behaviour
class TestSpecificBehaviours:
    def test_a_duplicate_is_reported_once_on_the_resubmission(
        self, population: dict[str, str], labels: dict[str, dict[str, str]]
    ) -> None:
        """Both halves of a pair flagging each other would double the queue."""
        flagged = sum(1 for value in population.values() if value == "duplicate_invoice")
        seeded = sum(
            1 for row in labels.values() if row["expected_exception"] == "duplicate_invoice"
        )
        assert flagged == seeded

    def test_a_missing_receipt_is_not_also_reported_as_over_billing(
        self,
        sandbox: Settings,
        repository: I2PRepository,
        labels: dict[str, dict[str, str]],
        tmp_path: Path,
    ) -> None:
        """Saying the same thing twice makes the queue longer, not clearer."""
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        target = next(
            invoice_id
            for invoice_id, row in labels.items()
            if row["scenario"] == "missing_goods_receipt"
        )
        result = engine.run(target)
        types = {finding.exception_type for finding in result.findings}
        assert "missing_or_delayed_goods_receipt" in types
        assert "quantity_variance" not in types

    def test_a_derived_gl_account_records_that_it_was_derived(
        self,
        sandbox: Settings,
        repository: I2PRepository,
        labels: dict[str, dict[str, str]],
        tmp_path: Path,
    ) -> None:
        """A value the system worked out must not look like one the vendor supplied."""
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        target = next(
            invoice_id
            for invoice_id, row in labels.items()
            if row["scenario"] == "gl_account_not_stated"
        )
        result = engine.run(target)
        derived = [r for r in result.resolutions if r.gl_source == "derived"]
        assert derived
        assert all(r.gl_account is not None for r in derived)

    def test_bank_details_are_checked_even_when_everything_else_matches(
        self,
        sandbox: Settings,
        repository: I2PRepository,
        labels: dict[str, dict[str, str]],
        tmp_path: Path,
    ) -> None:
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        target = next(
            invoice_id
            for invoice_id, row in labels.items()
            if row["expected_exception"] == "bank_details_mismatch"
        )
        result = engine.run(target)
        finding = next(f for f in result.findings if f.exception_type == "bank_details_mismatch")
        assert finding.severity == "high"
        assert finding.evidence["master_iban"] != finding.evidence["stated_iban"]

    def test_precedence_is_by_severity_not_by_iteration_order(
        self, sandbox: Settings, repository: I2PRepository, tmp_path: Path
    ) -> None:
        """Routing must not depend on which rule happened to run first."""
        from datetime import UTC, datetime

        from fcca.i2p.models import ExceptionFinding, InvoiceResult
        from fcca.shared.routing import route

        routing = route(
            exception_type="bank_details_mismatch",
            is_exception=True,
            document_value=1000.0,
            auto_clear_max_value=sandbox.i2p.auto_clear_max_value,
            propose_max_value=sandbox.i2p.propose_max_value,
            auto_clear_min_confidence=sandbox.i2p.auto_clear_min_confidence,
        )
        result = InvoiceResult(
            invoice_id="INV-TEST",
            category="MM",
            document_value=1000.0,
            currency="EUR",
            resolutions=(),
            findings=(
                ExceptionFinding(
                    rule_id="I2P-R-005",
                    exception_type="gl_account_missing",
                    severity="low",
                    detail="derived",
                ),
                ExceptionFinding(
                    rule_id="I2P-R-001",
                    exception_type="bank_details_mismatch",
                    severity="high",
                    detail="mismatch",
                ),
            ),
            routing=routing,
            evaluated_at=datetime.now(UTC),
        )
        assert result.primary_exception == "bank_details_mismatch"

    def test_a_clean_invoice_has_no_findings_at_all(
        self,
        sandbox: Settings,
        repository: I2PRepository,
        labels: dict[str, dict[str, str]],
        tmp_path: Path,
    ) -> None:
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        engine = InvoiceEngine(repository, sandbox, trace=trace)
        target = next(
            invoice_id
            for invoice_id, row in labels.items()
            if row["scenario"] == "clean_mm_messy_pricing"
        )
        result = engine.run(target)
        assert result.findings == ()
        assert result.is_exception is False
        assert result.primary_exception == "no_exception"


# ---------------------------------------------------------------- repository
class TestRepository:
    def test_receipts_convert_to_base_units(self, repository: I2PRepository) -> None:
        """Summing receipts in their printed units would manufacture variances."""
        for (po_id, po_line), receipts in repository.goods_receipts.items():
            order = repository.purchase_order(po_id)
            assert order is not None
            line = order.line(po_line)
            assert line is not None
            base = repository.received_base_quantity(po_id, po_line, line.material_id)
            raw = sum(r.quantity for r in receipts)
            assert base >= raw - 1e-6  # conversion never shrinks a quantity
            break

    def test_an_unknown_invoice_raises_a_meaningful_error(self, repository: I2PRepository) -> None:
        from fcca.shared.errors import DataNotFoundError

        with pytest.raises(DataNotFoundError, match="INV-NOPE"):
            repository.invoice("INV-NOPE")
