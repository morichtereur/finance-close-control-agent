"""Tests for the routing tiers, the agent layer and the two together.

The assertions that matter most are the ones about what the model *cannot* do:
it is never called on a clean invoice, it cannot move a case to a laxer tier
however confident it is, and it cannot stop a bank-detail change from being
escalated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fcca.i2p.agent import ExceptionAgent, ExceptionAssessment, FieldCitation, ground_evidence
from fcca.i2p.models import ExceptionFinding, InvoiceResult
from fcca.i2p.repository import I2PRepository
from fcca.i2p.resolver import InvoiceResolver
from fcca.shared.config import I2PConfig, Settings
from fcca.shared.errors import FCCAError
from fcca.shared.routing import ALWAYS_ESCALATE, route
from fcca.shared.trace import TraceWriter, read_trace

CONFIG = I2PConfig()


def _route(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "exception_type": "price_variance",
        "is_exception": True,
        "document_value": 1_000.0,
        "auto_clear_max_value": CONFIG.auto_clear_max_value,
        "propose_max_value": CONFIG.propose_max_value,
        "auto_clear_min_confidence": CONFIG.auto_clear_min_confidence,
        "model_confidence": 0.95,
        "severity": "medium",
    }
    defaults.update(overrides)
    return route(**defaults)  # type: ignore[arg-type]


# ===========================================================================
class TestTheFraudControl:
    """Bank-detail changes escalate unconditionally. This is the hardcoded rule."""

    def test_a_bank_detail_change_escalates(self) -> None:
        decision = _route(exception_type="bank_details_mismatch", severity="high")
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_it_escalates_at_maximum_confidence(self) -> None:
        """Confidence is not evidence here — a good fraud looks like a clean invoice."""
        decision = _route(
            exception_type="bank_details_mismatch", model_confidence=1.0, severity="low"
        )
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_it_escalates_on_a_trivial_amount(self) -> None:
        """No value floor. A small test payment is how the account gets validated."""
        decision = _route(
            exception_type="bank_details_mismatch", document_value=1.0, model_confidence=1.0
        )
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_the_reason_explains_why_rather_than_asserting_it(self) -> None:
        decision = _route(exception_type="bank_details_mismatch")
        assert "fraud control" in decision.deciding_reason  # type: ignore[attr-defined]
        assert "independently held" in decision.deciding_reason  # type: ignore[attr-defined]

    def test_the_rule_is_not_configurable(self) -> None:
        """A configurable fraud control is one deployment mistake from being off."""
        assert "bank_details_mismatch" in ALWAYS_ESCALATE
        assert not hasattr(CONFIG, "always_escalate")


# ===========================================================================
class TestTiers:
    def test_a_small_clean_invoice_clears(self) -> None:
        decision = _route(is_exception=False, exception_type="no_exception", document_value=900.0)
        assert decision.tier == "auto_clear"  # type: ignore[attr-defined]

    def test_a_large_clean_invoice_is_proposed_not_cleared(self) -> None:
        """A clean match on a large invoice is a different risk from a small one."""
        decision = _route(
            is_exception=False, exception_type="no_exception", document_value=50_000.0
        )
        assert decision.tier == "propose_and_approve"  # type: ignore[attr-defined]

    def test_an_exception_is_never_auto_cleared(self) -> None:
        for value in (1.0, 100.0, 4_999.0):
            decision = _route(document_value=value, model_confidence=1.0, severity="low")
            assert decision.tier != "auto_clear"  # type: ignore[attr-defined]

    def test_a_high_severity_exception_escalates(self) -> None:
        decision = _route(severity="high", model_confidence=1.0)
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_a_large_exception_escalates_rather_than_being_proposed(self) -> None:
        decision = _route(document_value=60_000.0, severity="medium", model_confidence=1.0)
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_low_confidence_escalates(self) -> None:
        decision = _route(model_confidence=0.10)
        assert decision.tier == "escalate"  # type: ignore[attr-defined]

    def test_the_ordinary_case_is_a_proposal(self) -> None:
        decision = _route()
        assert decision.tier == "propose_and_approve"  # type: ignore[attr-defined]


class TestConfidenceCanOnlyTighten:
    """The load-bearing property: a model's opinion is an input to a rule, not the rule."""

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.85, 0.99, 1.0])
    def test_no_confidence_makes_an_exception_auto_clearable(self, confidence: float) -> None:
        decision = _route(model_confidence=confidence, document_value=10.0, severity="low")
        assert decision.tier != "auto_clear"  # type: ignore[attr-defined]

    def test_raising_confidence_never_loosens_the_tier(self) -> None:
        order = {"auto_clear": 0, "propose_and_approve": 1, "escalate": 2}
        previous = None
        for confidence in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
            tier = _route(model_confidence=confidence).tier  # type: ignore[attr-defined]
            if previous is not None:
                assert order[tier] <= order[previous]
            previous = tier

    def test_a_clean_invoice_tier_does_not_depend_on_confidence_at_all(self) -> None:
        """Because no model was consulted; the field exists only to be recorded."""
        low = _route(is_exception=False, exception_type="no_exception", model_confidence=0.0)
        high = _route(is_exception=False, exception_type="no_exception", model_confidence=1.0)
        assert low.tier == high.tier  # type: ignore[attr-defined]


class TestConfigurationReachesTheDecision:
    def test_raising_the_auto_clear_limit_clears_a_larger_invoice(self) -> None:
        strict = _route(is_exception=False, exception_type="no_exception", document_value=9_000.0)
        assert strict.tier == "propose_and_approve"  # type: ignore[attr-defined]
        relaxed = _route(
            is_exception=False,
            exception_type="no_exception",
            document_value=9_000.0,
            auto_clear_max_value=10_000.0,
        )
        assert relaxed.tier == "auto_clear"  # type: ignore[attr-defined]

    def test_the_limits_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            I2PConfig(auto_clear_max_value=100_000.0, propose_max_value=1_000.0)


# ===========================================================================
class TestEvidenceGrounding:
    """A citation naming a field the model was not given is stripped and recorded."""

    @pytest.fixture
    def case(self, sandbox: Settings) -> tuple[object, InvoiceResult]:
        repository = I2PRepository(sandbox)
        from fcca.i2p.engine import InvoiceEngine

        labels = json.loads(Path(sandbox.i2p_labels_path).read_text())
        target = next(
            row["invoice_id"] for row in labels if row["expected_exception"] == "price_variance"
        )
        engine = InvoiceEngine(repository, sandbox)
        return repository.invoice(target), engine.run(target)

    def _assessment(self, invoice_id: str, paths: list[str]) -> ExceptionAssessment:
        return ExceptionAssessment(
            invoice_id=invoice_id,
            classification="price_variance",
            proposed_action="block_for_price_review",
            rationale="Test.",
            evidence=[FieldCitation(field_path=path) for path in paths],
            confidence=0.9,
        )

    def test_a_real_field_survives(self, case: tuple[object, InvoiceResult]) -> None:
        invoice, result = case
        line = result.resolutions[0].line_no
        assessment = self._assessment(result.invoice_id, [f"line[{line}].price.residual_pct"])
        grounded, report = ground_evidence(assessment, invoice, result)  # type: ignore[arg-type]
        assert report.ungrounded_citations == []
        assert len(grounded.evidence) == 1

    def test_an_invented_field_is_stripped_and_recorded(
        self, case: tuple[object, InvoiceResult]
    ) -> None:
        invoice, result = case
        assessment = self._assessment(result.invoice_id, ["line[1].price.definitely_not_a_field"])
        grounded, report = ground_evidence(assessment, invoice, result)  # type: ignore[arg-type]
        assert grounded.evidence == []
        assert report.ungrounded_citations == ["line[1].price.definitely_not_a_field"]
        assert report.is_fully_grounded is False


class TestTheAgentRefusesCleanInvoices:
    def test_assessing_a_clean_result_raises(self, sandbox: Settings) -> None:
        """A clean invoice reaching a model would mean the pipeline had been rewired."""
        from fcca.shared.routing import route as route_fn

        clean = InvoiceResult(
            invoice_id="INV-CLEAN",
            category="MM",
            document_value=100.0,
            currency="EUR",
            resolutions=(),
            findings=(),
            routing=route_fn(
                exception_type="no_exception",
                is_exception=False,
                document_value=100.0,
                auto_clear_max_value=CONFIG.auto_clear_max_value,
                propose_max_value=CONFIG.propose_max_value,
                auto_clear_min_confidence=CONFIG.auto_clear_min_confidence,
            ),
            evaluated_at=datetime.now(UTC),
        )
        agent = ExceptionAgent.build(provider="mock", settings=sandbox)
        with pytest.raises(FCCAError, match="only ever"):
            agent.assess(_stub_invoice(), clean, [])


def _stub_invoice() -> object:
    from fcca.i2p.models import Invoice, InvoiceLine, PriceElements

    return Invoice(
        invoice_id="INV-CLEAN",
        vendor_id="V-10010",
        company_code="DE10",
        category="MM",
        invoice_date=datetime.now(UTC).date(),
        received_date=datetime.now(UTC).date(),
        vendor_reference="X1",
        currency="EUR",
        lines=(
            InvoiceLine(
                line_no=1,
                description="x",
                supplier_item_no="NW-8840-SS",
                quantity=1.0,
                uom="PCE",
                price=PriceElements(list_price=1.0),
                tax_rate=19.0,
            ),
        ),
        stated_bank_iban="DE44500105175407324931",
        stated_total_net=1.0,
        stated_total_tax=0.19,
        stated_total_gross=1.19,
    )


# ===========================================================================
class TestTheResolver:
    @pytest.fixture(scope="class")
    def resolved(
        self, sandbox: Settings, tmp_path_factory: pytest.TempPathFactory
    ) -> dict[str, object]:
        repository = I2PRepository(sandbox)
        trace = TraceWriter(tmp_path_factory.mktemp("resolve") / "trace.jsonl", module="i2p")
        resolver = InvoiceResolver.build(
            provider="mock", settings=sandbox, repository=repository, trace=trace
        )
        return {
            "results": {
                invoice_id: resolver.resolve(invoice_id) for invoice_id in repository.invoices
            },
            "trace": trace.path,
        }

    @pytest.fixture(scope="class")
    def labels(self, sandbox: Settings) -> dict[str, dict[str, str]]:
        return {
            row["invoice_id"]: row for row in json.loads(Path(sandbox.i2p_labels_path).read_text())
        }

    def test_the_model_is_called_on_exceptions_and_only_on_exceptions(
        self, resolved: dict[str, object]
    ) -> None:
        results = resolved["results"]
        assert isinstance(results, dict)
        for item in results.values():
            assert item.model_called == item.result.is_exception

    def test_no_false_auto_post(
        self, resolved: dict[str, object], labels: dict[str, dict[str, str]]
    ) -> None:
        """Required to be zero. A nonzero count is a failing test, not a metric."""
        results = resolved["results"]
        assert isinstance(results, dict)
        offenders = [
            invoice_id
            for invoice_id, item in results.items()
            if item.touchless and labels[invoice_id]["expected_exception"] != "no_exception"
        ]
        assert offenders == []

    def test_every_bank_detail_case_escalates_end_to_end(
        self, resolved: dict[str, object], labels: dict[str, dict[str, str]]
    ) -> None:
        results = resolved["results"]
        assert isinstance(results, dict)
        cases = [
            item
            for invoice_id, item in results.items()
            if labels[invoice_id]["expected_exception"] == "bank_details_mismatch"
        ]
        assert cases
        assert all(item.routing.tier == "escalate" for item in cases)

    def test_the_trace_shows_the_tier_before_and_after_the_model(
        self, resolved: dict[str, object], labels: dict[str, dict[str, str]]
    ) -> None:
        """Two routing records: what the rules said, then what the model changed."""
        results = resolved["results"]
        trace_path = resolved["trace"]
        assert isinstance(results, dict) and isinstance(trace_path, Path)
        target = next(
            invoice_id for invoice_id, item in results.items() if item.result.is_exception
        )
        records = read_trace(trace_path, case_id=target)
        routing_records = [r for r in records if r.step_name == "routing_decision"]
        assert len(routing_records) == 2
        assert all(r.actor == "rule" for r in routing_records)

        model_records = [r for r in records if r.actor == "model"]
        assert len(model_records) == 1
        assert model_records[0].step_name == "exception_assessment"
        assert model_records[0].prompt_version == "i2p-v1"

    def test_a_clean_invoice_has_no_model_record_in_its_trace(
        self, resolved: dict[str, object]
    ) -> None:
        results = resolved["results"]
        trace_path = resolved["trace"]
        assert isinstance(results, dict) and isinstance(trace_path, Path)
        target = next(
            invoice_id for invoice_id, item in results.items() if not item.result.is_exception
        )
        assert [r for r in read_trace(trace_path, case_id=target) if r.actor == "model"] == []

    def test_a_proposed_cost_centre_is_always_one_that_exists(
        self, resolved: dict[str, object], sandbox: Settings
    ) -> None:
        """Checked rather than trusted: a plausible invented code would post to nothing."""
        results = resolved["results"]
        assert isinstance(results, dict)
        valid = set(I2PRepository(sandbox).cost_centers)
        proposals = [
            item.assessment.assessment.proposed_cost_center
            for item in results.values()
            if item.assessment and item.assessment.assessment.proposed_cost_center
        ]
        assert proposals
        assert all(code in valid for code in proposals)

    def test_findings_are_produced_by_rules_not_by_the_model(
        self, resolved: dict[str, object]
    ) -> None:
        """The model classifies and proposes; it cannot add or remove a finding."""
        results = resolved["results"]
        assert isinstance(results, dict)
        for item in results.values():
            for finding in item.result.findings:
                assert isinstance(finding, ExceptionFinding)
                assert finding.rule_id.startswith("I2P-R-")
