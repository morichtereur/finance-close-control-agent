"""The integration seams, and the five properties that make them worth having.

Every test here asserts something that would be a real incident if it stopped
being true: a model reasoning about a number nobody read correctly, a payload
built for an invoice a person was supposed to look at, a dry-run flag that turned
out to be flippable, a document paid twice, or a false auto-post appearing the
moment the input got noisy.
"""

from __future__ import annotations

import pathlib

import pytest

from fcca.i2p.degrade import build_payloads
from fcca.i2p.engine import InvoiceEngine
from fcca.i2p.extraction import (
    DocumentSource,
    ExtractedField,
    ExtractionPayload,
    SyntheticSource,
)
from fcca.i2p.posting import (
    PostedKeyLedger,
    PostingBlocked,
    SapODataTarget,
    SimulatedPosting,
    posting_key,
)
from fcca.i2p.repository import I2PRepository
from fcca.i2p.resolver import InvoiceResolver
from fcca.shared.config import Settings
from fcca.shared.providers.base import get_llm
from fcca.shared.trace import TraceWriter, read_trace


@pytest.fixture(scope="module")
def repository(sandbox: Settings) -> I2PRepository:
    return I2PRepository(sandbox)


class CountingAgent:
    """An agent that refuses to be called and counts the attempts.

    Used rather than a mock so the failure is loud: if the pipeline ever reaches
    the model on a gated invoice, the test does not merely record a count — the
    run raises where it happened.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def assess(self, invoice, result, valid_cost_centers):
        self.calls.append(result.invoice_id)
        raise AssertionError(
            f"the agent layer was reached for {result.invoice_id}, which is "
            f"extraction_gated={result.extraction_gated}"
        )


def _degraded_source(repository: I2PRepository, ids: list[str], *, seed: int, rate: float):
    """A document source whose readings are bad enough to trip the gate."""
    invoices = [repository.invoice(i) for i in ids]
    payloads = build_payloads(invoices, seed=seed, dropout_rate=0.0, digit_confusion_rate=rate)
    return DocumentSource(payloads, SyntheticSource(repository))


# ---------------------------------------------------------------------------
# 1. No model call on a gated invoice
# ---------------------------------------------------------------------------


class TestTheGateComesBeforeTheModel:
    def test_a_gated_invoice_never_reaches_the_agent(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        ids = sorted(repository.invoices)[:40]
        source = _degraded_source(repository, ids, seed=11, rate=1.0)
        trace = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        agent = CountingAgent()
        resolver = InvoiceResolver(
            repository=repository,
            settings=sandbox,
            agent=agent,  # type: ignore[arg-type]
            trace=trace,
            source=source,
            ledger=PostedKeyLedger(tmp_path / "keys.jsonl"),
        )

        gated = 0
        for invoice_id in ids:
            resolved = resolver.resolve(invoice_id)
            if resolved.result.extraction_gated:
                gated += 1
                assert resolved.model_called is False
                assert resolved.routing.tier == "escalate"

        # The degradation has to actually bite, or the test asserts nothing.
        assert gated > 0, "no invoice was gated; the fixture is not exercising the gate"
        assert agent.calls == []

    def test_the_gate_reason_names_the_configured_field(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        invoice_id = sorted(repository.invoices)[0]
        base = repository.invoice(invoice_id)
        payload = ExtractionPayload(
            document_id="DOC-1",
            invoice_id=invoice_id,
            engine="tesseract",
            engine_version="5.3.4",
            fields=(
                ExtractedField(
                    field_path="stated_total_gross",
                    value=base.stated_total_gross,
                    confidence=0.11,
                ),
            ),
        )
        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            source=DocumentSource({invoice_id: payload}, SyntheticSource(repository)),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        result = engine.run(invoice_id)

        assert result.extraction_gated is True
        assert result.extraction_gate_reasons == ("low_confidence_field:stated_total_gross",)
        assert result.routing.tier == "escalate"

    def test_the_gate_step_is_recorded_even_when_it_does_nothing(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        """A control that leaves no record when it passes cannot be audited."""
        invoice_id = sorted(repository.invoices)[0]
        path = tmp_path / "t.jsonl"
        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(path, module="i2p"),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        result = engine.run(invoice_id)

        steps = [
            r for r in read_trace(path, invoice_id) if r.step_name == "extraction_confidence_gate"
        ]
        assert len(steps) == 1
        assert steps[0].outcome == "pass"
        assert result.extraction_gated is False


# ---------------------------------------------------------------------------
# 2. A payload exists only where nobody is going to look
# ---------------------------------------------------------------------------


class TestNothingOutsideAutoClearProducesAPayload:
    def test_no_payload_for_any_invoice_a_person_must_see(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        resolver = InvoiceResolver(
            repository=repository,
            settings=sandbox,
            agent=None,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        checked = 0
        for invoice_id in sorted(repository.invoices):
            result = resolver.engine.run(invoice_id)
            if result.is_exception:
                continue  # would need a model; covered by the evaluation test
            resolved = resolver.resolve(invoice_id)
            checked += 1
            if resolved.routing.tier == "auto_clear":
                assert resolved.posting is not None
            else:
                assert resolved.posting is None
        assert checked > 0

    def test_the_adapter_refuses_independently_of_the_caller(
        self, repository: I2PRepository, sandbox: Settings, tmp_path
    ) -> None:
        """Belt and braces: the adapter checks the tier even if the caller forgot."""
        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        escalated = next(
            engine.run(i)
            for i in sorted(repository.invoices)
            if engine.run(i).routing.tier != "auto_clear"
        )
        invoice = repository.invoice(escalated.invoice_id)
        for target in (SimulatedPosting(), SapODataTarget()):
            with pytest.raises(PostingBlocked, match="auto_clear"):
                target.build(invoice, escalated, escalated.provenance)


# ---------------------------------------------------------------------------
# 3. Dry run is not a setting
# ---------------------------------------------------------------------------


class TestDryRunCannotBeDisabled:
    def test_dispatch_always_raises(self, repository: I2PRepository) -> None:
        invoice = repository.invoice(sorted(repository.invoices)[0])
        payload_target = SapODataTarget()
        assert payload_target.dry_run is True
        with pytest.raises(PostingBlocked, match="no transport"):
            payload_target.dispatch(None)  # type: ignore[arg-type]
        with pytest.raises(PostingBlocked, match="never dispatches"):
            SimulatedPosting().dispatch(None)  # type: ignore[arg-type]
        assert invoice is not None

    def test_dry_run_is_read_only(self) -> None:
        """Not a constructor argument, not a setting — a property with no setter."""
        target = SapODataTarget()
        with pytest.raises(AttributeError):
            target.dry_run = False  # type: ignore[misc]

    def test_no_http_client_is_importable_from_the_module(self) -> None:
        """There is no disabled transport, because there is no transport."""
        import fcca.i2p.posting as posting

        source = posting.__file__
        assert source is not None
        text = pathlib.Path(source).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import httpx", "urllib.request", "aiohttp"):
            assert forbidden not in text


# ---------------------------------------------------------------------------
# 4. A replayed document is one key and one duplicate
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_replaying_an_invoice_claims_one_key_and_raises_one_duplicate(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        ledger = PostedKeyLedger(tmp_path / "keys.jsonl")
        resolver = InvoiceResolver(
            repository=repository,
            settings=sandbox,
            agent=None,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=ledger,
        )
        cleared = next(
            invoice_id
            for invoice_id in sorted(repository.invoices)
            if resolver.resolve(invoice_id).routing.tier == "auto_clear"
        )

        first = resolver.resolve(cleared)
        assert first.posting is not None
        keys_after_first = len(ledger.keys())

        second = resolver.resolve(cleared)

        # One key, not two: the same document is the same posting.
        assert len(ledger.keys()) == keys_after_first
        assert second.posting is not None
        assert second.posting.posting_key == first.posting.posting_key

    def test_a_different_invoice_with_the_same_document_number_is_a_duplicate(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        """The case the population scan misses: same document, next run."""
        ledger = PostedKeyLedger(tmp_path / "keys.jsonl")
        invoice_id = sorted(repository.invoices)[0]
        invoice = repository.invoice(invoice_id)

        # A prior run already claimed this vendor + reference + year.
        key = posting_key(invoice.vendor_id, invoice.vendor_reference, invoice.invoice_date)
        ledger.path.write_text(
            f'{{"posting_key":"{key}","invoice_id":"INV-PRIOR-RUN","target":"simulated",'
            f'"dry_run":true,"recorded_at":"2026-01-01T00:00:00+00:00"}}\n',
            encoding="utf-8",
        )

        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=ledger,
        )
        result = engine.run(invoice_id)

        duplicates = [f for f in result.findings if f.exception_type == "duplicate_invoice"]
        assert len(duplicates) == 1
        assert "INV-PRIOR-RUN" in duplicates[0].detail
        assert result.routing.tier != "auto_clear"

    def test_the_key_survives_reference_formatting(self, repository: I2PRepository) -> None:
        """'INV 4471', 'inv-4471' and 'INV-4471' are one document."""
        from datetime import date

        day = date(2026, 5, 1)
        assert (
            posting_key("V1", "INV-4471", day)
            == posting_key("V1", "inv 4471", day)
            == posting_key("v1", "INV4471", day)
        )
        # A different fiscal year is a different posting.
        assert posting_key("V1", "INV-4471", day) != posting_key("V1", "INV-4471", date(2027, 5, 1))


# ---------------------------------------------------------------------------
# 5. Noise degrades the touchless rate; it does not produce a false auto-post
# ---------------------------------------------------------------------------


class TestNoiseDoesNotProduceAFalseAutoPost:
    @pytest.mark.parametrize("rate", [0.15, 0.40])
    def test_false_auto_post_stays_zero_under_extraction_noise(
        self, sandbox: Settings, repository: I2PRepository, tmp_path, rate: float
    ) -> None:
        """The required-passing invariant, restated for a degraded input.

        A false auto-post is an invoice cleared without a person that a person
        should have seen. Noise must cost throughput, never safety: the gate is
        allowed to escalate anything it is unsure of, and is not allowed to let
        one through.
        """
        ids = sorted(repository.invoices)[:60]
        source = _degraded_source(repository, ids, seed=7, rate=rate)
        resolver = InvoiceResolver(
            repository=repository,
            settings=sandbox,
            agent=None,
            trace=TraceWriter(tmp_path / f"t{rate}.jsonl", module="i2p"),
            source=source,
            ledger=PostedKeyLedger(tmp_path / f"k{rate}.jsonl"),
        )

        false_auto_posts = []
        for invoice_id in ids:
            result = resolver.engine.run(invoice_id)
            if result.routing.tier != "auto_clear":
                continue
            # Cleared without a person. It must have no finding and no gate hit.
            if result.is_exception or result.extraction_gated:
                false_auto_posts.append(invoice_id)

        assert false_auto_posts == []

    def test_noise_visibly_reduces_the_touchless_rate(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        """The contrast that makes the gate legible rather than theoretical."""
        ids = sorted(repository.invoices)[:60]

        def touchless(source, tag: str) -> float:
            engine = InvoiceEngine(
                repository,
                sandbox,
                trace=TraceWriter(tmp_path / f"{tag}.jsonl", module="i2p"),
                source=source,
                ledger=PostedKeyLedger(tmp_path / f"{tag}-keys.jsonl"),
            )
            cleared = sum(1 for i in ids if engine.run(i).routing.tier == "auto_clear")
            return cleared / len(ids)

        clean = touchless(SyntheticSource(repository), "clean")
        noisy = touchless(_degraded_source(repository, ids, seed=3, rate=0.5), "noisy")

        assert noisy < clean, (
            f"extraction noise did not reduce the touchless rate ({noisy:.3f} vs {clean:.3f}); "
            "either the gate is not firing or the degradation is too gentle to matter"
        )


# ---------------------------------------------------------------------------
# The rule that keeps a model out of the confidence number
# ---------------------------------------------------------------------------


class TestConfidenceIsNotModelProduced:
    @pytest.mark.parametrize(
        "engine",
        ["gpt-4o-vision", "claude-sonnet-4-5", "azure-llm-reader", "gemini-ocr"],
    )
    def test_a_model_scored_payload_is_refused(self, engine: str) -> None:
        with pytest.raises(ValueError, match="language model"):
            ExtractionPayload(
                document_id="D",
                invoice_id="INV-1",
                engine=engine,
                engine_version="1",
                fields=(ExtractedField(field_path="currency", value="EUR", confidence=0.99),),
            )

    @pytest.mark.parametrize("engine", ["tesseract", "azure-di", "klippa", "abbyy"])
    def test_a_real_extraction_engine_is_accepted(self, engine: str) -> None:
        payload = ExtractionPayload(
            document_id="D",
            invoice_id="INV-1",
            engine=engine,
            engine_version="1",
            fields=(ExtractedField(field_path="currency", value="EUR", confidence=0.99),),
        )
        assert payload.engine == engine

    def test_a_payload_naming_a_field_the_schema_lacks_is_caught(self) -> None:
        payload = ExtractionPayload(
            document_id="D",
            invoice_id="INV-1",
            engine="tesseract",
            engine_version="1",
            fields=(
                ExtractedField(field_path="total_including_shipping", value=1.0, confidence=0.9),
            ),
        )
        assert payload.unknown_paths() == ("total_including_shipping",)


# ---------------------------------------------------------------------------
# The mapping, which is the part worth reviewing
# ---------------------------------------------------------------------------


class TestTheSapMapping:
    def test_the_payload_carries_the_service_shape(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        cleared = next(
            engine.run(i)
            for i in sorted(repository.invoices)
            if engine.run(i).routing.tier == "auto_clear"
        )
        invoice = repository.invoice(cleared.invoice_id)
        payload = SapODataTarget().build(invoice, cleared, cleared.provenance)

        assert payload.service == "API_SUPPLIERINVOICE_PROCESS_SRV"
        assert payload.dry_run is True
        for field in (
            "CompanyCode",
            "DocumentDate",
            "PostingDate",
            "InvoicingParty",
            "DocumentCurrency",
            "InvoiceGrossAmount",
            "SupplierInvoiceIDByInvcgParty",
        ):
            assert field in payload.document
        assert payload.document["to_SuplrInvcItemPurOrdRef"]["results"]
        # Amounts are strings in this service, and dates are OData literals.
        assert isinstance(payload.document["InvoiceGrossAmount"], str)
        assert payload.document["DocumentDate"].startswith("/Date(")

    def test_every_mapped_field_says_where_it_came_from(
        self, sandbox: Settings, repository: I2PRepository, tmp_path
    ) -> None:
        engine = InvoiceEngine(
            repository,
            sandbox,
            trace=TraceWriter(tmp_path / "t.jsonl", module="i2p"),
            ledger=PostedKeyLedger(tmp_path / "k.jsonl"),
        )
        cleared = next(
            engine.run(i)
            for i in sorted(repository.invoices)
            if engine.run(i).routing.tier == "auto_clear"
        )
        invoice = repository.invoice(cleared.invoice_id)
        payload = SapODataTarget().build(invoice, cleared, cleared.provenance)

        for entry in payload.mapping:
            # Either it came from a named invoice field, or it is derived and
            # the note says how. Nothing appears without an account of itself.
            assert entry.source_path is not None or entry.note


def test_the_llm_is_never_imported_by_the_deterministic_seam_modules() -> None:
    """The architectural rule, asserted rather than asserted-in-prose."""
    import fcca.i2p.degrade as degrade
    import fcca.i2p.extraction as extraction
    import fcca.i2p.posting as posting

    for module in (extraction, posting, degrade):
        source = module.__file__
        assert source is not None
        text = pathlib.Path(source).read_text(encoding="utf-8")
        assert "get_llm" not in text
        assert "invoke_structured" not in text
    assert callable(get_llm)
