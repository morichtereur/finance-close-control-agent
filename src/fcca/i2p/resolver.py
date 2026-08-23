"""End-to-end resolution of one invoice: rules, then model, then routing.

The sequence is the architecture:

.. code-block:: text

    InvoiceEngine.run()          twelve deterministic steps, incl. a first
                                 routing decision taken with no model input
      |
      +-- clean ---------------> done. No model is called, ever.
      |
      +-- exception -----------> ExceptionAgent.assess()   [actor = model]
                                 route() again, with confidence  [actor = rule]

Two properties follow from writing it this way, and both are tested.

**A clean invoice never reaches a model.** The touchless rate is therefore a
property of the deterministic layer. If it were a model deciding which invoices
were fine, the rate would move with the model, the prompt and the temperature,
and it would not be a control.

**The model can only tighten.** The engine's routing record shows the tier the
rules alone assigned; this module's shows the tier after the model's confidence
was added. Confidence is only ever consulted to move a case *toward* scrutiny —
there is no path in :func:`fcca.shared.routing.route` where a higher confidence
moves a case to a laxer tier. The two trace records side by side are the
evidence, which is better than a claim in a docstring.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from fcca.i2p.agent import AssessmentOutcome, ExceptionAgent, failed_assessment
from fcca.i2p.engine import InvoiceEngine
from fcca.i2p.extraction import InvoiceSource
from fcca.i2p.models import InvoiceResult
from fcca.i2p.posting import (
    PostedKeyLedger,
    PostingBlocked,
    PostingPayload,
    PostingTarget,
    get_target,
)
from fcca.i2p.prompts import PROMPT_VERSION
from fcca.i2p.repository import I2PRepository
from fcca.shared.config import ProviderName, Settings, get_settings
from fcca.shared.errors import FCCAError, StructuredOutputError
from fcca.shared.models import GroundingReport, RunMetadata
from fcca.shared.routing import RoutingDecision, route
from fcca.shared.trace import TraceWriter

logger = logging.getLogger(__name__)


class ResolvedInvoice(BaseModel):
    """Everything that happened to one invoice."""

    result: InvoiceResult
    routing: RoutingDecision = Field(description="The final tier, after any model input.")
    assessment: AssessmentOutcome | None = Field(
        default=None, description="None where no model was called — that is, on clean invoices."
    )
    model_called: bool
    posting: PostingPayload | None = Field(
        default=None,
        description=(
            "The payload that would post, for auto_clear invoices only. None everywhere "
            "else — a payload exists precisely where no person is going to look at one."
        ),
    )
    resolved_at: datetime

    @property
    def invoice_id(self) -> str:
        return self.result.invoice_id

    @property
    def touchless(self) -> bool:
        """Cleared without a person. The headline metric, defined in one place."""
        return self.routing.tier == "auto_clear"


class InvoiceResolver:
    """Runs the full pipeline for one invoice."""

    def __init__(
        self,
        repository: I2PRepository | None = None,
        settings: Settings | None = None,
        agent: ExceptionAgent | None = None,
        trace: TraceWriter | None = None,
        source: InvoiceSource | None = None,
        ledger: PostedKeyLedger | None = None,
        target: PostingTarget | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or I2PRepository(self.settings)
        self.trace = trace or TraceWriter(self.settings.i2p_trace_path, module="i2p")
        self.ledger = ledger or PostedKeyLedger(self.settings.i2p_posted_keys_path)
        self.engine = InvoiceEngine(
            self.repository, self.settings, trace=self.trace, source=source, ledger=self.ledger
        )
        self.target = target or get_target(self.settings.i2p.posting_target)
        self._agent = agent

    @classmethod
    def build(
        cls,
        provider: ProviderName | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
        repository: I2PRepository | None = None,
        trace: TraceWriter | None = None,
    ) -> InvoiceResolver:
        settings = settings or get_settings()
        return cls(
            repository=repository,
            settings=settings,
            agent=ExceptionAgent.build(provider, model_name, settings),
            trace=trace,
        )

    @property
    def agent(self) -> ExceptionAgent:
        if self._agent is None:
            self._agent = ExceptionAgent.build(settings=self.settings)
        return self._agent

    # ------------------------------------------------------------------- run
    def resolve(self, invoice_id: str) -> ResolvedInvoice:
        config = self.settings.i2p
        result = self.engine.run(invoice_id)

        if not result.model_may_be_called:
            # No model call. Not an optimisation — the point. Two ways to land
            # here: the invoice is clean, or we did not read it well enough to
            # ask anything about it. The second escalates; asking a model to
            # explain a misread amount produces a fluent account of a number
            # that was never on the document.
            return ResolvedInvoice(
                result=result,
                routing=result.routing,
                assessment=None,
                model_called=False,
                posting=self._posting_for(result),
                resolved_at=datetime.now(UTC),
            )

        invoice = self.repository.invoice(invoice_id)
        valid_cost_centers = [
            code
            for code, centre in self.repository.cost_centers.items()
            if centre.company_code == invoice.company_code
        ]

        try:
            outcome: AssessmentOutcome | None = self.agent.assess(
                invoice, result, valid_cost_centers
            )
        except (StructuredOutputError, FCCAError) as exc:
            logger.warning("assessment failed for %s: %s", invoice_id, exc)
            outcome = None
            confidence = 0.0
            failure_reason = str(exc)
        else:
            assert outcome is not None
            confidence = outcome.assessment.confidence
            failure_reason = ""

        if outcome is not None:
            self.trace.step(
                case_id=invoice_id,
                step_name="exception_assessment",
                actor="model",
                model=outcome.run.model,
                prompt_version=PROMPT_VERSION,
                inputs={"prompt_sha256": outcome.run.prompt_sha256},
                outcome=outcome.assessment.classification,
                summary=(
                    f"Classified as {outcome.assessment.classification}; proposes "
                    f"{outcome.assessment.proposed_action} at confidence "
                    f"{confidence:.2f}, citing "
                    f"{outcome.grounding.grounded_citations} field(s)."
                ),
                detail={
                    "proposed_action": outcome.assessment.proposed_action,
                    "proposed_cost_center": outcome.assessment.proposed_cost_center,
                    "confidence": confidence,
                    "evidence": [c.field_path for c in outcome.assessment.evidence],
                    "ungrounded": outcome.grounding.ungrounded_citations,
                    "rationale": outcome.assessment.rationale,
                },
            )
        else:
            self.trace.step(
                case_id=invoice_id,
                step_name="exception_assessment",
                actor="rule",
                rule_id="I2P-S-13",
                inputs={"invoice_id": invoice_id},
                outcome="assessment_failed",
                summary=(
                    "The model produced no valid assessment; the case proceeds at "
                    "confidence zero and is escalated."
                ),
                detail={"error": failure_reason},
            )

        extra: list[str] = []
        if outcome is not None and outcome.grounding.ungrounded_citations:
            extra.append(
                "The model cited evidence that was not supplied to it: "
                + ", ".join(outcome.grounding.ungrounded_citations)
                + ". A conclusion resting on a field that does not exist is not a "
                "conclusion a person should approve on sight."
            )
        if outcome is None:
            extra.append(
                "Automated assessment failed and was not completed; a failure is never a pass."
            )

        routing = route(
            exception_type=result.primary_exception,
            is_exception=True,
            document_value=result.document_value,
            auto_clear_max_value=config.auto_clear_max_value,
            propose_max_value=config.propose_max_value,
            auto_clear_min_confidence=config.auto_clear_min_confidence,
            model_confidence=confidence,
            severity=_severity_of(result),
            extra_escalations=extra,
        )

        self.trace.step(
            case_id=invoice_id,
            step_name="routing_decision",
            actor="rule",
            rule_id="I2P-S-12",
            inputs={
                "exception_type": result.primary_exception,
                "document_value": result.document_value,
                "model_confidence": confidence,
            },
            outcome=routing.tier,
            summary=(
                f"Routed to {routing.tier}"
                + (
                    f", tightened from {result.routing.tier}"
                    if routing.tier != result.routing.tier
                    else ""
                )
                + f": {_first_clause(routing.deciding_reason)}"
            ),
            detail={"reasons": routing.reasons, "tier_before_model": result.routing.tier},
        )

        return ResolvedInvoice(
            result=result,
            routing=routing,
            assessment=outcome
            or AssessmentOutcome(
                assessment=failed_assessment(invoice_id, failure_reason),
                grounding=_empty_grounding(),
                run=_null_run(self.settings),
                raw_output="",
            ),
            model_called=True,
            posting=self._posting_for(result, routing),
            resolved_at=datetime.now(UTC),
        )

    # --------------------------------------------------------------- posting
    def _posting_for(
        self, result: InvoiceResult, routing: RoutingDecision | None = None
    ) -> PostingPayload | None:
        """Build the payload a cleared invoice would post, and claim its key.

        Only ``auto_clear`` produces one, and the adapter enforces that
        independently — see :func:`fcca.i2p.posting._require_auto_clear`. The
        belt and braces are deliberate: this is the single place in the system
        where something happens to an invoice that no person will look at, so
        the condition is checked by the caller that knows the tier *and* by the
        adapter that builds the payload.

        Claiming the key here rather than at dispatch is what makes the control
        work at all, because there is no dispatch. The key is claimed the moment
        the system decides it would post, which is the moment a second sighting
        of the same document becomes a duplicate.
        """
        final = routing or result.routing
        if final.tier != "auto_clear":
            return None
        invoice = self.repository.invoice(result.invoice_id)
        try:
            payload = self.target.build(invoice, result, result.provenance)
        except PostingBlocked:
            logger.exception("posting payload refused for %s", result.invoice_id)
            return None
        claimed = self.ledger.record(payload)
        self.trace.step(
            case_id=result.invoice_id,
            step_name="posting_payload",
            actor="rule",
            rule_id="I2P-S-14",
            inputs={"posting_key": payload.posting_key, "target": payload.target},
            outcome="built" if claimed else "key_already_claimed",
            summary=(
                f"Built a {payload.target} payload for {payload.line_count} line(s); "
                f"key {payload.posting_key} claimed. Dry run — nothing dispatched."
                if claimed
                else (
                    f"Key {payload.posting_key} was already claimed; no second payload "
                    "recorded. Nothing dispatched."
                )
            ),
            detail={
                "posting_key": payload.posting_key,
                "target": payload.target,
                "service": payload.service,
                "dry_run": payload.dry_run,
                "newly_claimed": claimed,
                "transformed_fields": [m.target_field for m in payload.mapping if m.transformed],
            },
        )
        return payload


def _first_clause(reason: str, limit: int = 180) -> str:
    """Trim a reason to something that fits one scanned line.

    The full text is always in the record's ``detail``, so nothing is lost —
    but a summary long enough to wrap is a summary nobody reads, and the trace
    schema caps it at 300 characters for that reason.
    """
    if len(reason) <= limit:
        return reason
    return reason[: limit - 1].rsplit(" ", 1)[0] + "…"


def _severity_of(result: InvoiceResult) -> Literal["low", "medium", "high"] | None:
    primary = result.primary_exception
    match = next((f for f in result.findings if f.exception_type == primary), None)
    return match.severity if match else None


def _empty_grounding() -> GroundingReport:
    return GroundingReport(total_citations=0, grounded_citations=0, ungrounded_citations=[])


def _null_run(settings: Settings) -> RunMetadata:
    """Run metadata for a case where no valid model output was obtained.

    Recorded rather than omitted: an audit trail that simply lacks a record for
    a failed call cannot be distinguished from one where the call never happened.
    """
    return RunMetadata(
        provider=settings.llm_provider,
        model=settings.model_name_for(),
        latency_ms=0,
        prompt_sha256="",
        parse_attempts=0,
    )


__all__ = ["InvoiceResolver", "ResolvedInvoice"]
