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
from fcca.i2p.models import InvoiceResult
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
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or I2PRepository(self.settings)
        self.trace = trace or TraceWriter(self.settings.i2p_trace_path, module="i2p")
        self.engine = InvoiceEngine(self.repository, self.settings, trace=self.trace)
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

        if not result.is_exception:
            # No model call. Not an optimisation — the point.
            return ResolvedInvoice(
                result=result,
                routing=result.routing,
                assessment=None,
                model_called=False,
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
            resolved_at=datetime.now(UTC),
        )


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
