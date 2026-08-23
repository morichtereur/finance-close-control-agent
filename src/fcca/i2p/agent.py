"""The agent layer.

A language model is called here and nowhere else in the invoice-to-pay module,
and it is called under three restrictions that are enforced by code rather than
by instruction.

**Only on exceptions.** :meth:`ExceptionAgent.assess` refuses a clean result. A
clean invoice never reaches a model, which is not a cost optimisation — it is
what makes the touchless rate a property of the deterministic layer rather than
of a model's mood.

**Only for three things.** Classify the exception into a closed vocabulary,
propose a resolution from a closed vocabulary, and cite the specific fields it
used as evidence. It is not asked to compute anything, check anything, or decide
whether a number exceeds a threshold. Every number it sees has already been
computed, compared and thresholded.

**Only as validated JSON.** :class:`ExceptionAssessment` is the entire return
path. There is no free-text field a downstream step reads, no unstructured
reasoning that reaches a decision, and a response that does not validate is
retried once and then fails the case safely into escalation.

The evidence citations are grounded the same way the close module grounds policy
citations: a field the model names is checked against the fields it was actually
given, and an ungrounded citation is stripped and recorded. A model that cites a
field which does not exist has told you something useful about its answer.
"""

from __future__ import annotations

import hashlib
import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fcca.i2p.models import (
    ExceptionType,
    Invoice,
    InvoiceResult,
    ResolutionAction,
)
from fcca.i2p.prompts import PROMPT_VERSION, build_assessment_messages, evidence_fields
from fcca.shared.config import ProviderName, Settings, get_settings
from fcca.shared.errors import FCCAError
from fcca.shared.models import GroundingReport, RunMetadata
from fcca.shared.providers.base import ProviderSpec, describe_provider, get_llm

logger = logging.getLogger(__name__)


class FieldCitation(BaseModel):
    """A field the model says it used.

    Deliberately narrow: a path and nothing else. The model may not supply the
    value, because a value it supplied could disagree with the document and
    there would be no way to tell which was right. Values come from the
    deterministic layer; the model may only point at them.
    """

    model_config = ConfigDict(frozen=True)

    field_path: str = Field(description="Dotted path, e.g. 'line[1].price.residual_pct'.")

    @field_validator("field_path")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class ExceptionAssessment(BaseModel):
    """The model's contract. Nothing else it produces is read."""

    invoice_id: str
    classification: ExceptionType
    proposed_action: ResolutionAction
    rationale: str = Field(
        max_length=1200,
        description="Why the cited evidence implies this classification and action.",
    )
    evidence: list[FieldCitation] = Field(
        default_factory=list, description="The fields the conclusion rests on."
    )
    confidence: float = Field(ge=0.0, le=1.0)

    #: The one place a free-text value is permitted to influence anything, and it
    #: is constrained: a proposed cost centre must be one that exists, which is
    #: checked below rather than trusted.
    proposed_cost_center: str | None = Field(
        default=None,
        description="Cost centre extracted from free text, where that is the exception.",
    )


class AssessmentOutcome(BaseModel):
    """One model call, its grounding result and what it cost."""

    assessment: ExceptionAssessment
    grounding: GroundingReport
    run: RunMetadata
    raw_output: str


class ExceptionAgent:
    """Assesses one flagged invoice."""

    def __init__(
        self,
        llm: BaseChatModel,
        spec: ProviderSpec,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.spec = spec
        self.settings = settings or get_settings()

    @classmethod
    def build(
        cls,
        provider: ProviderName | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> ExceptionAgent:
        settings = settings or get_settings()
        return cls(
            llm=get_llm(provider, model_name, settings),
            spec=describe_provider(provider, model_name, settings),
            settings=settings,
        )

    # ------------------------------------------------------------------ assess
    def assess(
        self, invoice: Invoice, result: InvoiceResult, valid_cost_centers: list[str]
    ) -> AssessmentOutcome:
        """Classify and propose a resolution for one flagged invoice.

        Raises if the invoice is not an exception. That is a programming error,
        not a runtime condition: a clean invoice reaching a model would mean the
        pipeline had been rewired, and failing loudly is the point.
        """
        if not result.is_exception:
            raise FCCAError(
                f"{result.invoice_id} raised no finding; the agent layer is only ever "
                "called on exceptions"
            )

        messages = build_assessment_messages(invoice, result, valid_cost_centers, self.settings)
        prompt_hash = hashlib.sha256(
            "\n".join(str(m.content) for m in messages).encode("utf-8")
        ).hexdigest()[:32]

        from fcca.close.workflow.structured import invoke_structured

        invocation = invoke_structured(self.llm, messages, ExceptionAssessment, self.settings)
        assessment = invocation.value
        assert isinstance(assessment, ExceptionAssessment)

        # The pipeline owns the invoice id; the model only echoes it.
        if assessment.invoice_id != result.invoice_id:
            assessment = assessment.model_copy(update={"invoice_id": result.invoice_id})

        assessment, grounding = ground_evidence(assessment, invoice, result)
        assessment = _reject_invented_cost_center(assessment, valid_cost_centers)

        return AssessmentOutcome(
            assessment=assessment,
            grounding=grounding,
            run=RunMetadata(
                provider=self.spec.provider,
                model=self.spec.model,
                latency_ms=invocation.latency_ms,
                prompt_sha256=prompt_hash,
                input_tokens=invocation.input_tokens,
                output_tokens=invocation.output_tokens,
                parse_attempts=invocation.attempts,
                structured_output_mode=self.settings.structured_output_mode,
            ),
            raw_output=invocation.raw_text,
        )


def ground_evidence(
    assessment: ExceptionAssessment, invoice: Invoice, result: InvoiceResult
) -> tuple[ExceptionAssessment, GroundingReport]:
    """Strip evidence citations that name a field the model was not given.

    The close module grounds policy citations against retrieved passages; this is
    the same control over a different corpus. A citation that does not resolve is
    removed from the assessment and recorded, because a conclusion resting on a
    field that does not exist is a conclusion resting on nothing.
    """
    available = set(evidence_fields(invoice, result))
    grounded: list[FieldCitation] = []
    ungrounded: list[str] = []
    for citation in assessment.evidence:
        if citation.field_path in available:
            grounded.append(citation)
        else:
            ungrounded.append(citation.field_path)

    report = GroundingReport(
        total_citations=len(assessment.evidence),
        grounded_citations=len(grounded),
        ungrounded_citations=ungrounded,
    )
    return assessment.model_copy(update={"evidence": grounded}), report


def _reject_invented_cost_center(
    assessment: ExceptionAssessment, valid_cost_centers: list[str]
) -> ExceptionAssessment:
    """A proposed cost centre must be one that exists.

    The model is given the list and told to choose from it. This checks rather
    than trusts, because the failure mode — a plausible-looking code that is not
    in the master — produces a posting to an account nobody owns, and it would
    pass every downstream validation that only checks the format.
    """
    proposed = assessment.proposed_cost_center
    if proposed is None or proposed in valid_cost_centers:
        return assessment
    logger.warning(
        "discarding proposed cost centre %r for %s: not in the cost centre master",
        proposed,
        assessment.invoice_id,
    )
    return assessment.model_copy(
        update={
            "proposed_cost_center": None,
            # A proposal the system had to discard is not a confident one.
            "confidence": min(assessment.confidence, 0.4),
        }
    )


def failed_assessment(invoice_id: str, error: str) -> ExceptionAssessment:
    """The assessment used when the model could not produce a valid one.

    Confidence zero and an escalation action, so that a failure routes the same
    way an uncertain answer does. A case the automation could not assess is a
    case a person still has to look at.
    """
    return ExceptionAssessment(
        invoice_id=invoice_id,
        classification="no_exception",
        proposed_action="escalate_to_ap_manager",
        rationale=f"Automated assessment failed and was not completed: {error}",
        evidence=[],
        confidence=0.0,
    )


__all__ = [
    "PROMPT_VERSION",
    "AssessmentOutcome",
    "ExceptionAgent",
    "ExceptionAssessment",
    "FieldCitation",
    "failed_assessment",
    "ground_evidence",
]
