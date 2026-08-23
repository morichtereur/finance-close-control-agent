"""Contracts that both process modules share.

A close exception and an invoice exception are different domain objects, but the
things the *instrumentation* says about them are the same: how risky it is, how
confident the model was, whether a person has to look at it, what a reviewer
subsequently did. Those types live here so that a record produced by either
module reads the same way to an auditor — and so that the routing layer can be
written once against one vocabulary.

Nothing in this module knows what a journal entry or an invoice is.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high"]
Severity = Literal["info", "warning", "critical"]

#: The actor responsible for one step of a pipeline. The distinction is the
#: whole point of the trace: a reviewer must be able to see at a glance which
#: conclusions were computed and which were inferred.
Actor = Literal["rule", "model", "human"]

#: Where a case ends up. ``auto_clear`` means the recommendation was applied
#: without a second pair of eyes — in this repository that is a *simulated*
#: posting, never a write to a real ledger. See :mod:`fcca.shared.routing`.
RoutingTier = Literal["auto_clear", "propose_and_approve", "escalate"]


class GateOutcome(BaseModel):
    """Result of a deterministic human-in-the-loop gate.

    The gate, not the model, decides whether a recommendation may be applied
    without a person looking at it.
    """

    model_config = ConfigDict(frozen=True)

    requires_human_review: bool
    disposition: Literal["auto_recommendation", "human_review"]
    reasons: list[str] = Field(
        default_factory=list, description="Why review was required; empty if auto-approved."
    )


class GroundingReport(BaseModel):
    """Whether the model's citations are supported by the evidence it was given.

    The close module grounds citations against retrieved policy sections; the
    I2P module grounds them against fields that exist on the invoice. The shape
    of the answer is the same either way.
    """

    model_config = ConfigDict(frozen=True)

    total_citations: int
    grounded_citations: int
    ungrounded_citations: list[str] = Field(default_factory=list)

    @property
    def is_fully_grounded(self) -> bool:
        return not self.ungrounded_citations


class RunMetadata(BaseModel):
    """Everything needed to reproduce and cost a single model invocation."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    latency_ms: int
    prompt_sha256: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    parse_attempts: int = 1
    structured_output_mode: str = "json_schema_prompt"


class ReviewRecord(BaseModel):
    """A human reviewer's disposition of a case.

    The system produces recommendations; only a person accepts, rejects or
    escalates them. This record closes the loop and is appended to the audit log.
    """

    model_config = ConfigDict(populate_by_name=True)

    # The close module calls its cases "exceptions" and the I2P module calls
    # them "invoices". ``case_id`` is the neutral name; the alias keeps the
    # close module's own vocabulary working where it reads better.
    case_id: str = Field(validation_alias=AliasChoices("case_id", "exception_id"))
    reviewer: str
    action: Literal["approved", "rejected", "escalated"]
    comment: str = ""
    reviewed_at: datetime

    @property
    def exception_id(self) -> str:
        return self.case_id


__all__ = [
    "Actor",
    "GateOutcome",
    "GroundingReport",
    "ReviewRecord",
    "RiskLevel",
    "RoutingTier",
    "RunMetadata",
    "Severity",
]
