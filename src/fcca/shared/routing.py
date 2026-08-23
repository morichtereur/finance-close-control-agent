"""Three-tier routing, shared by both process modules.

Every case ends in one of three places:

``auto_clear``
    The recommendation stands without a second pair of eyes. In this repository
    that means a *simulated* posting and nothing more — see the warning below.

``propose_and_approve``
    The system states what it would do and a named person approves or rejects
    it. This is where most exceptions belong: the work of finding and explaining
    the problem is done, and the decision remains a person's.

``escalate``
    A person has to investigate, not merely approve. Reserved for cases where
    the right answer is not yet known, or where being wrong is expensive enough
    that a confirmation click is not enough of a control.

**Nothing here posts to a ledger or an ERP.** ``auto_clear`` is the name of a
decision, not an action: no code path in this repository writes to a financial
system, and the README says so plainly. The tier is recorded, the trace shows
why, and a real deployment would have to build the posting step — along with
everything in the close module's architecture note about what would need to
change first.

**The tier is a function of four inputs**, in this order of authority:

1. **The deterministic outcome.** Which rules fired, and how severe they are.
2. **Document value.** A clean match on a small invoice is a different risk from
   a clean match on a large one, and no amount of model confidence changes that.
3. **Exception type.** Some kinds of problem are never suitable for automation
   regardless of the other three inputs.
4. **Model confidence**, where a model was involved at all. It is the *last*
   input, and it can only ever move a case toward more scrutiny, never less.

That ordering is the whole design. A model's opinion is an input to a rule; it
is never the rule.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fcca.shared.models import RoutingTier

#: Exception types that always escalate, whatever else is true.
#:
#: Bank-detail changes are here as a fraud control, and the reasoning is worth
#: stating because it is not a statement about model quality.
#:
#: A payment-redirection fraud is *designed* to look like a clean invoice. The
#: goods were really delivered, the price is really right, the purchase order
#: really exists — the attacker's whole objective is that every other check
#: passes. So the signals a confidence score is built from are exactly the
#: signals the attack arranges to look normal, and a model that is 99% sure the
#: invoice is fine is not evidence that the account is right. It is evidence
#: that the fraud is working.
#:
#: The control that actually catches this is a person telephoning the vendor on
#: a number held independently of the invoice. That is not something software can
#: do or verify, so the only defensible routing is to put it in front of someone
#: who can — every time, with no confidence threshold that bypasses it and no
#: value floor beneath which it is skipped. A hardcoded rule rather than a
#: configurable one, because a configurable fraud control is one deployment
#: mistake away from being switched off.
ALWAYS_ESCALATE: frozenset[str] = frozenset({"bank_details_mismatch"})


class RoutingDecision(BaseModel):
    """Where a case goes, and every reason it went there."""

    model_config = ConfigDict(frozen=True)

    tier: RoutingTier
    reasons: list[str] = Field(
        description="Every condition that applied, in the order they were evaluated."
    )
    deciding_reason: str = Field(description="The condition that set the final tier.")
    model_confidence: float | None = Field(
        default=None, description="None where no model was consulted."
    )
    document_value: float
    exception_type: str

    @property
    def requires_human(self) -> bool:
        return self.tier != "auto_clear"


def route(
    *,
    exception_type: str,
    is_exception: bool,
    document_value: float,
    auto_clear_max_value: float,
    propose_max_value: float,
    auto_clear_min_confidence: float,
    model_confidence: float | None = None,
    severity: Literal["low", "medium", "high"] | None = None,
    extra_escalations: list[str] | None = None,
) -> RoutingDecision:
    """Decide the tier.

    Evaluated as a sequence of conditions rather than a lookup table so that the
    reasons are accumulated as it goes: a reviewer is owed every condition that
    applied, not only the one that won.
    """
    reasons: list[str] = []

    # ---------------------------------------------------------- hard controls
    if exception_type in ALWAYS_ESCALATE:
        reason = (
            f"Exception type {exception_type!r} always escalates as a fraud control, "
            "regardless of model confidence or document value: verification requires "
            "contacting the vendor on independently held details, which no automated "
            "check can perform."
        )
        return RoutingDecision(
            tier="escalate",
            reasons=[reason],
            deciding_reason=reason,
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    for escalation in extra_escalations or []:
        reasons.append(escalation)
    if reasons:
        return RoutingDecision(
            tier="escalate",
            reasons=reasons,
            deciding_reason=reasons[0],
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    # ------------------------------------------------------------ clean cases
    if not is_exception:
        if document_value >= auto_clear_max_value:
            reason = (
                f"Clean match, but document value {document_value:,.2f} is at or above the "
                f"auto-clear limit of {auto_clear_max_value:,.2f}."
            )
            reasons.append(reason)
            return RoutingDecision(
                tier="propose_and_approve",
                reasons=reasons,
                deciding_reason=reason,
                model_confidence=model_confidence,
                document_value=document_value,
                exception_type=exception_type,
            )
        reason = (
            f"No rule fired and document value {document_value:,.2f} is below the "
            f"auto-clear limit of {auto_clear_max_value:,.2f}."
        )
        return RoutingDecision(
            tier="auto_clear",
            reasons=[reason],
            deciding_reason=reason,
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    # --------------------------------------------------------- exception cases
    # An exception is never auto-cleared. The most it can be is a proposal.
    if severity == "high":
        reason = f"Exception {exception_type!r} is rated high severity."
        reasons.append(reason)
        return RoutingDecision(
            tier="escalate",
            reasons=reasons,
            deciding_reason=reason,
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    if document_value >= propose_max_value:
        reason = (
            f"Document value {document_value:,.2f} is at or above the proposal limit of "
            f"{propose_max_value:,.2f}; an exception this large is investigated, not approved."
        )
        reasons.append(reason)
        return RoutingDecision(
            tier="escalate",
            reasons=reasons,
            deciding_reason=reason,
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    if model_confidence is not None and model_confidence < auto_clear_min_confidence:
        reason = (
            f"Model confidence {model_confidence:.2f} is below {auto_clear_min_confidence:.2f}; "
            "the proposed resolution is not reliable enough to put in front of an approver "
            "as a recommendation."
        )
        reasons.append(reason)
        return RoutingDecision(
            tier="escalate",
            reasons=reasons,
            deciding_reason=reason,
            model_confidence=model_confidence,
            document_value=document_value,
            exception_type=exception_type,
        )

    reason = (
        f"Exception {exception_type!r} at {document_value:,.2f} with a resolution the system "
        "can state; a named person approves or rejects it."
    )
    reasons.append(reason)
    return RoutingDecision(
        tier="propose_and_approve",
        reasons=reasons,
        deciding_reason=reason,
        model_confidence=model_confidence,
        document_value=document_value,
        exception_type=exception_type,
    )


__all__ = ["ALWAYS_ESCALATE", "RoutingDecision", "route"]
