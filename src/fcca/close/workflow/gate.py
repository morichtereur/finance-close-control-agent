"""The human-in-the-loop gate.

The gate is deterministic and it always wins. A model that is certain it has
found nothing cannot clear an item that the control layer flagged for mandatory
escalation, and no confidence value can override that.

Auto-recommendation is permitted only when *all* of the following hold:

* no deterministic mandatory-escalation trigger fired;
* the risk rating is not ``high``;
* confidence is at or above the configured threshold;
* enough grounded policy evidence exists;
* every citation the model made is grounded;
* the recommended action is not one with external consequence.

Anything else becomes an explicit ``human_review`` state with the reasons
recorded. "Auto" here still means *a recommendation applied without a second
pair of eyes*, never a posting to the ledger — no path in this system writes to
an ERP.
"""

from __future__ import annotations

from fcca.close.controls.engine import mandatory_escalation_triggers
from fcca.close.models import ControlDecision, ControlSignal, GateOutcome, GroundingReport
from fcca.shared.config import Settings, get_settings

#: Actions that always involve a person, whatever the model's confidence.
CONSEQUENTIAL_ACTIONS = frozenset(
    {
        "escalate_to_financial_controller",
        "propose_correcting_entry",
        "refer_to_internal_audit",
    }
)


def apply_gate(
    decision: ControlDecision,
    grounding: GroundingReport,
    signals: list[ControlSignal],
    settings: Settings | None = None,
) -> GateOutcome:
    """Decide whether this case may proceed as an automatic recommendation."""
    settings = settings or get_settings()
    reasons: list[str] = []

    triggers = mandatory_escalation_triggers(signals)
    if triggers:
        reasons.append(
            "Deterministic mandatory escalation trigger(s): "
            + "; ".join(t.split(":")[0] for t in triggers)
        )

    if decision.risk_level == "high":
        reasons.append("Risk rated high; policy reserves high-rated items for human review.")

    if decision.confidence < settings.auto_approve_min_confidence:
        reasons.append(
            f"Model confidence {decision.confidence:.2f} is below the auto-approval "
            f"threshold {settings.auto_approve_min_confidence:.2f}."
        )

    if grounding.grounded_citations < settings.auto_approve_min_evidence:
        reasons.append(
            f"Only {grounding.grounded_citations} grounded policy citation(s); "
            f"{settings.auto_approve_min_evidence} required for auto-approval."
        )

    if grounding.ungrounded_citations:
        reasons.append(
            "Model cited policy that was not retrieved: "
            + ", ".join(grounding.ungrounded_citations)
        )

    if decision.action_category in CONSEQUENTIAL_ACTIONS:
        reasons.append(
            f"Recommended action '{decision.action_category}' requires a named person to act."
        )

    if decision.requires_human_review and not reasons:
        reasons.append("Model requested human review.")

    requires_review = bool(reasons)
    return GateOutcome(
        requires_human_review=requires_review,
        disposition="human_review" if requires_review else "auto_recommendation",
        reasons=reasons,
    )


def failed_case_gate(reason: str) -> GateOutcome:
    """Gate outcome for a case that could not be decided at all.

    A failure is never a pass. If the model could not produce a valid decision,
    the item goes to a person with the failure recorded.
    """
    return GateOutcome(
        requires_human_review=True,
        disposition="human_review",
        reasons=[f"Automated assessment failed and was not completed: {reason}"],
    )
