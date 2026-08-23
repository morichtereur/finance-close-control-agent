"""Control engine: runs every deterministic check for one exception.

The engine is the boundary between the ERP world and the reasoning layer. It
turns a row in a ledger into a list of typed, thresholded, explainable signals.
The model downstream never sees raw amounts without the control context that a
finance reviewer would apply to them.
"""

from __future__ import annotations

from typing import Any

from fcca.close.analytics import CloseAnalytics
from fcca.close.controls import journal_checks as jc
from fcca.close.controls import reconciliation as rc
from fcca.close.controls.materiality import check_materiality
from fcca.close.models import CloseException, ControlSignal, JournalEntry
from fcca.shared.config import Settings, get_settings


def to_journal_entry(row: dict[str, Any]) -> JournalEntry:
    """Convert a database row into a validated :class:`JournalEntry`."""
    data = dict(row)
    for key in ("supporting_document", "approved_by"):
        if not data.get(key):
            data[key] = None
    return JournalEntry.model_validate(data)


def to_close_exception(row: dict[str, Any]) -> CloseException:
    """Convert a database row into a validated :class:`CloseException`."""
    return CloseException.model_validate(dict(row))


def run_controls(
    entry: JournalEntry,
    exception: CloseException,
    analytics: CloseAnalytics,
    settings: Settings | None = None,
) -> list[ControlSignal]:
    """Run the full deterministic control set for one exception.

    Population-level facts (duplicates, prior usage, variance, reconciliation
    status) are read from DuckDB; the checks themselves stay pure functions.
    """
    settings = settings or get_settings()
    period = exception.close_period

    duplicates = analytics.duplicate_candidates(entry.journal_id)
    combination_uses = analytics.combination_history(
        entry.company_code, entry.account, entry.cost_center, period
    )
    user_history = analytics.user_account_history(
        entry.company_code, entry.account, entry.user_id, period
    )
    same_day_total = analytics.same_day_aggregate(entry.journal_id)
    variance = analytics.account_variance(entry.company_code, entry.account, period)
    recon = analytics.reconciliation(entry.company_code, entry.account, period)

    return [
        jc.check_supporting_document(entry, settings),
        jc.check_approval_threshold(entry, settings),
        jc.check_segregation_of_duties(entry, settings),
        jc.check_posting_timeliness(entry, settings),
        jc.check_business_hours(entry, settings),
        jc.check_high_risk_account(entry, settings),
        jc.check_duplicate_posting(entry, duplicates, settings),
        jc.check_round_amount(entry, settings),
        jc.check_narrative_quality(entry, settings),
        jc.check_period_integrity(entry, settings),
        jc.check_account_cost_center_combination(entry, combination_uses, settings),
        jc.check_preparer_familiarity(entry, user_history, settings),
        check_materiality(entry, settings),
        rc.check_reconciliation_status(entry, recon, settings),
        rc.check_reconciliation_difference(entry, recon, settings),
        rc.check_suspense_cleared(entry, recon, settings),
        rc.check_account_variance(entry, variance, settings),
        jc.check_same_day_aggregation(entry, same_day_total, settings),
    ]


def mandatory_escalation_triggers(signals: list[ControlSignal]) -> list[str]:
    """Deterministic escalation triggers (Materiality and Escalation Policy §3.1).

    These are evaluated *before and independently of* the model. If any fires,
    the case goes to a human whatever the model concluded and however confident
    it was. This is the mechanism that keeps model quality out of the control
    path for the cases that matter most.
    """
    by_id = {s.check_id: s for s in signals}
    reasons: list[str] = []

    for signal in signals:
        if signal.triggered and signal.severity == "critical":
            reasons.append(f"{signal.check_id} {signal.name}: {signal.detail}")

    high_risk = by_id.get("CHK-06")
    approval = by_id.get("CHK-02")
    if (
        high_risk is not None
        and approval is not None
        and high_risk.triggered
        and approval.triggered
    ):
        reasons.append(
            "CHK-06+CHK-02: posting to a high-risk account without documented second-level approval"
        )

    return reasons
