"""Reconciliation and variance checks (CHK-14 … CHK-17)."""

from __future__ import annotations

from typing import Any

from fcca.analytics import VarianceResult
from fcca.config import Settings, get_settings
from fcca.models import ControlSignal, JournalEntry, Severity

#: Accounts that must clear to zero at month-end (Account Reconciliation Policy §5).
SUSPENSE_ACCOUNTS = ("199000",)

#: Ageing beyond which a reconciling item is escalated regardless of amount (§4.3).
AGEING_ESCALATION_DAYS = 60

#: Variance escalation trigger (Materiality and Escalation Policy §3.3).
VARIANCE_PERCENT_TRIGGER = 50.0
VARIANCE_ABSOLUTE_TRIGGER = 100_000.0


def check_reconciliation_status(
    entry: JournalEntry,
    record: dict[str, Any] | None,
    settings: Settings | None = None,
) -> ControlSignal:
    """CHK-14 — is the account reconciliation complete for the close period?"""
    status = str(record["status"]) if record else entry.reconciliation_status
    incomplete = status in {"open", "in_progress"}
    reviewer = str(record.get("reviewer") or "") if record else ""
    return ControlSignal(
        check_id="CHK-14",
        name="reconciliation_status",
        triggered=incomplete,
        severity="warning" if incomplete else "info",
        detail=(
            f"Reconciliation for account {entry.account} is '{status}'"
            + (" with no reviewer recorded." if incomplete and not reviewer else ".")
        ),
        observed_value=status,
        threshold="reconciled",
    )


def check_reconciliation_difference(
    entry: JournalEntry,
    record: dict[str, Any] | None,
    settings: Settings | None = None,
) -> ControlSignal:
    """CHK-15 — unexplained reconciling difference against policy thresholds."""
    settings = settings or get_settings()
    if record is None:
        return ControlSignal(
            check_id="CHK-15",
            name="reconciliation_difference",
            triggered=False,
            severity="info",
            detail="No reconciliation record exists for this account and period (P&L account).",
            observed_value=None,
            threshold=settings.trivial_threshold,
        )

    difference = abs(float(record.get("difference") or 0.0))
    days_open = int(record.get("days_open") or 0)
    above_investigation = difference >= settings.trivial_threshold
    material = difference >= settings.materiality_group
    aged = days_open > AGEING_ESCALATION_DAYS and difference > 0

    severity: Severity
    if material:
        detail = (
            "Unexplained difference at or above group materiality; Financial Controller "
            "review required."
        )
        severity = "critical"
    elif above_investigation:
        detail = (
            "Difference at or above the investigation threshold; must be explained before sign-off."
        )
        severity = "warning"
    elif aged:
        detail = f"Reconciling item open for {days_open} days; ageing indicates a process failure."
        severity = "warning"
    else:
        detail = "Difference below the investigation threshold."
        severity = "info"

    return ControlSignal(
        check_id="CHK-15",
        name="reconciliation_difference",
        triggered=above_investigation or aged,
        severity=severity,
        detail=detail,
        observed_value=round(difference, 2),
        threshold=settings.trivial_threshold,
    )


def check_suspense_cleared(
    entry: JournalEntry,
    record: dict[str, Any] | None,
    settings: Settings | None = None,
) -> ControlSignal:
    """CHK-16 — suspense and clearing accounts must be zero at month-end."""
    if entry.account not in SUSPENSE_ACCOUNTS:
        return ControlSignal(
            check_id="CHK-16",
            name="suspense_account_cleared",
            triggered=False,
            severity="info",
            detail="Not a suspense or clearing account.",
            observed_value=entry.account,
        )
    balance = abs(float(record.get("gl_balance") or 0.0)) if record else 0.0
    difference = abs(float(record.get("difference") or 0.0)) if record else 0.0
    residual = max(balance, difference)
    open_status = record is not None and str(record.get("status")) in {"open", "in_progress"}
    triggered = residual > 0.01 and open_status
    return ControlSignal(
        check_id="CHK-16",
        name="suspense_account_cleared",
        triggered=triggered,
        severity="critical" if triggered else "info",
        detail=(
            "Suspense and clearing account carries a residual balance at sign-off and must be "
            "escalated with an ageing analysis."
            if triggered
            else "Suspense account is cleared or reconciled."
        ),
        observed_value=round(residual, 2),
        threshold=0.0,
    )


def check_account_variance(
    entry: JournalEntry,
    variance: VarianceResult,
    settings: Settings | None = None,
) -> ControlSignal:
    """CHK-17 — month-on-month movement above the variance escalation trigger."""
    percent = variance.percent_change
    absolute = abs(variance.absolute_change)
    triggered = (
        percent is not None
        and abs(percent) > VARIANCE_PERCENT_TRIGGER
        and absolute > VARIANCE_ABSOLUTE_TRIGGER
    )
    percent_text = f"{percent:.1f}%" if percent is not None else "n/a (no prior balance)"
    return ControlSignal(
        check_id="CHK-17",
        name="account_variance",
        triggered=triggered,
        severity="warning" if triggered else "info",
        detail=(
            f"Account movement of {absolute:,.0f} reporting currency ({percent_text}) versus the "
            "prior-period average"
            + (" exceeds the variance escalation trigger." if triggered else ".")
        ),
        observed_value=round(absolute, 2),
        threshold=VARIANCE_ABSOLUTE_TRIGGER,
    )
