"""Journal-entry level control checks (CHK-01 … CHK-12).

Each function takes only what it needs: the entry, configuration, and — where a
check requires population context — a fact already computed by
:mod:`fcca.close.analytics`. This keeps the checks unit-testable without a database.
"""

from __future__ import annotations

from typing import Any

from fcca.close.controls.materiality import amount_in_reporting_currency
from fcca.close.models import ControlSignal, JournalEntry, Severity
from fcca.shared.config import Settings, get_settings

#: Narratives that do not allow an independent reviewer to understand the event
#: (Journal Entry Policy §3).
WEAK_NARRATIVE_MARKERS = (
    "adjustment",
    "correction per instruction",
    "as discussed",
    "reclass",
    "manual posting",
    "per instruction",
)


def check_supporting_document(
    entry: JournalEntry, settings: Settings | None = None
) -> ControlSignal:
    """CHK-01 — is the posting supported by a resolvable document reference?"""
    settings = settings or get_settings()
    amount = amount_in_reporting_currency(entry)
    missing = entry.manual_posting and not entry.has_supporting_document
    severity: Severity
    if missing:
        above_trivial = amount >= settings.trivial_threshold
        severity = "critical" if above_trivial else "warning"
        detail = (
            "Manual posting carries no supporting document reference; "
            f"amount is {'at or above' if above_trivial else 'below'} the clearly trivial threshold."
        )
    else:
        severity = "info"
        detail = (
            "Supporting document reference present."
            if entry.has_supporting_document
            else "System-generated posting; support is held in the source sub-ledger."
        )
    return ControlSignal(
        check_id="CHK-01",
        name="supporting_documentation",
        triggered=missing,
        severity=severity,
        detail=detail,
        observed_value=entry.supporting_document or "none",
        threshold=settings.trivial_threshold,
    )


def check_approval_threshold(
    entry: JournalEntry, settings: Settings | None = None
) -> ControlSignal:
    """CHK-02 — second-level approval where the amount requires it."""
    settings = settings or get_settings()
    amount = amount_in_reporting_currency(entry)
    needs_approval = amount >= settings.journal_approval_threshold
    breached = needs_approval and not entry.approved_by
    return ControlSignal(
        check_id="CHK-02",
        name="approval_threshold",
        triggered=breached,
        severity="critical" if breached else "info",
        detail=(
            "Amount requires documented second-level approval but no approver is recorded."
            if breached
            else (
                f"Approved by {entry.approved_by}."
                if entry.approved_by
                else "Below the second-level approval threshold."
            )
        ),
        observed_value=round(amount, 2),
        threshold=settings.journal_approval_threshold,
    )


def check_segregation_of_duties(
    entry: JournalEntry, settings: Settings | None = None
) -> ControlSignal:
    """CHK-03 — preparer and approver must be different people."""
    breached = bool(entry.approved_by) and entry.approved_by == entry.user_id
    return ControlSignal(
        check_id="CHK-03",
        name="segregation_of_duties",
        triggered=breached,
        severity="critical" if breached else "info",
        detail=(
            "Entry was prepared and approved by the same user id."
            if breached
            else "Preparer and approver differ, or no approval was required."
        ),
        observed_value=f"preparer={entry.user_id}, approver={entry.approved_by or 'none'}",
    )


def check_posting_timeliness(
    entry: JournalEntry, settings: Settings | None = None
) -> ControlSignal:
    """CHK-04 — days between document date and posting date."""
    settings = settings or get_settings()
    days = entry.days_to_post
    late = entry.manual_posting and days > settings.late_posting_days
    return ControlSignal(
        check_id="CHK-04",
        name="posting_timeliness",
        triggered=late,
        severity="warning" if late else "info",
        detail=(
            f"Manual posting made {days} calendar days after the document date."
            if late
            else f"Posted {days} calendar days after the document date."
        ),
        observed_value=days,
        threshold=settings.late_posting_days,
    )


def check_business_hours(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-05 — posting outside normal business hours or at a weekend."""
    settings = settings or get_settings()
    hour = entry.posting_timestamp.hour
    weekend = entry.posting_timestamp.weekday() >= 5
    outside = hour < settings.business_hours_start or hour >= settings.business_hours_end
    triggered = entry.manual_posting and (outside or weekend)
    return ControlSignal(
        check_id="CHK-05",
        name="posting_window",
        triggered=triggered,
        severity="warning" if triggered else "info",
        detail=(
            "Manual posting created outside normal business hours"
            + (" and at a weekend." if weekend else ".")
            if triggered
            else "Posted within the normal posting window."
        ),
        observed_value=entry.posting_timestamp.strftime("%a %H:%M"),
        threshold=f"{settings.business_hours_start:02d}:00-{settings.business_hours_end:02d}:00 Mon-Fri",
    )


def check_high_risk_account(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-06 — account designated high risk by the control catalogue."""
    settings = settings or get_settings()
    is_high_risk = entry.account in settings.high_risk_accounts
    # Journal Entry Policy §6 makes *manual* entries to these accounts a mandatory
    # human-review item; a system-generated posting is only an elevated indicator.
    severity: Severity
    if is_high_risk and entry.manual_posting:
        severity = "critical"
        detail = (
            f"Manual entry to account {entry.account} ({entry.account_name}), designated high "
            "risk; such entries are always reviewed by a person and may not be auto-approved."
        )
    elif is_high_risk:
        severity = "warning"
        detail = (
            f"System-generated posting to high-risk account {entry.account} ({entry.account_name})."
        )
    else:
        severity = "info"
        detail = f"Account {entry.account} is not on the high-risk list."
    return ControlSignal(
        check_id="CHK-06",
        name="account_risk_rating",
        triggered=is_high_risk,
        severity=severity,
        detail=detail,
        observed_value=entry.account,
    )


def check_duplicate_posting(
    entry: JournalEntry,
    candidates: list[dict[str, Any]],
    settings: Settings | None = None,
) -> ControlSignal:
    """CHK-07 — another posting with identical entity, account, amount and document date."""
    found = bool(candidates)
    ids = ", ".join(str(c.get("journal_id")) for c in candidates[:3])
    return ControlSignal(
        check_id="CHK-07",
        name="duplicate_posting",
        triggered=found,
        severity="critical" if found else "info",
        detail=(
            f"{len(candidates)} matching posting(s) found: {ids}."
            if found
            else "No matching posting found on entity, account, amount and document date."
        ),
        observed_value=len(candidates),
        threshold=0,
    )


def check_round_amount(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-08 — large round amounts are an indicator of estimated or arbitrary postings."""
    settings = settings or get_settings()
    amount = amount_in_reporting_currency(entry)
    is_round = amount >= 100_000 and abs(amount % 10_000) < 0.01
    return ControlSignal(
        check_id="CHK-08",
        name="round_amount",
        triggered=is_round,
        severity="warning" if is_round else "info",
        detail=(
            "Large round amount, consistent with an estimate rather than a transaction."
            if is_round
            else "Amount does not show a round-number pattern."
        ),
        observed_value=round(amount, 2),
        threshold=100_000,
    )


def check_narrative_quality(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-09 — the description must be self-explanatory (Journal Entry Policy §3)."""
    text = entry.description.strip().lower()
    weak = len(text) < 12 or any(text.startswith(m) or text == m for m in WEAK_NARRATIVE_MARKERS)
    return ControlSignal(
        check_id="CHK-09",
        name="narrative_quality",
        triggered=weak,
        severity="warning" if weak else "info",
        detail=(
            "Description does not allow an independent reviewer to understand the business event."
            if weak
            else "Description identifies the business event."
        ),
        observed_value=entry.description[:80],
    )


def check_period_integrity(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-10 — document date in an earlier period than the posting period."""
    doc_period = entry.document_date.strftime("%Y-%m")
    post_period = entry.posting_date.strftime("%Y-%m")
    breached = doc_period < post_period
    return ControlSignal(
        check_id="CHK-10",
        name="period_integrity",
        triggered=breached,
        severity="warning" if breached else "info",
        detail=(
            f"Document dated in {doc_period} but posted in {post_period}."
            if breached
            else "Document date and posting date fall in the same period."
        ),
        observed_value=f"{doc_period} -> {post_period}",
    )


def check_account_cost_center_combination(
    entry: JournalEntry, prior_uses: int, settings: Settings | None = None
) -> ControlSignal:
    """CHK-11 — combination not seen for this entity in prior periods."""
    unexpected = prior_uses == 0
    return ControlSignal(
        check_id="CHK-11",
        name="account_cost_center_combination",
        triggered=unexpected,
        severity="warning" if unexpected else "info",
        detail=(
            f"Account {entry.account} with cost center {entry.cost_center} was not used by "
            f"{entry.company_code} in prior periods of the dataset."
            if unexpected
            else f"Combination used {prior_uses} time(s) in prior periods."
        ),
        observed_value=prior_uses,
        threshold=0,
    )


def check_preparer_familiarity(
    entry: JournalEntry, prior_postings: int, settings: Settings | None = None
) -> ControlSignal:
    """CHK-12 — a user posting to an account they do not normally use."""
    settings = settings or get_settings()
    unfamiliar = prior_postings == 0 and entry.account in settings.high_risk_accounts
    return ControlSignal(
        check_id="CHK-12",
        name="preparer_account_familiarity",
        triggered=unfamiliar,
        severity="warning" if unfamiliar else "info",
        detail=(
            f"User {entry.user_id} has not posted to high-risk account {entry.account} before."
            if unfamiliar
            else f"User {entry.user_id} posted to this account {prior_postings} time(s) previously."
        ),
        observed_value=prior_postings,
        threshold=0,
    )


def check_same_day_aggregation(
    entry: JournalEntry, same_day_total: float, settings: Settings | None = None
) -> ControlSignal:
    """CHK-18 — anti-splitting rule (Journal Entry Policy §4.3)."""
    settings = settings or get_settings()
    own = amount_in_reporting_currency(entry)
    split_risk = (
        own < settings.journal_approval_threshold
        and same_day_total >= settings.journal_approval_threshold
    )
    return ControlSignal(
        check_id="CHK-18",
        name="same_day_aggregation",
        triggered=split_risk,
        severity="warning" if split_risk else "info",
        detail=(
            "Entry is individually below the approval threshold but the same account, cost "
            "center and posting date aggregate to at or above it."
            if split_risk
            else "Same-day aggregate does not indicate threshold splitting."
        ),
        observed_value=round(same_day_total, 2),
        threshold=settings.journal_approval_threshold,
    )
