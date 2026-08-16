"""Materiality assessment.

Thresholds come from :class:`~fcca.config.Settings` and mirror the Materiality
and Escalation Policy. Changing the policy document alone does not change
behaviour — that is deliberate, and the architecture notes explain the trade-off.
"""

from __future__ import annotations

from typing import Literal

from fcca.config import Settings, get_settings
from fcca.masterdata import FX_TO_EUR
from fcca.models import ControlSignal, JournalEntry

MaterialityBand = Literal["trivial", "below_approval", "above_approval", "material"]


def amount_in_reporting_currency(entry: JournalEntry) -> float:
    """Absolute entry amount converted to the group reporting currency.

    The generator already stores a converted amount; recomputing it here keeps the
    control independent of a field that an upstream system could get wrong.
    """
    rate = FX_TO_EUR.get(entry.currency)
    if rate is None:
        return abs(entry.amount_reporting_ccy)
    return abs(entry.amount) * rate


def materiality_band(amount_reporting: float, settings: Settings | None = None) -> MaterialityBand:
    """Classify an absolute reporting-currency amount into a policy band."""
    settings = settings or get_settings()
    if amount_reporting >= settings.materiality_group:
        return "material"
    if amount_reporting >= settings.journal_approval_threshold:
        return "above_approval"
    if amount_reporting < settings.trivial_threshold:
        return "trivial"
    return "below_approval"


def check_materiality(entry: JournalEntry, settings: Settings | None = None) -> ControlSignal:
    """CHK-13 — position the entry against group materiality."""
    settings = settings or get_settings()
    amount = amount_in_reporting_currency(entry)
    band = materiality_band(amount, settings)
    detail = {
        "material": "At or above group materiality; reportable to Group Accounting.",
        "above_approval": "At or above the second-level approval threshold.",
        "below_approval": "Below the approval threshold but not clearly trivial.",
        "trivial": "Below the clearly trivial threshold.",
    }[band]
    return ControlSignal(
        check_id="CHK-13",
        name="materiality_assessment",
        triggered=band == "material",
        severity="critical" if band == "material" else "info",
        detail=f"{detail} Band: {band}.",
        observed_value=round(amount, 2),
        threshold=settings.materiality_group,
    )
