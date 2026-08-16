"""Deterministic control checks.

These are the rules a finance reviewer would challenge first, so they are tested
at their boundaries rather than in the middle of their ranges.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from tests.conftest import make_entry

from fcca.analytics import VarianceResult
from fcca.config import Settings
from fcca.controls import journal_checks as jc
from fcca.controls.engine import mandatory_escalation_triggers
from fcca.controls.materiality import (
    amount_in_reporting_currency,
    check_materiality,
    materiality_band,
)
from fcca.controls.reconciliation import (
    check_account_variance,
    check_reconciliation_difference,
    check_reconciliation_status,
    check_suspense_cleared,
)


# --------------------------------------------------------------------- materiality
def test_materiality_bands_at_their_boundaries(settings: Settings) -> None:
    assert materiality_band(settings.trivial_threshold - 0.01, settings) == "trivial"
    assert materiality_band(settings.trivial_threshold, settings) == "below_approval"
    assert materiality_band(settings.journal_approval_threshold, settings) == "above_approval"
    assert materiality_band(settings.materiality_group, settings) == "material"


def test_materiality_check_triggers_only_at_group_materiality(settings: Settings) -> None:
    below = check_materiality(make_entry(amount=settings.materiality_group - 1), settings)
    at = check_materiality(make_entry(amount=settings.materiality_group), settings)
    assert not below.triggered
    assert at.triggered and at.severity == "critical"


def test_amount_is_converted_to_reporting_currency() -> None:
    entry = make_entry(currency="CHF", amount=-10_000.0, amount_reporting_ccy=-10_500.0)
    assert amount_in_reporting_currency(entry) == pytest.approx(10_500.0)


# ---------------------------------------------------------------- documentation
def test_unsupported_manual_entry_above_trivial_is_critical(settings: Settings) -> None:
    signal = jc.check_supporting_document(
        make_entry(manual_posting=True, supporting_document=None, amount=20_000.0), settings
    )
    assert signal.triggered and signal.severity == "critical"


def test_unsupported_manual_entry_below_trivial_is_only_a_warning(settings: Settings) -> None:
    signal = jc.check_supporting_document(
        make_entry(manual_posting=True, supporting_document=None, amount=900.0), settings
    )
    assert signal.triggered and signal.severity == "warning"


def test_system_generated_entry_without_reference_is_not_flagged(settings: Settings) -> None:
    signal = jc.check_supporting_document(
        make_entry(manual_posting=False, supporting_document=None), settings
    )
    assert not signal.triggered


# -------------------------------------------------------------------- approvals
def test_missing_approval_above_threshold_is_a_breach(settings: Settings) -> None:
    signal = jc.check_approval_threshold(
        make_entry(amount=settings.journal_approval_threshold, approved_by=None), settings
    )
    assert signal.triggered and signal.severity == "critical"


def test_approved_entry_above_threshold_passes(settings: Settings) -> None:
    signal = jc.check_approval_threshold(
        make_entry(amount=settings.journal_approval_threshold, approved_by="u.klein"), settings
    )
    assert not signal.triggered


def test_preparer_equal_to_approver_is_a_control_failure(settings: Settings) -> None:
    signal = jc.check_segregation_of_duties(
        make_entry(user_id="u.becker", approved_by="u.becker"), settings
    )
    assert signal.triggered and signal.severity == "critical"


# ------------------------------------------------------------------- timeliness
def test_late_manual_posting_is_flagged(settings: Settings) -> None:
    entry = make_entry(
        manual_posting=True, document_date=date(2026, 6, 20), posting_date=date(2026, 7, 14)
    )
    assert jc.check_posting_timeliness(entry, settings).triggered


def test_timely_posting_is_not_flagged(settings: Settings) -> None:
    entry = make_entry(document_date=date(2026, 7, 12), posting_date=date(2026, 7, 14))
    assert not jc.check_posting_timeliness(entry, settings).triggered


def test_system_postings_are_exempt_from_the_timeliness_rule(settings: Settings) -> None:
    entry = make_entry(
        manual_posting=False, document_date=date(2026, 6, 1), posting_date=date(2026, 7, 14)
    )
    assert not jc.check_posting_timeliness(entry, settings).triggered


def test_out_of_hours_manual_posting_is_an_indicator_not_a_breach(settings: Settings) -> None:
    signal = jc.check_business_hours(
        make_entry(posting_timestamp=datetime(2026, 7, 14, 2, 15)), settings
    )
    assert signal.triggered and signal.severity == "warning"


# ---------------------------------------------------------------- account risk
def test_manual_entry_to_high_risk_account_is_critical(settings: Settings) -> None:
    signal = jc.check_high_risk_account(make_entry(account="610000", manual_posting=True), settings)
    assert signal.triggered and signal.severity == "critical"


def test_system_entry_to_high_risk_account_is_only_a_warning(settings: Settings) -> None:
    signal = jc.check_high_risk_account(
        make_entry(account="610000", manual_posting=False), settings
    )
    assert signal.triggered and signal.severity == "warning"


def test_standard_account_is_not_flagged(settings: Settings) -> None:
    assert not jc.check_high_risk_account(make_entry(account="600000"), settings).triggered


# ------------------------------------------------------------------ other rules
def test_duplicate_candidates_produce_a_critical_signal(settings: Settings) -> None:
    signal = jc.check_duplicate_posting(make_entry(), [{"journal_id": "JE-202607-00002"}], settings)
    assert signal.triggered and signal.severity == "critical"


def test_round_amount_rule_needs_both_size_and_roundness(settings: Settings) -> None:
    assert jc.check_round_amount(make_entry(amount=150_000.0), settings).triggered
    assert not jc.check_round_amount(make_entry(amount=150_432.17), settings).triggered
    assert not jc.check_round_amount(make_entry(amount=20_000.0), settings).triggered


def test_weak_narratives_are_flagged(settings: Settings) -> None:
    assert jc.check_narrative_quality(make_entry(description="Adjustment"), settings).triggered
    assert not jc.check_narrative_quality(
        make_entry(description="Accrual for external audit fees"), settings
    ).triggered


def test_period_integrity_detects_a_prior_period_document(settings: Settings) -> None:
    entry = make_entry(document_date=date(2026, 6, 28), posting_date=date(2026, 7, 3))
    assert jc.check_period_integrity(entry, settings).triggered


def test_unused_account_cost_center_combination_is_flagged(settings: Settings) -> None:
    assert jc.check_account_cost_center_combination(make_entry(), 0, settings).triggered
    assert not jc.check_account_cost_center_combination(make_entry(), 7, settings).triggered


def test_threshold_splitting_is_detected(settings: Settings) -> None:
    entry = make_entry(amount=30_000.0)
    assert jc.check_same_day_aggregation(entry, 80_000.0, settings).triggered
    assert not jc.check_same_day_aggregation(entry, 40_000.0, settings).triggered


# --------------------------------------------------------------- reconciliation
def test_open_reconciliation_is_flagged(settings: Settings) -> None:
    record = {"status": "open", "reviewer": "", "difference": 0.0, "days_open": 3}
    assert check_reconciliation_status(make_entry(account="113000"), record, settings).triggered


def test_material_reconciliation_difference_is_critical(settings: Settings) -> None:
    record = {"status": "in_progress", "difference": settings.materiality_group + 1, "days_open": 4}
    signal = check_reconciliation_difference(make_entry(account="113000"), record, settings)
    assert signal.triggered and signal.severity == "critical"


def test_small_reconciliation_difference_is_not_flagged(settings: Settings) -> None:
    record = {"status": "reconciled", "difference": 120.0, "days_open": 2}
    assert not check_reconciliation_difference(
        make_entry(account="113000"), record, settings
    ).triggered


def test_aged_reconciling_item_is_flagged_regardless_of_amount(settings: Settings) -> None:
    record = {"status": "open", "difference": 300.0, "days_open": 90}
    assert check_reconciliation_difference(make_entry(account="113000"), record, settings).triggered


def test_uncleared_suspense_account_is_critical(settings: Settings) -> None:
    record = {"status": "open", "difference": 40_000.0, "gl_balance": 40_000.0, "days_open": 55}
    signal = check_suspense_cleared(make_entry(account="199000"), record, settings)
    assert signal.triggered and signal.severity == "critical"


def test_variance_needs_both_percentage_and_absolute_movement(settings: Settings) -> None:
    big = VarianceResult(600_000.0, 200_000.0, 400_000.0, 200.0)
    small_absolute = VarianceResult(9_000.0, 3_000.0, 6_000.0, 200.0)
    assert check_account_variance(make_entry(), big, settings).triggered
    assert not check_account_variance(make_entry(), small_absolute, settings).triggered


# ------------------------------------------------------- mandatory escalation
def test_critical_signals_become_mandatory_escalation_triggers(settings: Settings) -> None:
    signals = [
        jc.check_supporting_document(
            make_entry(manual_posting=True, supporting_document=None, amount=20_000.0), settings
        ),
        jc.check_business_hours(
            make_entry(posting_timestamp=datetime(2026, 7, 14, 3, 0)), settings
        ),
    ]
    triggers = mandatory_escalation_triggers(signals)
    assert any("CHK-01" in t for t in triggers)
    assert not any("CHK-05" in t for t in triggers), "a warning must not force escalation"


def test_clean_entry_produces_no_mandatory_triggers(settings: Settings) -> None:
    signals = [
        jc.check_supporting_document(make_entry(), settings),
        jc.check_approval_threshold(make_entry(), settings),
        check_materiality(make_entry(), settings),
    ]
    assert mandatory_escalation_triggers(signals) == []
