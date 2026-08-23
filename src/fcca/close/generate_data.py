"""Reproducible synthetic close dataset.

Produces four artefacts from a fixed seed:

``data/raw/journal_entries.csv``      ~800 postings across three periods
``data/raw/reconciliations.csv``      balance-sheet reconciliation status per period
``data/raw/close_exceptions.csv``     ~60 exceptions raised by close monitoring
``data/evaluation/labelled_exceptions.json``  expected outcome for every exception

Everything is fictional. The point of generating rather than committing the data
is that a reviewer can regenerate it, change a threshold and see the effect,
which is not possible with a static CSV of unexplained numbers.

**On the labels.** Each exception is built from a named scenario whose expected
risk rating, review requirement, remediation category and governing policy follow
from the policy set in ``policies/``. The labels are therefore ground truth *by
construction*, derived from the scenario definition and not from a human review
of production data. That is a real limitation and it is stated in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fcca.close.masterdata import (
    ACCOUNT_NAMES,
    ACCOUNTS,
    BUSINESS_NARRATIVES,
    CONTROLLERS,
    COST_CENTERS,
    DOCUMENT_TYPES,
    ENTITIES,
    FX_TO_EUR,
    USERS,
    WEAK_NARRATIVES,
)
from fcca.close.models import ActionCategory, LabelledCase, RiskLevel
from fcca.shared.config import Settings, get_settings

logger = logging.getLogger(__name__)

PERIODS: tuple[str, ...] = ("2026-05", "2026-06", "2026-07")
CLOSE_PERIOD = "2026-07"

BS_ACCOUNTS = [a.number for a in ACCOUNTS if a.kind == "bs"]
PL_ACCOUNTS = [a.number for a in ACCOUNTS if a.kind == "pl"]

#: Each entity uses a stable subset of cost centres. A posting to a cost centre
#: outside this subset is what the close policy calls an unexpected combination.
ENTITY_COST_CENTERS: dict[str, tuple[str, ...]] = {
    "DE10": ("CC-1000", "CC-3000", "CC-6000", "CC-2000"),
    "CH20": ("CC-1000", "CC-2000", "CC-7000"),
    "NL30": ("CC-1000", "CC-3000", "CC-2000"),
    "US40": ("CC-1000", "CC-2000", "CC-4000"),
    "PL50": ("CC-1000", "CC-4000", "CC-5000"),
}

ALL_COST_CENTERS = [c for c, _ in COST_CENTERS]

#: Balance-sheet accounts reserved for reconciliation scenarios. Keeping them out
#: of every other scenario guarantees that one exception's patched reconciliation
#: record can never change the expected outcome of a different exception.
RECON_ACCOUNTS: tuple[str, ...] = ("113000", "141000", "210000", "160000", "231000")

#: Suspense and clearing accounts have a dedicated scenario and are excluded from
#: the other high-risk-account scenarios so each case tests one thing.
SUSPENSE_ACCOUNTS: tuple[str, ...] = ("199000",)

#: (company_code, account) pairs already claimed by a reconciliation scenario in
#: the current generation run. Reset by :func:`generate`.
_RESERVED_RECON: set[tuple[str, str]] = set()


def _reserve_recon_account(entry: dict[str, Any], s: Settings) -> str:
    """Claim an unused balance-sheet account for this entity's reconciliation case."""
    for account in RECON_ACCOUNTS:
        key = (entry["company_code"], account)
        if key not in _RESERVED_RECON and account not in s.high_risk_accounts:
            _RESERVED_RECON.add(key)
            return account
    fallback = [
        a
        for a in BS_ACCOUNTS
        if a not in s.high_risk_accounts and (entry["company_code"], a) not in _RESERVED_RECON
    ]
    account = fallback[0] if fallback else BS_ACCOUNTS[0]
    _RESERVED_RECON.add((entry["company_code"], account))
    return account


JOURNAL_FIELDS = [
    "journal_id",
    "company_code",
    "account",
    "account_name",
    "cost_center",
    "posting_date",
    "document_date",
    "posting_timestamp",
    "amount",
    "currency",
    "amount_reporting_ccy",
    "user_id",
    "document_type",
    "description",
    "manual_posting",
    "supporting_document",
    "reconciliation_status",
    "approved_by",
]

RECON_FIELDS = [
    "reconciliation_id",
    "company_code",
    "account",
    "account_name",
    "period",
    "gl_balance",
    "supporting_balance",
    "difference",
    "status",
    "preparer",
    "reviewer",
    "days_open",
    "last_updated",
]

EXCEPTION_FIELDS = [
    "exception_id",
    "journal_id",
    "company_code",
    "exception_type",
    "detected_at",
    "close_period",
    "source_system",
    "description",
    "reported_amount",
    "currency",
]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Build:
    """What a scenario produced: a patched entry plus optional side effects."""

    entry: dict[str, Any]
    description: str
    extra_entries: list[dict[str, Any]] = field(default_factory=list)
    reconciliation_patch: dict[str, Any] | None = None


@dataclass(frozen=True)
class Scenario:
    """A named exception pattern and the outcome the policy set implies for it."""

    key: str
    exception_type: str
    build: Callable[[random.Random, dict[str, Any], Settings], Build]
    expected_risk_level: RiskLevel
    expected_requires_human_review: bool
    expected_action_category: ActionCategory
    expected_policy_document: str
    notes: str


def _sc_unsupported_manual_material(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["supporting_document"] = ""
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.materiality_group * 1.2, s.materiality_group * 2.5
    )
    e["approved_by"] = _approver(rng, e)
    e["description"] = rng.choice(WEAK_NARRATIVES)
    return Build(
        e,
        "Manual journal entry above group materiality posted without a supporting document reference.",
    )


def _sc_unsupported_manual_minor(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SB"
    e["supporting_document"] = ""
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold * 1.5, s.journal_approval_threshold * 0.6
    )
    e["approved_by"] = ""
    e["description"] = rng.choice(WEAK_NARRATIVES)
    return Build(
        e,
        "Manual journal entry below the approval threshold posted without supporting documentation.",
    )


def _sc_unsupported_manual_high_risk(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["account"] = _elevated_risk_account(rng, s)
    e["supporting_document"] = ""
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold * 2, s.journal_approval_threshold * 0.8
    )
    e["approved_by"] = ""
    e["description"] = rng.choice(WEAK_NARRATIVES)
    return Build(e, "Unsupported manual entry posted to a high-risk account.")


def _sc_late_manual_posting(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold * 2, s.journal_approval_threshold * 0.8
    )
    e["supporting_document"] = f"ACCR-{CLOSE_PERIOD}-{rng.randint(100, 999)}"
    e["document_date"] = _shift(e["posting_date"], -rng.randint(12, 34))
    e["approved_by"] = ""
    return Build(
        e, "Manual posting made well after the document date without a recorded justification."
    )


def _approver(rng: random.Random, entry: dict[str, Any]) -> str:
    """Entity Financial Controller, unless that is the preparer.

    Without this guard a scenario that only meant to test one control would also
    trip the segregation-of-duties check whenever the controller happened to be
    drawn as the preparer, and the labelled outcome would no longer match the
    scenario it is named after.
    """
    controller = CONTROLLERS[entry["company_code"]]
    if controller == entry["user_id"]:
        return _other_user(rng, entry["company_code"], entry["user_id"])
    return controller


def _elevated_risk_account(rng: random.Random, s: Settings) -> str:
    """A high-risk account other than suspense, which has its own scenario."""
    return rng.choice([a for a in s.high_risk_accounts if a not in SUSPENSE_ACCOUNTS])


def _routine_account(rng: random.Random, s: Settings) -> str:
    """Pick an account with no elevated inherent risk and no reconciliation scenario."""
    return rng.choice(
        [a for a in ACCOUNT_NAMES if a not in s.high_risk_accounts and a not in RECON_ACCOUNTS]
    )


def _sc_late_manual_posting_trivial(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SB"
    e["amount"] = _amount_in_reporting_range(rng, e, 200.0, s.trivial_threshold * 0.5)
    e["supporting_document"] = f"INV-{rng.randint(100000, 999999)}"
    e["document_date"] = _shift(e["posting_date"], -rng.randint(7, 11))
    e["approved_by"] = ""
    return Build(e, "Small supported manual posting made a few days after the policy window.")


def _sc_out_of_hours_supported(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(rng, e, 500.0, s.trivial_threshold * 0.6)
    e["supporting_document"] = f"ACCR-{CLOSE_PERIOD}-{rng.randint(100, 999)}"
    e["posting_timestamp"] = _at_hour(e["posting_date"], rng.choice([5, 6, 21, 22]), rng)
    e["approved_by"] = ""
    return Build(e, "Manual posting created outside normal business hours during the close window.")


def _sc_out_of_hours_unsupported(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["account"] = _elevated_risk_account(rng, s)
    e["supporting_document"] = ""
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.journal_approval_threshold * 1.3, s.materiality_group * 0.7
    )
    e["posting_timestamp"] = _at_hour(e["posting_date"], rng.choice([2, 3, 23]), rng)
    e["approved_by"] = ""
    e["description"] = rng.choice(WEAK_NARRATIVES)
    return Build(
        e,
        "Unsupported manual entry to a high-risk account posted outside business hours and without approval.",
    )


def _sc_threshold_breach_unapproved(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.journal_approval_threshold * 1.1, s.materiality_group * 0.6
    )
    e["supporting_document"] = f"CONTR-{rng.randint(1000, 9999)}"
    e["approved_by"] = ""
    return Build(
        e, "Entry above the second-level approval threshold posted without a recorded approver."
    )


def _sc_threshold_breach_material(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.materiality_group * 1.1, s.materiality_group * 3.0
    )
    e["supporting_document"] = f"CONTR-{rng.randint(1000, 9999)}"
    e["approved_by"] = ""
    return Build(
        e, "Entry above group materiality posted without documented second-level approval."
    )


def _sc_duplicate_posting(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = False
    e["document_type"] = "KR"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.journal_approval_threshold * 0.8, s.materiality_group * 0.9
    )
    doc = f"INV-{rng.randint(100000, 999999)}"
    e["supporting_document"] = doc
    e["approved_by"] = _approver(rng, e) if abs(e["amount"]) >= s.journal_approval_threshold else ""
    twin = dict(e)
    twin["journal_id"] = e["journal_id"] + "D"
    twin["posting_timestamp"] = _at_hour(
        e["posting_date"], min(23, _hour_of(e["posting_timestamp"]) + 1), rng
    )
    twin["user_id"] = _other_user(rng, e["company_code"], e["user_id"])
    return Build(
        e,
        "Two postings with identical company code, account, amount and document date detected in the same period.",
        extra_entries=[twin],
    )


def _sc_segregation_of_duties(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.journal_approval_threshold * 1.2, s.materiality_group * 0.8
    )
    e["supporting_document"] = f"MEMO-{rng.randint(100, 999)}"
    e["approved_by"] = e["user_id"]
    return Build(e, "Journal entry prepared and approved by the same user id.")


def _sc_round_amount_high_risk(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["account"] = _elevated_risk_account(rng, s)
    e["amount"] = float(rng.choice([100_000, 150_000, 200_000, 250_000]))
    e["supporting_document"] = f"MEMO-{rng.randint(100, 999)}"
    e["approved_by"] = _approver(rng, e)
    e["description"] = rng.choice(WEAK_NARRATIVES)
    e["user_id"] = _other_user(rng, e["company_code"], e["user_id"])
    return Build(
        e, "Round-amount manual entry to a high-risk account with a non-descriptive narrative."
    )


def _sc_period_integrity(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["account"] = _routine_account(rng, s)
    e["document_type"] = "SB"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold * 3, s.journal_approval_threshold * 0.9
    )
    e["supporting_document"] = f"INV-{rng.randint(100000, 999999)}"
    e["document_date"] = _shift(e["posting_date"], -rng.randint(45, 75))
    e["approved_by"] = ""
    return Build(
        e,
        "Entry recognised in a later period than the underlying document date without explanation.",
    )


def _sc_recon_mismatch_material(rng: random.Random, e: dict, s: Settings) -> Build:
    e["account"] = _reserve_recon_account(e, s)
    e["reconciliation_status"] = "in_progress"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold, s.journal_approval_threshold
    )
    e["supporting_document"] = f"REC-{rng.randint(1000, 9999)}"
    diff = round(rng.uniform(s.materiality_group * 1.1, s.materiality_group * 2.2), 2)
    return Build(
        e,
        "Balance sheet reconciliation shows an unexplained difference above group materiality.",
        reconciliation_patch={
            "difference": diff,
            "status": "in_progress",
            "days_open": rng.randint(5, 20),
        },
    )


def _sc_recon_mismatch_minor(rng: random.Random, e: dict, s: Settings) -> Build:
    e["account"] = _reserve_recon_account(e, s)
    e["reconciliation_status"] = "in_progress"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold, s.journal_approval_threshold
    )
    e["supporting_document"] = f"REC-{rng.randint(1000, 9999)}"
    diff = round(rng.uniform(s.trivial_threshold * 1.2, s.journal_approval_threshold * 0.7), 2)
    return Build(
        e,
        "Balance sheet reconciliation shows a difference above the investigation threshold.",
        reconciliation_patch={
            "difference": diff,
            "status": "in_progress",
            "days_open": rng.randint(3, 25),
        },
    )


def _sc_incomplete_reconciliation(rng: random.Random, e: dict, s: Settings) -> Build:
    e["account"] = _reserve_recon_account(e, s)
    e["reconciliation_status"] = "open"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold, s.journal_approval_threshold
    )
    e["supporting_document"] = f"REC-{rng.randint(1000, 9999)}"
    return Build(
        e,
        "Account reconciliation still open at entity sign-off.",
        reconciliation_patch={
            "difference": round(rng.uniform(200.0, s.trivial_threshold * 0.8), 2),
            "status": "open",
            "reviewer": "",
            "days_open": rng.randint(30, 95),
        },
    )


def _sc_suspense_account_open(rng: random.Random, e: dict, s: Settings) -> Build:
    e["account"] = "199000"
    e["reconciliation_status"] = "open"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.trivial_threshold * 2, s.journal_approval_threshold
    )
    e["supporting_document"] = ""
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["approved_by"] = ""
    return Build(
        e,
        "Suspense and clearing account carries a non-zero balance at sign-off.",
        reconciliation_patch={
            "difference": round(
                rng.uniform(s.journal_approval_threshold, s.materiality_group * 0.9), 2
            ),
            "status": "open",
            "reviewer": "",
            "days_open": rng.randint(40, 120),
        },
    )


def _sc_material_variance(rng: random.Random, e: dict, s: Settings) -> Build:
    e["account"] = rng.choice(["600000", "610000", "620000", "640000", "500000"])
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["amount"] = _amount_in_reporting_range(
        rng, e, s.materiality_group * 1.4, s.materiality_group * 2.4
    )
    e["supporting_document"] = f"ACCR-{CLOSE_PERIOD}-{rng.randint(100, 999)}"
    e["approved_by"] = _approver(rng, e)
    return Build(
        e,
        "Account balance moved by more than 50 percent month on month, driven by a single large accrual.",
    )


def _sc_unexpected_combination(rng: random.Random, e: dict, s: Settings) -> Build:
    unused = [c for c in ALL_COST_CENTERS if c not in ENTITY_COST_CENTERS[e["company_code"]]]
    e["cost_center"] = rng.choice(unused)
    e["account"] = rng.choice(["620000", "640000", "600000", "700000"])
    e["manual_posting"] = False
    e["document_type"] = "KR"
    e["amount"] = _amount_in_reporting_range(rng, e, 400.0, s.trivial_threshold * 0.7)
    e["supporting_document"] = f"INV-{rng.randint(100000, 999999)}"
    e["approved_by"] = ""
    return Build(
        e,
        "Posting uses an account and cost center combination not seen for this entity in prior periods.",
    )


def _sc_compliant_flagged(rng: random.Random, e: dict, s: Settings) -> Build:
    e["manual_posting"] = True
    e["document_type"] = "SA"
    e["account"] = rng.choice(["600000", "700000", "620000", "500000"])
    e["amount"] = _amount_in_reporting_range(rng, e, 800.0, s.trivial_threshold * 0.7)
    e["supporting_document"] = f"ACCR-{CLOSE_PERIOD}-{rng.randint(100, 999)}"
    e["document_date"] = _shift(e["posting_date"], -rng.randint(0, 3))
    e["posting_timestamp"] = _at_hour(e["posting_date"], rng.randint(9, 17), rng)
    e["approved_by"] = _other_user(rng, e["company_code"], e["user_id"])
    e["description"] = rng.choice(BUSINESS_NARRATIVES)
    return Build(
        e,
        "Routine accrual flagged by the monitoring rule for manual postings during the close window.",
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "unsupported_manual_material",
        "missing_supporting_documentation",
        _sc_unsupported_manual_material,
        "high",
        True,
        "escalate_to_financial_controller",
        "Supporting Documentation Standard",
        "Missing documentation above group materiality.",
    ),
    Scenario(
        "unsupported_manual_minor",
        "missing_supporting_documentation",
        _sc_unsupported_manual_minor,
        "medium",
        True,
        "request_supporting_documentation",
        "Supporting Documentation Standard",
        "Missing documentation, amount below approval threshold.",
    ),
    Scenario(
        "unsupported_manual_high_risk",
        "missing_supporting_documentation",
        _sc_unsupported_manual_high_risk,
        "high",
        True,
        "escalate_to_financial_controller",
        "Supporting Documentation Standard",
        "High-risk account overrides the amount-based rating.",
    ),
    Scenario(
        "late_manual_posting",
        "late_manual_posting",
        _sc_late_manual_posting,
        "medium",
        True,
        "request_justification",
        "Manual Posting Control",
        "Late posting requires documented justification.",
    ),
    Scenario(
        "late_manual_posting_trivial",
        "late_manual_posting",
        _sc_late_manual_posting_trivial,
        "low",
        False,
        "route_to_preparer",
        "Manual Posting Control",
        "Supported, clearly trivial, only marginally late.",
    ),
    Scenario(
        "out_of_hours_supported",
        "out_of_hours_posting",
        _sc_out_of_hours_supported,
        "low",
        False,
        "no_action",
        "Manual Posting Control",
        "Out-of-hours alone is a risk indicator, not a finding.",
    ),
    Scenario(
        "out_of_hours_unsupported",
        "out_of_hours_posting",
        _sc_out_of_hours_unsupported,
        "high",
        True,
        "escalate_to_financial_controller",
        "Manual Posting Control",
        "Several indicators coincide.",
    ),
    Scenario(
        "threshold_breach_unapproved",
        "threshold_breach",
        _sc_threshold_breach_unapproved,
        "medium",
        True,
        "route_to_reviewer",
        "Journal Entry Policy",
        "Above approval threshold, below materiality, supported.",
    ),
    Scenario(
        "threshold_breach_material",
        "threshold_breach",
        _sc_threshold_breach_material,
        "high",
        True,
        "escalate_to_financial_controller",
        "Journal Entry Policy",
        "Above group materiality without approval.",
    ),
    Scenario(
        "duplicate_posting",
        "duplicate_posting",
        _sc_duplicate_posting,
        "high",
        True,
        "propose_correcting_entry",
        "Journal Entry Policy",
        "Suspected duplicate must be cleared before sign-off.",
    ),
    Scenario(
        "segregation_of_duties",
        "unusual_journal_entry",
        _sc_segregation_of_duties,
        "high",
        True,
        "refer_to_internal_audit",
        "Journal Entry Policy",
        "Preparer equals approver: control failure.",
    ),
    Scenario(
        "round_amount_high_risk",
        "unusual_journal_entry",
        _sc_round_amount_high_risk,
        "high",
        True,
        "escalate_to_financial_controller",
        "Journal Entry Policy",
        "Round amount, high-risk account, weak narrative.",
    ),
    Scenario(
        "period_integrity",
        "unusual_journal_entry",
        _sc_period_integrity,
        "medium",
        True,
        "request_justification",
        "Month-End Close Policy",
        "Document date in an earlier period than the posting period.",
    ),
    Scenario(
        "reconciliation_mismatch_material",
        "reconciliation_mismatch",
        _sc_recon_mismatch_material,
        "high",
        True,
        "escalate_to_financial_controller",
        "Account Reconciliation Policy",
        "Unexplained difference above materiality.",
    ),
    Scenario(
        "reconciliation_mismatch_minor",
        "reconciliation_mismatch",
        _sc_recon_mismatch_minor,
        "medium",
        True,
        "route_to_preparer",
        "Account Reconciliation Policy",
        "Difference above the investigation threshold.",
    ),
    Scenario(
        "incomplete_reconciliation",
        "incomplete_reconciliation",
        _sc_incomplete_reconciliation,
        "medium",
        True,
        "route_to_preparer",
        "Account Reconciliation Policy",
        "Reconciliation open at sign-off.",
    ),
    Scenario(
        "suspense_account_open",
        "incomplete_reconciliation",
        _sc_suspense_account_open,
        "high",
        True,
        "escalate_to_financial_controller",
        "Account Reconciliation Policy",
        "Suspense account must clear to zero at month-end.",
    ),
    Scenario(
        "material_variance",
        "material_variance",
        _sc_material_variance,
        "high",
        True,
        "request_justification",
        "Materiality and Escalation Policy",
        "Movement above the variance escalation trigger.",
    ),
    Scenario(
        "unexpected_combination",
        "unexpected_account_cost_center",
        _sc_unexpected_combination,
        "low",
        False,
        "route_to_preparer",
        "Month-End Close Policy",
        "Data-quality finding, no aggravating indicator.",
    ),
    Scenario(
        "compliant_flagged",
        "no_finding",
        _sc_compliant_flagged,
        "low",
        False,
        "no_action",
        "Month-End Close Policy",
        "True negative: the monitoring rule fired but the entry is compliant.",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period_bounds(period: str) -> tuple[date, date]:
    year, month = (int(p) for p in period.split("-"))
    first = date(year, month, 1)
    last = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return first, last


def _working_day(rng: random.Random, period: str) -> date:
    first, last = _period_bounds(period)
    for _ in range(20):
        day = first + timedelta(days=rng.randint(0, (last - first).days))
        if day.weekday() < 5:
            return day
    return first


def _shift(day: date | str, days: int) -> date:
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return day + timedelta(days=days)


def _at_hour(day: date | str, hour: int, rng: random.Random) -> datetime:
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return datetime(day.year, day.month, day.day, hour, rng.randint(0, 59), rng.randint(0, 59))


def _hour_of(value: datetime | str) -> int:
    if isinstance(value, str):
        return datetime.fromisoformat(value).hour
    return value.hour


def _other_user(rng: random.Random, company_code: str, exclude: str) -> str:
    """Another *human* user of the same entity."""
    candidates = [
        u
        for u, (role, ent) in USERS.items()
        if ent == company_code and u != exclude and role != "system"
    ]
    if not candidates:
        candidates = [u for u, (role, _) in USERS.items() if u != exclude and role != "system"]
    return rng.choice(candidates)


def _amount_in_reporting_range(
    rng: random.Random, entry: dict[str, Any], low_eur: float, high_eur: float
) -> float:
    """Draw a transaction-currency amount whose EUR equivalent lies in a range."""
    fx = FX_TO_EUR[entry["currency"]]
    target_eur = rng.uniform(low_eur, high_eur)
    return round(target_eur / fx, 2)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _baseline_entry(
    rng: random.Random, seq: int, period: str, settings: Settings
) -> dict[str, Any]:
    entity = rng.choice(ENTITIES)
    cost_center = rng.choice(ENTITY_COST_CENTERS[entity.company_code])
    account = rng.choice(rng.choices([BS_ACCOUNTS, PL_ACCOUNTS], weights=[0.35, 0.65], k=1)[0])
    doc_type = rng.choices(
        list(DOCUMENT_TYPES), weights=[0.14, 0.08, 0.32, 0.16, 0.12, 0.10, 0.08], k=1
    )[0]
    manual = DOCUMENT_TYPES[doc_type][1]
    posting_date = _working_day(rng, period)
    document_date = _shift(posting_date, -rng.randint(0, 4))
    user = rng.choice(
        [u for u, (role, ent) in USERS.items() if ent == entity.company_code and role != "system"]
        or list(USERS)
    )
    if doc_type in {"AF", "ZP"}:
        user = "svc.interface"
        manual = False

    magnitude = rng.choices([1, 2, 3], weights=[0.62, 0.31, 0.07], k=1)[0]
    base_eur = {
        1: rng.uniform(150, 9_000),
        2: rng.uniform(9_000, 48_000),
        3: rng.uniform(48_000, 210_000),
    }[magnitude]
    amount = round(base_eur / FX_TO_EUR[entity.currency], 2)
    if rng.random() < 0.42:
        amount = -amount

    amount_eur = abs(amount) * FX_TO_EUR[entity.currency]
    approved_by = ""
    if amount_eur >= settings.journal_approval_threshold:
        approved_by = CONTROLLERS[entity.company_code]
        if approved_by == user:
            approved_by = _other_user(rng, entity.company_code, user)

    supporting = f"INV-{rng.randint(100000, 999999)}"
    if manual:
        supporting = f"ACCR-{period}-{rng.randint(100, 999)}" if rng.random() > 0.03 else ""

    if account in BS_ACCOUNTS:
        recon_status = rng.choices(
            ["reconciled", "in_progress", "open"], weights=[0.88, 0.08, 0.04], k=1
        )[0]
    else:
        recon_status = "not_applicable"

    return {
        "journal_id": f"JE-{period.replace('-', '')}-{seq:05d}",
        "company_code": entity.company_code,
        "account": account,
        "account_name": ACCOUNT_NAMES[account],
        "cost_center": cost_center,
        "posting_date": posting_date,
        "document_date": document_date,
        "posting_timestamp": _at_hour(posting_date, rng.randint(8, 18), rng),
        "amount": amount,
        "currency": entity.currency,
        "amount_reporting_ccy": 0.0,  # filled in _finalise
        "user_id": user,
        "document_type": doc_type,
        "description": rng.choice(BUSINESS_NARRATIVES),
        "manual_posting": manual,
        "supporting_document": supporting,
        "reconciliation_status": recon_status,
        "approved_by": approved_by,
    }


def _finalise(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive dependent fields and normalise types for CSV output."""
    entry["account_name"] = ACCOUNT_NAMES[entry["account"]]
    entry["amount_reporting_ccy"] = round(entry["amount"] * FX_TO_EUR[entry["currency"]], 2)
    if isinstance(entry["posting_date"], str):
        entry["posting_date"] = date.fromisoformat(entry["posting_date"])
    if isinstance(entry["document_date"], str):
        entry["document_date"] = date.fromisoformat(entry["document_date"])
    return entry


def _build_reconciliations(
    rng: random.Random, entries: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """One reconciliation record per (entity, balance-sheet account, period)."""
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    seq = 0
    for entity in ENTITIES:
        for account in BS_ACCOUNTS:
            for period in PERIODS:
                seq += 1
                gl = round(
                    sum(
                        e["amount_reporting_ccy"]
                        for e in entries
                        if e["company_code"] == entity.company_code
                        and e["account"] == account
                        and str(e["posting_date"])[:7] == period
                    ),
                    2,
                )
                status = (
                    "reconciled"
                    if period != CLOSE_PERIOD
                    else rng.choices(["reconciled", "in_progress"], weights=[0.9, 0.1], k=1)[0]
                )
                preparer = rng.choice(
                    [
                        u
                        for u, (role, ent) in USERS.items()
                        if ent == entity.company_code and role == "accountant"
                    ]
                    or ["u.osei"]
                )
                records[(entity.company_code, account, period)] = {
                    "reconciliation_id": f"REC-{seq:04d}",
                    "company_code": entity.company_code,
                    "account": account,
                    "account_name": ACCOUNT_NAMES[account],
                    "period": period,
                    "gl_balance": gl,
                    "supporting_balance": gl,
                    "difference": 0.0,
                    "status": status,
                    "preparer": preparer,
                    "reviewer": CONTROLLERS[entity.company_code] if status == "reconciled" else "",
                    "days_open": 0,
                    "last_updated": str(_period_bounds(period)[1]),
                }
    return records


def generate(settings: Settings | None = None) -> dict[str, Path]:
    """Generate the full dataset and write it to ``data/``.

    Returns a mapping of artefact name to path.
    """
    settings = settings or get_settings()
    settings.ensure_directories()
    rng = random.Random(settings.random_seed)
    _RESERVED_RECON.clear()

    # --- baseline population -------------------------------------------------
    entries: list[dict[str, Any]] = []
    per_period = settings.n_journal_entries // len(PERIODS)
    seq = 0
    for period in PERIODS:
        for _ in range(per_period):
            seq += 1
            entries.append(_finalise(_baseline_entry(rng, seq, period, settings)))

    reconciliations = _build_reconciliations(rng, entries)

    # --- exceptions ----------------------------------------------------------
    exceptions: list[dict[str, Any]] = []
    labels: list[LabelledCase] = []
    plan: list[Scenario] = []
    while len(plan) < settings.n_exceptions:
        plan.extend(SCENARIOS)
    plan = plan[: settings.n_exceptions]

    for i, scenario in enumerate(plan, start=1):
        seq += 1
        base = _baseline_entry(rng, seq, CLOSE_PERIOD, settings)
        base["journal_id"] = f"JE-{CLOSE_PERIOD.replace('-', '')}-X{i:04d}"
        if USERS[base["user_id"]][0] == "system":
            # Scenarios below may turn this posting into a manual entry, and a
            # manual entry by an interface service account is not a realistic case.
            base["user_id"] = _other_user(rng, base["company_code"], base["user_id"])
        build = scenario.build(rng, base, settings)
        entry = _finalise(build.entry)
        entries.append(entry)
        for extra in build.extra_entries:
            entries.append(_finalise(extra))

        if build.reconciliation_patch is not None:
            key = (entry["company_code"], entry["account"], CLOSE_PERIOD)
            record = reconciliations.get(key)
            if record is not None:
                record.update(build.reconciliation_patch)
                record["supporting_balance"] = round(
                    record["gl_balance"] - float(record["difference"]), 2
                )

        exception_id = f"EXC-{i:04d}"
        exceptions.append(
            {
                "exception_id": exception_id,
                "journal_id": entry["journal_id"],
                "company_code": entry["company_code"],
                "exception_type": scenario.exception_type,
                "detected_at": _at_hour(_shift(entry["posting_date"], 1), 6, rng).isoformat(),
                "close_period": CLOSE_PERIOD,
                "source_system": "CCM-Monitor",
                "description": build.description,
                "reported_amount": entry["amount"],
                "currency": entry["currency"],
            }
        )
        labels.append(
            LabelledCase(
                exception_id=exception_id,
                scenario=scenario.key,
                expected_risk_level=scenario.expected_risk_level,
                expected_requires_human_review=scenario.expected_requires_human_review,
                expected_action_category=scenario.expected_action_category,
                expected_policy_document=scenario.expected_policy_document,
                notes=scenario.notes,
            )
        )

    # --- write ---------------------------------------------------------------
    _write_csv(settings.journal_entries_path, JOURNAL_FIELDS, entries)
    _write_csv(settings.reconciliations_path, RECON_FIELDS, list(reconciliations.values()))
    _write_csv(settings.exceptions_path, EXCEPTION_FIELDS, exceptions)
    settings.labelled_set_path.write_text(
        json.dumps([label.model_dump() for label in labels], indent=2) + "\n",
        encoding="utf-8",
    )

    from fcca.close.analytics import build_close_database  # local import to avoid a cycle

    build_close_database(settings)

    logger.info(
        "generated %d journal entries, %d reconciliations, %d exceptions",
        len(entries),
        len(reconciliations),
        len(exceptions),
    )
    return {
        "journal_entries": settings.journal_entries_path,
        "reconciliations": settings.reconciliations_path,
        "close_exceptions": settings.exceptions_path,
        "labelled_exceptions": settings.labelled_set_path,
        "close_db": settings.close_db_path,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime | date):
        return value.isoformat()
    if value is None:
        return ""
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca generate-data",
        description="Generate the reproducible synthetic close dataset.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    parser.add_argument("--entries", type=int, default=None, help="Number of journal entries.")
    parser.add_argument("--exceptions", type=int, default=None, help="Number of close exceptions.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.entries is not None:
        overrides["n_journal_entries"] = args.entries
    if args.exceptions is not None:
        overrides["n_exceptions"] = args.exceptions
    if overrides:
        settings = settings.model_copy(update=overrides)

    written = generate(settings)
    for name, path in written.items():
        print(f"  {name:22s} {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
