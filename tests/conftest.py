"""Shared test fixtures.

Every test runs against a throwaway dataset generated into a temporary base
directory, so the suite never depends on artefacts left behind by a manual run
and never writes into the working tree. All tests use the mock provider: nothing
in this suite can make a paid API call.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from fcca.close.models import CloseException, JournalEntry
from fcca.shared.config import Settings, get_settings, reset_settings_cache

# Import paths come from `pythonpath` in pyproject.toml, so the suite runs the
# same way whether or not the package is installed, and however pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Generate a small, isolated dataset and policy index for the whole session."""
    base = tmp_path_factory.mktemp("fcca-sandbox")
    shutil.copytree(ROOT / "policies", base / "policies")

    os.environ.update(
        {
            "FCCA_BASE_DIR": str(base),
            "LLM_PROVIDER": "mock",
            "FCCA_N_JOURNAL_ENTRIES": "240",
            "FCCA_N_EXCEPTIONS": "20",
        }
    )
    reset_settings_cache()
    settings = get_settings()

    from fcca.close.generate_data import generate
    from fcca.close.retrieval.index import build_policy_index
    from fcca.close.retrieval.retriever import clear_retriever_cache

    generate(settings)
    build_policy_index(settings)
    clear_retriever_cache()
    return settings


@pytest.fixture
def settings(sandbox: Settings) -> Settings:
    return sandbox


def make_entry(**overrides: Any) -> JournalEntry:
    """A compliant baseline journal entry that individual tests deform."""
    defaults: dict[str, Any] = {
        "journal_id": "JE-202607-00001",
        "company_code": "DE10",
        "account": "600000",
        "account_name": "Personnel expenses",
        "cost_center": "CC-1000",
        "posting_date": date(2026, 7, 14),
        "document_date": date(2026, 7, 12),
        "posting_timestamp": datetime(2026, 7, 14, 10, 30),
        "amount": 12_500.00,
        "currency": "EUR",
        "amount_reporting_ccy": 12_500.00,
        "user_id": "u.becker",
        "document_type": "SA",
        "description": "Accrual for services received not invoiced",
        "manual_posting": True,
        "supporting_document": "ACCR-2026-07-114",
        "reconciliation_status": "not_applicable",
        "approved_by": None,
    }
    defaults.update(overrides)
    return JournalEntry.model_validate(defaults)


def make_exception(**overrides: Any) -> CloseException:
    defaults: dict[str, Any] = {
        "exception_id": "EXC-9001",
        "journal_id": "JE-202607-00001",
        "company_code": "DE10",
        "exception_type": "missing_supporting_documentation",
        "detected_at": datetime(2026, 7, 15, 6, 0),
        "close_period": "2026-07",
        "source_system": "CCM-Monitor",
        "description": "Test exception.",
        "reported_amount": 12_500.00,
        "currency": "EUR",
    }
    defaults.update(overrides)
    return CloseException.model_validate(defaults)


@pytest.fixture
def entry() -> JournalEntry:
    return make_entry()


@pytest.fixture
def exception_record() -> CloseException:
    return make_exception()
