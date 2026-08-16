"""Data generation reproducibility, the tool layer, and CLI smoke tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fcca.analytics import CloseAnalytics
from fcca.cli import main as cli_main
from fcca.config import Settings
from fcca.evaluation.benchmark import load_labels
from fcca.generate_data import SCENARIOS, generate
from fcca.retrieval.retriever import PolicyRetrievalService
from fcca.workflow.tools import build_control_tools, tool_catalogue


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_reproducible(settings: Settings, tmp_path: Path) -> None:
    """The same seed must produce byte-identical extracts."""
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    for target in (first, second):
        target.mkdir()
        (target / "policies").mkdir()
        generate(settings.model_copy(update={"base_dir": target}))
    assert _sha(first / "data" / "raw" / "journal_entries.csv") == _sha(
        second / "data" / "raw" / "journal_entries.csv"
    )


def test_a_different_seed_produces_different_data(settings: Settings, tmp_path: Path) -> None:
    base_a = tmp_path / "seed-a"
    base_b = tmp_path / "seed-b"
    base_a.mkdir()
    base_b.mkdir()
    generate(settings.model_copy(update={"base_dir": base_a, "random_seed": 1}))
    generate(settings.model_copy(update={"base_dir": base_b, "random_seed": 2}))
    assert _sha(base_a / "data" / "raw" / "journal_entries.csv") != _sha(
        base_b / "data" / "raw" / "journal_entries.csv"
    )


def test_every_scenario_is_represented_in_the_labelled_set(settings: Settings) -> None:
    labels = load_labels(settings)
    scenarios = {label.scenario for label in labels.values()}
    assert scenarios == {scenario.key for scenario in SCENARIOS}


def test_labels_cover_all_three_risk_levels_and_both_dispositions(settings: Settings) -> None:
    labels = list(load_labels(settings).values())
    assert {label.expected_risk_level for label in labels} == {"low", "medium", "high"}
    assert {label.expected_requires_human_review for label in labels} == {True, False}


def test_dataset_shape(settings: Settings) -> None:
    with CloseAnalytics(settings) as analytics:
        summary = analytics.dataset_summary()
    assert summary["entities"] == 5
    assert summary["close_exceptions"] == settings.n_exceptions
    assert summary["journal_entries"] >= settings.n_journal_entries


def test_account_numbers_stay_strings(settings: Settings) -> None:
    """A numeric-looking account must never become an integer in the database."""
    with CloseAnalytics(settings) as analytics:
        row = analytics.journal_entry(analytics.exceptions(limit=1)[0]["journal_id"])
    assert isinstance(row["account"], str)


# ------------------------------------------------------------------------ tools
def test_tools_are_typed_and_callable(settings: Settings) -> None:
    with CloseAnalytics(settings) as analytics:
        tools = build_control_tools(analytics, PolicyRetrievalService(settings), settings)
        names = {tool.name for tool in tools}
        assert names == {
            "calculate_materiality",
            "get_account_risk",
            "retrieve_policy",
            "check_document_support",
            "check_reconciliation_status",
            "calculate_variance",
        }
        assert all(entry["description"] for entry in tool_catalogue(tools))

        materiality = next(t for t in tools if t.name == "calculate_materiality")
        assert materiality.invoke({"amount_reporting_ccy": settings.materiality_group + 1})[
            "is_material"
        ]

        account_risk = next(t for t in tools if t.name == "get_account_risk")
        assert account_risk.invoke({"account": "610000"})["risk_rating"] == "high"
        assert account_risk.invoke({"account": "600000"})["risk_rating"] == "standard"


# -------------------------------------------------------------------------- CLI
def test_cli_info_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    assert cli_main(["info"]) == 0
    assert "Providers" in capsys.readouterr().out


def test_cli_run_case_emits_valid_json(settings: Settings, capsys) -> None:  # type: ignore[no-untyped-def]
    case = next(iter(load_labels(settings)))
    assert (
        cli_main(["run-case", "--exception", case, "--provider", "mock", "--json", "--no-audit"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"]["exception_id"] == case
    assert payload["gate"]["disposition"] in {"auto_recommendation", "human_review"}


def test_cli_run_case_renders_a_review(settings: Settings, capsys) -> None:  # type: ignore[no-untyped-def]
    case = next(iter(load_labels(settings)))
    assert cli_main(["run-case", "--exception", case, "--provider", "mock", "--no-audit"]) == 0
    out = capsys.readouterr().out
    assert "Deterministic controls" in out
    assert "Policy evidence" in out


def test_cli_rejects_an_unknown_command(capsys) -> None:  # type: ignore[no-untyped-def]
    assert cli_main(["frobnicate"]) == 2


def test_cli_reports_a_missing_exception_cleanly(settings: Settings) -> None:
    assert (
        cli_main(["run-case", "--exception", "EXC-9999", "--provider", "mock", "--no-audit"]) == 1
    )
