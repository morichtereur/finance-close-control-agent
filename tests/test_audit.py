"""The audit trail must be complete, reconstructable and free of secrets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fcca.close.evaluation.benchmark import load_labels
from fcca.close.models import ReviewRecord
from fcca.close.workflow.control_agent import ControlAgent
from fcca.shared.audit.logger import AuditLog
from fcca.shared.config import Settings
from fcca.shared.errors import AuditError

REQUIRED_FIELDS = (
    "timestamp",
    "exception_id",
    "provider",
    "model",
    "deterministic_checks",
    "policy_evidence",
    "validated_decision",
    "grounding",
    "gate",
    "confidence",
    "human_review_required",
    "latency_ms",
    "prompt_sha256",
    "settings_snapshot",
)

SECRET_MARKERS = ("key", "secret", "token", "password", "credential")


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture
def recorded_case(settings: Settings, audit_path: Path) -> str:
    case = next(iter(load_labels(settings)))
    agent = ControlAgent.build(provider="mock", settings=settings, with_audit=False)
    agent.audit = AuditLog(settings, path=audit_path)
    try:
        agent.run(case)
    finally:
        agent.close()
    return case


def test_every_required_field_is_recorded(
    settings: Settings, audit_path: Path, recorded_case: str
) -> None:
    with AuditLog(settings, path=audit_path) as log:
        record = log.reconstruct(recorded_case)["decisions"][0]
    for field in REQUIRED_FIELDS:
        assert record[field] is not None, f"audit record is missing {field}"


def test_reconstruction_returns_the_evidence_that_was_used(
    settings: Settings, audit_path: Path, recorded_case: str
) -> None:
    with AuditLog(settings, path=audit_path) as log:
        record = log.reconstruct(recorded_case)["decisions"][0]
    evidence = record["policy_evidence"]
    assert evidence, "no policy evidence was preserved"
    for item in evidence:
        assert item["document"] and item["section"] and item["node_id"]
        assert item["source_path"].startswith("policies/")
        assert len(item["passage_sha256"]) == 16
    checks = record["deterministic_checks"]
    assert len(checks) >= 15
    assert all("check_id" in c and "triggered" in c for c in checks)


def test_thresholds_in_force_are_recorded(
    settings: Settings, audit_path: Path, recorded_case: str
) -> None:
    with AuditLog(settings, path=audit_path) as log:
        snapshot = log.reconstruct(recorded_case)["decisions"][0]["settings_snapshot"]
    assert snapshot["materiality_group"] == settings.materiality_group
    assert snapshot["auto_approve_min_confidence"] == settings.auto_approve_min_confidence


def test_no_secret_looking_configuration_is_persisted(
    settings: Settings, audit_path: Path, recorded_case: str
) -> None:
    with AuditLog(settings, path=audit_path) as log:
        snapshot = log.reconstruct(recorded_case)["decisions"][0]["settings_snapshot"]
    for key in snapshot:
        assert not any(marker in key.lower() for marker in SECRET_MARKERS)


def test_reviews_close_the_loop(settings: Settings, audit_path: Path, recorded_case: str) -> None:
    with AuditLog(settings, path=audit_path) as log:
        log.record_review(
            ReviewRecord(
                exception_id=recorded_case,
                reviewer="u.klein",
                action="approved",
                comment="Documentation provided after posting; retained.",
                reviewed_at=datetime.now(UTC),
            )
        )
        record = log.reconstruct(recorded_case)
    assert record["reviews"][0]["reviewer"] == "u.klein"
    assert record["reviews"][0]["action"] == "approved"


def test_unknown_exception_cannot_be_reconstructed(settings: Settings, audit_path: Path) -> None:
    with AuditLog(settings, path=audit_path) as log, pytest.raises(AuditError):
        log.reconstruct("EXC-9999")


def test_register_and_export(
    settings: Settings, audit_path: Path, recorded_case: str, tmp_path: Path
) -> None:
    with AuditLog(settings, path=audit_path) as log:
        rows = log.register(limit=10)
        assert rows and rows[0].exception_id == recorded_case
        destination = log.export_jsonl(tmp_path / "audit.jsonl")
    assert destination.read_text(encoding="utf-8").count("\n") == log_count(settings, audit_path)


def log_count(settings: Settings, audit_path: Path) -> int:
    with AuditLog(settings, path=audit_path) as log:
        return log.count()
