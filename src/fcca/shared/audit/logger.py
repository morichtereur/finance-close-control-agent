"""Append-only audit trail in SQLite.

The question this module exists to answer is not "what did the system decide?"
but "what did the system know when it decided, and would the same inputs produce
the same answer today?".

Each record therefore stores the deterministic signals, the retrieved passages
with their node ids and relevance scores, the raw model output *before*
validation, the validated decision, the grounding result, the gate outcome, the
provider and model, latency, token usage where the provider reports it, and a
snapshot of the thresholds in force. :meth:`AuditLog.reconstruct` returns all of
it for one exception.

What is deliberately never written: credentials, environment variables, or any
configuration value that could carry a secret. Only
:meth:`Settings.public_snapshot` is persisted.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fcca import __version__
from fcca.shared.config import Settings, get_settings
from fcca.shared.errors import AuditError
from fcca.shared.models import ReviewRecord

if TYPE_CHECKING:
    # The audit log is written *about* a process module, never *by* one. The
    # close-module type is referenced for typing only and not imported at
    # runtime: nothing under ``shared/`` may depend on ``close/`` or ``i2p/``.
    from fcca.close.models import CaseResult

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              TEXT    NOT NULL,
    exception_id           TEXT    NOT NULL,
    journal_id             TEXT,
    status                 TEXT    NOT NULL,          -- decided | failed
    provider               TEXT    NOT NULL,
    model                  TEXT    NOT NULL,
    structured_output_mode TEXT,
    package_version        TEXT,
    code_revision          TEXT,
    policy_index_nodes     INTEGER,
    deterministic_checks   TEXT,                      -- JSON
    policy_evidence        TEXT,                      -- JSON
    llm_raw_output         TEXT,
    validated_decision     TEXT,                      -- JSON
    grounding              TEXT,                      -- JSON
    gate                   TEXT,                      -- JSON
    confidence             REAL,
    human_review_required  INTEGER,
    latency_ms             INTEGER,
    parse_attempts         INTEGER,
    input_tokens           INTEGER,
    output_tokens          INTEGER,
    estimated_cost_usd     REAL,
    prompt_sha256          TEXT,
    settings_snapshot      TEXT,                      -- JSON
    error                  TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_exception ON decisions(exception_id);

CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    exception_id  TEXT NOT NULL,
    reviewer      TEXT NOT NULL,
    action        TEXT NOT NULL,
    comment       TEXT,
    reviewed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_exception ON reviews(exception_id);
"""


@dataclass(frozen=True)
class AuditRecordSummary:
    """One line of the audit register."""

    id: int
    timestamp: str
    exception_id: str
    provider: str
    model: str
    status: str
    risk_level: str | None
    confidence: float | None
    human_review_required: bool
    latency_ms: int | None


def _code_revision() -> str:
    """Short git revision, so a decision can be tied to the code that made it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


class AuditLog:
    """SQLite-backed decision log."""

    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = path or self.settings.audit_db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._con = sqlite3.connect(str(self.path))
            self._con.row_factory = sqlite3.Row
            self._con.executescript(SCHEMA)
            self._con.commit()
        except sqlite3.Error as exc:
            raise AuditError(f"could not open the audit log at {self.path}: {exc}") from exc

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    # ------------------------------------------------------------------ write
    def record_case(self, result: CaseResult, policy_index_nodes: int | None = None) -> int:
        """Append a completed decision. Returns the audit record id."""
        payload = {
            "timestamp": result.decided_at.isoformat(),
            "exception_id": result.exception.exception_id,
            "journal_id": result.entry.journal_id,
            "status": "decided",
            "provider": result.run.provider,
            "model": result.run.model,
            "structured_output_mode": result.run.structured_output_mode,
            "package_version": __version__,
            "code_revision": _code_revision(),
            "policy_index_nodes": policy_index_nodes,
            "deterministic_checks": _dumps([s.model_dump() for s in result.signals]),
            "policy_evidence": _dumps(
                [
                    {
                        **item.model_dump(),
                        "passage_sha256": item.passage_sha256,
                    }
                    for item in result.evidence
                ]
            ),
            "llm_raw_output": None,  # set by record_case_with_raw
            "validated_decision": _dumps(result.decision.model_dump()),
            "grounding": _dumps(result.grounding.model_dump()),
            "gate": _dumps(result.gate.model_dump()),
            "confidence": result.decision.confidence,
            "human_review_required": int(result.final_requires_human_review),
            "latency_ms": result.run.latency_ms,
            "parse_attempts": result.run.parse_attempts,
            "input_tokens": result.run.input_tokens,
            "output_tokens": result.run.output_tokens,
            "estimated_cost_usd": result.run.estimated_cost_usd,
            "prompt_sha256": result.run.prompt_sha256,
            "settings_snapshot": _dumps(self.settings.public_snapshot()),
            "error": None,
        }
        return self._insert(payload)

    def record_case_with_raw(
        self, result: CaseResult, raw_output: str, policy_index_nodes: int | None = None
    ) -> int:
        """Append a decision together with the unvalidated model response."""
        record_id = self.record_case(result, policy_index_nodes)
        self._con.execute(
            "UPDATE decisions SET llm_raw_output = ? WHERE id = ?", (raw_output, record_id)
        )
        self._con.commit()
        return record_id

    def record_failure(
        self,
        exception_id: str,
        provider: str,
        model: str,
        error: str,
        journal_id: str | None = None,
        signals: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        raw_output: str | None = None,
    ) -> int:
        """Append a case that could not be decided.

        Failures are first-class audit records. An exception that the system could
        not assess must be visible in the register, not absent from it.
        """
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "exception_id": exception_id,
            "journal_id": journal_id,
            "status": "failed",
            "provider": provider,
            "model": model,
            "structured_output_mode": self.settings.structured_output_mode,
            "package_version": __version__,
            "code_revision": _code_revision(),
            "policy_index_nodes": None,
            "deterministic_checks": _dumps(signals or []),
            "policy_evidence": _dumps(evidence or []),
            "llm_raw_output": raw_output,
            "validated_decision": None,
            "grounding": None,
            "gate": _dumps(
                {
                    "requires_human_review": True,
                    "disposition": "human_review",
                    "reasons": [f"automated assessment failed: {error}"],
                }
            ),
            "confidence": None,
            "human_review_required": 1,
            "latency_ms": None,
            "parse_attempts": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "prompt_sha256": None,
            "settings_snapshot": _dumps(self.settings.public_snapshot()),
            "error": error,
        }
        return self._insert(payload)

    def record_review(self, review: ReviewRecord) -> int:
        """Append a human reviewer's disposition."""
        cursor = self._con.execute(
            "INSERT INTO reviews (exception_id, reviewer, action, comment, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                review.exception_id,
                review.reviewer,
                review.action,
                review.comment,
                review.reviewed_at.isoformat(),
            ),
        )
        self._con.commit()
        return int(cursor.lastrowid or 0)

    def _insert(self, payload: dict[str, Any]) -> int:
        columns = ", ".join(payload)
        placeholders = ", ".join("?" for _ in payload)
        try:
            cursor = self._con.execute(
                f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
                list(payload.values()),
            )
            self._con.commit()
        except sqlite3.Error as exc:
            raise AuditError(f"could not write the audit record: {exc}") from exc
        return int(cursor.lastrowid or 0)

    # ------------------------------------------------------------------- read
    def reconstruct(self, exception_id: str) -> dict[str, Any]:
        """Return everything recorded for one exception.

        This is the reviewer-facing question: *what information did the system use
        when it made this recommendation?*
        """
        rows = self._con.execute(
            "SELECT * FROM decisions WHERE exception_id = ? ORDER BY id", (exception_id,)
        ).fetchall()
        if not rows:
            raise AuditError(f"no audit record found for {exception_id!r}")
        reviews = self._con.execute(
            "SELECT * FROM reviews WHERE exception_id = ? ORDER BY id", (exception_id,)
        ).fetchall()
        return {
            "exception_id": exception_id,
            "decisions": [_row_to_dict(row) for row in rows],
            "reviews": [dict(row) for row in reviews],
        }

    def register(self, limit: int = 50) -> list[AuditRecordSummary]:
        """Most recent audit records, newest first."""
        rows = self._con.execute(
            "SELECT id, timestamp, exception_id, provider, model, status, validated_decision, "
            "confidence, human_review_required, latency_ms "
            "FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        summaries = []
        for row in rows:
            decision = json.loads(row["validated_decision"]) if row["validated_decision"] else {}
            summaries.append(
                AuditRecordSummary(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    exception_id=row["exception_id"],
                    provider=row["provider"],
                    model=row["model"],
                    status=row["status"],
                    risk_level=decision.get("risk_level"),
                    confidence=row["confidence"],
                    human_review_required=bool(row["human_review_required"]),
                    latency_ms=row["latency_ms"],
                )
            )
        return summaries

    def count(self) -> int:
        row = self._con.execute("SELECT count(*) AS n FROM decisions").fetchone()
        return int(row["n"]) if row else 0

    def export_jsonl(self, destination: Path) -> Path:
        """Export the full register as JSON Lines for downstream GRC tooling."""
        rows = self._con.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_row_to_dict(row), default=str) + "\n")
        return destination


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Expand the JSON columns so the record reads as one nested document."""
    data = dict(row)
    for key in (
        "deterministic_checks",
        "policy_evidence",
        "validated_decision",
        "grounding",
        "gate",
        "settings_snapshot",
    ):
        raw = data.get(key)
        if isinstance(raw, str) and raw:
            with contextlib.suppress(json.JSONDecodeError):
                data[key] = json.loads(raw)
    return data
