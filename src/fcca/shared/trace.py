"""The append-only execution trace.

The trace is the product, not a debug log. Everything else in this repository
exists to produce a record that a reviewer — or an auditor six months later —
can read end to end and understand *why* an invoice or a journal entry ended up
where it did, without reading any Python.

That imposes three properties, and they are enforced here rather than left to
the discipline of callers:

**One record per step, not one per case.** A case-level record says what the
system concluded. A step-level record says how it got there, and it is the only
form in which a disagreement is diagnosable: if the price check and the
tolerance evaluation disagree, you need to see both, with the inputs each one
actually had.

**Every record names its actor.** ``rule`` means a deterministic function
computed this. ``model`` means a language model inferred it. ``human`` means a
person decided it. A trace where those are indistinguishable is not an audit
trail — it is a transcript. Records carry ``rule_id`` or
``model`` + ``prompt_version`` accordingly, and the validator below refuses a
record that claims one actor while carrying another's provenance.

**Append-only.** :class:`TraceWriter` opens for append and never seeks. There is
no update method and no delete method, because a trace you can edit is evidence
of nothing. Correcting a mistake means appending a record that says so.

The ``input_hash`` is what makes reproduction checkable: the same inputs to the
same step should produce the same hash, so a rerun that diverges is visible
rather than merely suspected.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fcca.shared.models import Actor

#: Which process module produced the record. The two modules share this file
#: format, so a mixed trace is readable without knowing which wrote it.
ModuleName = Literal["close", "i2p"]


def hash_input(payload: Any) -> str:
    """Stable short hash of a step's input.

    Canonical JSON with sorted keys, so a dict that happens to be built in a
    different order still hashes the same. Truncated to 16 hex characters: long
    enough that a collision is not a practical concern for a trace, short enough
    that a human can compare two of them by eye, which is what they are for.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TraceRecord(BaseModel):
    """One step of one pipeline run."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    case_id: str = Field(description="Invoice id or close-exception id this step ran for.")
    module: ModuleName
    step_name: str = Field(description="Pipeline step, e.g. 'price_check'.")
    actor: Actor
    input_hash: str = Field(description="Stable hash of the inputs this step received.")
    outcome: str = Field(description="Closed-vocabulary result, e.g. 'pass' / 'exception'.")
    summary: str = Field(
        max_length=300,
        description="One line a reviewer can read without opening anything else.",
    )

    # Provenance — exactly one of these two groups is populated, enforced below.
    rule_id: str | None = Field(
        default=None, description="Identifier of the deterministic rule that ran."
    )
    model: str | None = Field(default=None, description="Model id, for actor='model'.")
    prompt_version: str | None = Field(
        default=None, description="Prompt template version, for actor='model'."
    )

    #: Free-form structured payload — the numbers behind the summary. Kept out
    #: of the required schema so that the record stays readable when it is long.
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _provenance_matches_actor(self) -> Self:
        """A record may not claim one actor and carry another's provenance."""
        if self.actor == "rule":
            if not self.rule_id:
                raise ValueError("actor 'rule' requires a rule_id")
            if self.model or self.prompt_version:
                raise ValueError("actor 'rule' must not carry model provenance")
        elif self.actor == "model":
            if not (self.model and self.prompt_version):
                raise ValueError("actor 'model' requires both model and prompt_version")
            if self.rule_id:
                raise ValueError("actor 'model' must not carry a rule_id")
        elif self.rule_id or self.model or self.prompt_version:
            raise ValueError("actor 'human' must not carry rule or model provenance")
        return self

    @property
    def provenance(self) -> str:
        """Short human-readable attribution, for rendering one line of trace."""
        if self.actor == "rule":
            return self.rule_id or "rule"
        if self.actor == "model":
            return f"{self.model}@{self.prompt_version}"
        return "human"

    def to_line(self) -> str:
        """One JSONL line. Datetimes are ISO-8601; nothing is pretty-printed."""
        return json.dumps(self.model_dump(mode="json"), default=str, separators=(",", ":"))


class TraceWriter:
    """Append-only JSONL writer.

    Deliberately minimal: ``append`` and nothing else. There is no ``update``,
    no ``delete`` and no way to reopen a record, because the value of the trace
    depends on it being impossible to tidy up after the fact.
    """

    def __init__(self, path: Path, module: ModuleName) -> None:
        self.path = path
        self.module = module
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    @property
    def records_written(self) -> int:
        return self._count

    def append(self, record: TraceRecord) -> TraceRecord:
        """Append one record. Returns it, so callers can log and pass through."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_line() + "\n")
        self._count += 1
        return record

    def step(
        self,
        *,
        case_id: str,
        step_name: str,
        actor: Actor,
        inputs: Any,
        outcome: str,
        summary: str,
        rule_id: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """Build and append a record in one call — the form callers actually use."""
        return self.append(
            TraceRecord(
                timestamp=datetime.now(UTC),
                case_id=case_id,
                module=self.module,
                step_name=step_name,
                actor=actor,
                input_hash=hash_input(inputs),
                outcome=outcome,
                summary=summary,
                rule_id=rule_id,
                model=model,
                prompt_version=prompt_version,
                detail=detail or {},
            )
        )


@contextmanager
def trace_writer(path: Path, module: ModuleName) -> Iterator[TraceWriter]:
    """Context-manager form, for pipelines that want a scoped writer."""
    yield TraceWriter(path, module)


def read_trace(path: Path, case_id: str | None = None) -> list[TraceRecord]:
    """Read a trace file back, optionally filtered to one case.

    Reading is a first-class operation: a trace nothing can parse is a log file
    with extra steps. This is what the UI and the worked README example use.
    """
    if not path.exists():
        return []
    records: list[TraceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = TraceRecord.model_validate_json(line)
            if case_id is None or record.case_id == case_id:
                records.append(record)
    return records


def render_trace(records: list[TraceRecord]) -> str:
    """Fixed-width plain-text rendering, used by the CLI and the README example."""
    if not records:
        return "(no trace records)"
    width = max(len(r.step_name) for r in records)
    lines = []
    for record in records:
        lines.append(
            f"{record.step_name.ljust(width)}  {record.actor:<5}  "
            f"{record.outcome:<12}  {record.summary}"
        )
    return "\n".join(lines)


__all__ = [
    "ModuleName",
    "TraceRecord",
    "TraceWriter",
    "hash_input",
    "read_trace",
    "render_trace",
    "trace_writer",
]
