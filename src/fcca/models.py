"""Domain and decision contracts.

Two families of models live here:

* **Facts** (:class:`JournalEntry`, :class:`CloseException`) describe the ERP
  world. They are produced by the synthetic data generator and never by a model.
* **Judgements** (:class:`ControlSignal`, :class:`ControlDecision`,
  :class:`CaseResult`) describe what the system concluded, and always carry the
  evidence that supports the conclusion.

The separation matters for auditability: a reviewer must be able to tell at a
glance which fields are deterministic and which came out of a language model.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RiskLevel = Literal["low", "medium", "high"]
Severity = Literal["info", "warning", "critical"]

#: Closed vocabulary for the model's classification. Free-text classifications
#: would make evaluation and downstream routing impossible.
ExceptionClassification = Literal[
    "unusual_journal_entry",
    "late_manual_posting",
    "reconciliation_mismatch",
    "duplicate_posting",
    "material_variance",
    "missing_supporting_documentation",
    "out_of_hours_posting",
    "threshold_breach",
    "incomplete_reconciliation",
    "unexpected_account_cost_center",
    "no_finding",
]

#: Closed vocabulary for recommended remediation. Keeps the LLM out of the
#: business of inventing actions the organisation has no process for.
ActionCategory = Literal[
    "no_action",
    "request_supporting_documentation",
    "request_justification",
    "route_to_preparer",
    "route_to_reviewer",
    "escalate_to_financial_controller",
    "propose_correcting_entry",
    "refer_to_internal_audit",
]


class JournalEntry(BaseModel):
    """A single posted line from the synthetic ERP extract."""

    model_config = ConfigDict(frozen=True)

    journal_id: str
    company_code: str
    account: str
    account_name: str
    cost_center: str
    posting_date: date
    document_date: date
    posting_timestamp: datetime
    amount: float
    currency: str
    amount_reporting_ccy: float
    user_id: str
    document_type: str
    description: str
    manual_posting: bool
    supporting_document: str | None = None
    reconciliation_status: Literal["reconciled", "open", "in_progress", "not_applicable"]
    approved_by: str | None = None

    @property
    def days_to_post(self) -> int:
        """Calendar days between the document date and the posting date."""
        return (self.posting_date - self.document_date).days

    @property
    def has_supporting_document(self) -> bool:
        return bool(self.supporting_document)


class CloseException(BaseModel):
    """An exception raised by the close monitoring layer for one journal entry.

    In a real deployment these would arrive from a close-management or
    continuous-controls-monitoring tool; here they are generated from documented
    scenarios so the pipeline is reproducible.
    """

    model_config = ConfigDict(frozen=True)

    exception_id: str
    journal_id: str
    company_code: str
    exception_type: str
    detected_at: datetime
    close_period: str
    source_system: str
    description: str
    reported_amount: float
    currency: str


class ControlSignal(BaseModel):
    """Result of one deterministic control check.

    Signals are computed in Python/DuckDB, never by a model. They are the
    factual basis the model is asked to interpret, and they are stored verbatim
    in the audit trail.
    """

    model_config = ConfigDict(frozen=True)

    check_id: str
    name: str
    triggered: bool
    severity: Severity
    detail: str
    observed_value: int | float | str | None = None
    threshold: int | float | str | None = None

    def as_prompt_line(self) -> str:
        parts = [f"{self.check_id} ({self.name}): {'TRIGGERED' if self.triggered else 'clear'}"]
        if self.observed_value is not None:
            parts.append(f"observed={self.observed_value}")
        if self.threshold is not None:
            parts.append(f"threshold={self.threshold}")
        parts.append(self.detail)
        return " | ".join(str(p) for p in parts)


class PolicyEvidence(BaseModel):
    """A retrieved policy passage, with everything needed to find it again."""

    model_config = ConfigDict(frozen=True)

    document: str = Field(description="Policy document title, e.g. 'Journal Entry Policy'.")
    section: str = Field(description="Section reference, e.g. '4.2 Approval thresholds'.")
    passage: str = Field(description="Verbatim retrieved text.")
    score: float = Field(description="Retriever relevance score (higher is more relevant).")
    node_id: str = Field(description="Stable chunk id in the persisted policy index.")
    source_path: str = Field(description="Repository-relative path of the source document.")

    @property
    def citation(self) -> str:
        return f"{self.document} §{self.section_number}"

    @property
    def section_number(self) -> str:
        """The leading numeric part of the section heading, e.g. ``4.2``."""
        head = self.section.split(" ", 1)[0].strip().rstrip(".")
        return head or self.section

    @property
    def passage_sha256(self) -> str:
        """Hash of the passage, so an auditor can prove the text has not changed."""
        return hashlib.sha256(self.passage.encode("utf-8")).hexdigest()[:16]


class PolicyCitation(BaseModel):
    """A citation as returned by the model.

    Deliberately narrow: the model may only name a document and a section, never
    supply passage text. Passages come from the retriever, so quoted policy text
    can never be hallucinated. Citations are then grounded against the retrieved
    set (see :mod:`fcca.workflow.grounding`).
    """

    document: str
    section: str

    @field_validator("document", "section")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class ControlDecision(BaseModel):
    """The validated, structured decision for one exception.

    This is the model's contract. Anything the model returns that does not
    validate against this schema is rejected and retried once; a second failure
    fails the case safely into human review.
    """

    exception_id: str
    classification: ExceptionClassification
    risk_level: RiskLevel
    finding: str = Field(
        max_length=400,
        description="One or two sentences stating what happened, in control language.",
    )
    recommended_action: str = Field(
        max_length=300, description="Concrete next step for the close team."
    )
    action_category: ActionCategory
    requires_human_review: bool = Field(
        description="Model's own view. The final value is set by the review gate."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    rationale: str = Field(
        max_length=1200,
        description="Why this classification and action follow from the signals and policy.",
    )

    @model_validator(mode="after")
    def _no_action_must_be_low_risk(self) -> ControlDecision:
        """A 'no action' recommendation on a high-risk item is self-contradictory."""
        if self.action_category == "no_action" and self.risk_level == "high":
            raise ValueError("action_category 'no_action' is not permitted at risk_level 'high'")
        return self


class GateOutcome(BaseModel):
    """Result of the deterministic human-in-the-loop gate.

    The gate, not the model, decides whether a recommendation may be applied
    without a person looking at it.
    """

    model_config = ConfigDict(frozen=True)

    requires_human_review: bool
    disposition: Literal["auto_recommendation", "human_review"]
    reasons: list[str] = Field(
        default_factory=list, description="Why review was required; empty if auto-approved."
    )


class GroundingReport(BaseModel):
    """Whether the model's citations are supported by what was actually retrieved."""

    model_config = ConfigDict(frozen=True)

    total_citations: int
    grounded_citations: int
    ungrounded_citations: list[str] = Field(default_factory=list)

    @property
    def is_fully_grounded(self) -> bool:
        return not self.ungrounded_citations

    @property
    def has_any_evidence(self) -> bool:
        return self.grounded_citations > 0


class RunMetadata(BaseModel):
    """Everything needed to reproduce and cost a single model invocation."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    latency_ms: int
    prompt_sha256: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    parse_attempts: int = 1
    structured_output_mode: str = "json_schema_prompt"


class CaseResult(BaseModel):
    """The complete, self-contained outcome for one exception.

    Everything a reviewer needs is in this object: the facts, the deterministic
    signals, the retrieved evidence, the model's decision, the grounding check
    and the gate outcome.
    """

    exception: CloseException
    entry: JournalEntry
    signals: list[ControlSignal]
    evidence: list[PolicyEvidence]
    decision: ControlDecision
    grounding: GroundingReport
    gate: GateOutcome
    run: RunMetadata
    decided_at: datetime

    @property
    def triggered_signals(self) -> list[ControlSignal]:
        return [s for s in self.signals if s.triggered]

    @property
    def final_requires_human_review(self) -> bool:
        """Authoritative review flag: the gate always wins over the model."""
        return self.gate.requires_human_review


class CaseFailure(BaseModel):
    """A case the system could not decide.

    Recorded and surfaced rather than swallowed: an exception the automation
    failed on is an exception a person still has to look at.
    """

    exception_id: str
    stage: Literal["load", "controls", "retrieval", "inference", "validation"]
    error: str
    provider: str
    model: str
    gate: GateOutcome
    failed_at: datetime


class ReviewRecord(BaseModel):
    """A human reviewer's disposition of a case.

    The system produces recommendations; only a person accepts, rejects or
    escalates them. This record closes the loop and is appended to the audit log.
    """

    exception_id: str
    reviewer: str
    action: Literal["approved", "rejected", "escalated"]
    comment: str = ""
    reviewed_at: datetime


class LabelledCase(BaseModel):
    """One entry of the labelled evaluation set.

    Labels are derived by construction from the scenario used to synthesise the
    exception, then checked against the policy set. They are *not* human
    annotations of production data, and the README says so.
    """

    exception_id: str
    scenario: str
    expected_risk_level: RiskLevel
    expected_requires_human_review: bool
    expected_action_category: ActionCategory
    expected_policy_document: str
    notes: str = ""
