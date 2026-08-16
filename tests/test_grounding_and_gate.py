"""Citation grounding and the human-in-the-loop gate."""

from __future__ import annotations

from fcca.config import Settings
from fcca.controls import journal_checks as jc
from fcca.models import ControlDecision, GroundingReport, PolicyEvidence
from fcca.workflow.gate import apply_gate, failed_case_gate
from fcca.workflow.grounding import ground_citations
from tests.conftest import make_entry


def _evidence() -> list[PolicyEvidence]:
    return [
        PolicyEvidence(
            document="Supporting Documentation Standard",
            section="4.2 Escalation",
            passage="Where supporting documentation is missing ...",
            score=1.0,
            node_id="pol-aaa",
            source_path="policies/supporting_documentation_standard.md",
        ),
        PolicyEvidence(
            document="Materiality and Escalation Policy",
            section="3.1 Mandatory escalation",
            passage="The following are escalated ...",
            score=0.8,
            node_id="pol-bbb",
            source_path="policies/materiality_and_escalation_policy.md",
        ),
    ]


def _decision(**overrides: object) -> ControlDecision:
    payload: dict[str, object] = {
        "exception_id": "EXC-0001",
        "classification": "missing_supporting_documentation",
        "risk_level": "medium",
        "finding": "Unsupported manual entry.",
        "recommended_action": "Request the supporting document.",
        "action_category": "request_supporting_documentation",
        "requires_human_review": False,
        "confidence": 0.9,
        "policy_citations": [{"document": "Supporting Documentation Standard", "section": "4.2"}],
        "rationale": "CHK-01 fired.",
    }
    payload.update(overrides)
    return ControlDecision.model_validate(payload)


# ------------------------------------------------------------------- grounding
def test_retrieved_citation_is_grounded() -> None:
    cleaned, report = ground_citations(_decision(), _evidence())
    assert report.is_fully_grounded
    assert len(cleaned.policy_citations) == 1


def test_section_reference_formats_are_normalised() -> None:
    decision = _decision(
        policy_citations=[
            {"document": "supporting documentation standard", "section": "§4.2 Escalation"}
        ]
    )
    _, report = ground_citations(decision, _evidence())
    assert report.is_fully_grounded


def test_invented_citation_is_stripped_and_reported() -> None:
    decision = _decision(
        policy_citations=[
            {"document": "Supporting Documentation Standard", "section": "4.2"},
            {"document": "Group Fraud Policy", "section": "9.9"},
        ]
    )
    cleaned, report = ground_citations(decision, _evidence())
    assert [c.document for c in cleaned.policy_citations] == ["Supporting Documentation Standard"]
    assert report.ungrounded_citations == ["Group Fraud Policy §9.9"]
    assert not report.is_fully_grounded


def test_citation_to_a_section_that_was_not_retrieved_is_stripped() -> None:
    decision = _decision(
        policy_citations=[{"document": "Supporting Documentation Standard", "section": "7.1"}]
    )
    cleaned, report = ground_citations(decision, _evidence())
    assert cleaned.policy_citations == []
    assert report.grounded_citations == 0


# ------------------------------------------------------------------------ gate
def _grounded(n: int = 1) -> GroundingReport:
    return GroundingReport(total_citations=n, grounded_citations=n, ungrounded_citations=[])


def test_confident_low_risk_case_with_evidence_is_auto_recommended(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(risk_level="low", action_category="no_action", confidence=0.9),
        _grounded(),
        [jc.check_supporting_document(make_entry(), settings)],
        settings,
    )
    assert outcome.disposition == "auto_recommendation"
    assert not outcome.requires_human_review


def test_high_risk_always_requires_review(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(risk_level="high", confidence=0.99, requires_human_review=False),
        _grounded(),
        [],
        settings,
    )
    assert outcome.requires_human_review


def test_low_confidence_requires_review(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(risk_level="low", action_category="no_action", confidence=0.4),
        _grounded(),
        [],
        settings,
    )
    assert outcome.requires_human_review


def test_missing_evidence_requires_review(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(risk_level="low", action_category="no_action", policy_citations=[]),
        GroundingReport(total_citations=0, grounded_citations=0, ungrounded_citations=[]),
        [],
        settings,
    )
    assert outcome.requires_human_review


def test_ungrounded_citation_requires_review(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(risk_level="low", action_category="no_action"),
        GroundingReport(
            total_citations=2, grounded_citations=1, ungrounded_citations=["Made Up Policy §1"]
        ),
        [],
        settings,
    )
    assert outcome.requires_human_review


def test_deterministic_trigger_overrides_a_confident_clearance(settings: Settings) -> None:
    """The control layer wins. A model cannot clear a mandatory escalation."""
    unsupported = make_entry(manual_posting=True, supporting_document=None, amount=20_000.0)
    signals = [jc.check_supporting_document(unsupported, settings)]
    outcome = apply_gate(
        _decision(
            risk_level="low",
            action_category="no_action",
            confidence=1.0,
            requires_human_review=False,
        ),
        _grounded(),
        signals,
        settings,
    )
    assert outcome.requires_human_review
    assert any("mandatory escalation" in reason for reason in outcome.reasons)


def test_consequential_actions_always_involve_a_person(settings: Settings) -> None:
    outcome = apply_gate(
        _decision(
            risk_level="low",
            action_category="refer_to_internal_audit",
            confidence=0.99,
        ),
        _grounded(),
        [],
        settings,
    )
    assert outcome.requires_human_review


def test_a_failed_case_is_never_an_automatic_pass() -> None:
    outcome = failed_case_gate("model returned prose")
    assert outcome.requires_human_review
    assert outcome.disposition == "human_review"
