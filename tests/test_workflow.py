"""End-to-end workflow behaviour, including provider portability and failure."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fcca.close.analytics import CloseAnalytics
from fcca.close.evaluation.benchmark import load_labels
from fcca.close.models import CaseFailure, CaseResult
from fcca.close.retrieval.retriever import PolicyRetrievalService
from fcca.close.workflow.control_agent import ControlAgent
from fcca.shared.audit.logger import AuditLog
from fcca.shared.config import Settings
from fcca.shared.providers.base import ProviderSpec


def _scenario_case(settings: Settings, scenario: str) -> str:
    labels = load_labels(settings)
    for label in labels.values():
        if label.scenario == scenario:
            return label.exception_id
    raise AssertionError(f"no case generated for scenario {scenario!r}")


def test_high_risk_case_requires_human_review(settings: Settings) -> None:
    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as agent:
        result = agent.run(_scenario_case(settings, "unsupported_manual_material"))
    assert isinstance(result, CaseResult)
    assert result.decision.risk_level == "high"
    assert result.final_requires_human_review
    assert result.gate.disposition == "human_review"


def test_compliant_case_can_be_auto_recommended(settings: Settings) -> None:
    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as agent:
        result = agent.run(_scenario_case(settings, "compliant_flagged"))
    assert result.decision.risk_level == "low"
    assert not result.final_requires_human_review
    assert result.gate.disposition == "auto_recommendation"


def test_duplicate_case_is_detected_across_the_population(settings: Settings) -> None:
    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as agent:
        result = agent.run(_scenario_case(settings, "duplicate_posting"))
    duplicate = next(s for s in result.signals if s.check_id == "CHK-07")
    assert duplicate.triggered
    assert result.decision.action_category == "propose_correcting_entry"


def test_every_decision_carries_grounded_evidence(settings: Settings) -> None:
    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as agent:
        result = agent.run(_scenario_case(settings, "threshold_breach_material"))
    assert result.evidence
    assert result.grounding.is_fully_grounded
    cited = {(c.document, c.section) for c in result.decision.policy_citations}
    retrieved = {(e.document, e.section) for e in result.evidence}
    assert cited <= retrieved


def test_the_decision_is_reproducible(settings: Settings) -> None:
    case = _scenario_case(settings, "late_manual_posting")
    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as agent:
        first = agent.run(case)
        second = agent.run(case)
    assert first.decision.model_dump() == second.decision.model_dump()
    assert first.run.prompt_sha256 == second.run.prompt_sha256


# --------------------------------------------------------- provider portability
class _AlternateProviderModel(BaseChatModel):
    """Stands in for a cloud model: different class, same interface."""

    payload: str = ""

    @property
    def _llm_type(self) -> str:
        return "alternate"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.payload))])


def _agent_with(llm: BaseChatModel, settings: Settings, provider: str) -> ControlAgent:
    return ControlAgent(
        settings=settings,
        analytics=CloseAnalytics(settings),
        retrieval=PolicyRetrievalService(settings),
        llm=llm,
        spec=ProviderSpec(
            provider=provider,
            model="stand-in-model",
            supports_native_structured_output=True,
            installed=True,
            requires_credentials=False,
        ),
        audit=None,
    )


def test_workflow_runs_unchanged_against_a_different_provider(settings: Settings) -> None:
    """Swapping the chat model changes nothing above the provider boundary."""
    case = _scenario_case(settings, "unsupported_manual_material")

    with ControlAgent.build(provider="mock", settings=settings, with_audit=False) as reference:
        expected = reference.run(case)

    payload = json.dumps(
        {
            "exception_id": case,
            "classification": "missing_supporting_documentation",
            "risk_level": "high",
            "finding": "Unsupported manual entry above group materiality.",
            "recommended_action": "Escalate to the entity Financial Controller.",
            "action_category": "escalate_to_financial_controller",
            "requires_human_review": True,
            "confidence": 0.91,
            "policy_citations": [
                {
                    "document": expected.evidence[0].document,
                    "section": expected.evidence[0].section,
                }
            ],
            "rationale": "Missing documentation above materiality.",
        }
    )
    with _agent_with(_AlternateProviderModel(payload=payload), settings, "pretend-cloud") as agent:
        result = agent.run(case)

    assert result.run.provider == "pretend-cloud"
    assert result.run.model == "stand-in-model"
    assert result.decision.risk_level == "high"
    assert result.final_requires_human_review
    assert result.grounding.is_fully_grounded
    # Same deterministic evidence regardless of which model produced the decision.
    assert [e.node_id for e in result.evidence] == [e.node_id for e in expected.evidence]


def test_a_model_that_returns_prose_fails_into_human_review(settings: Settings) -> None:
    case = _scenario_case(settings, "compliant_flagged")
    with _agent_with(
        _AlternateProviderModel(payload="I think this one is fine, honestly."),
        settings,
        "pretend-cloud",
    ) as agent:
        result = agent.run_safe(case)
    assert isinstance(result, CaseFailure)
    assert result.stage == "validation"
    assert result.gate.requires_human_review


def test_a_failure_is_written_to_the_audit_trail(settings: Settings, tmp_path: Any) -> None:
    case = _scenario_case(settings, "compliant_flagged")
    audit = AuditLog(settings, path=tmp_path / "failure-audit.db")
    agent = _agent_with(_AlternateProviderModel(payload="nope"), settings, "pretend-cloud")
    agent.audit = audit
    try:
        agent.run_safe(case)
    finally:
        agent.close()
    record = AuditLog(settings, path=tmp_path / "failure-audit.db").reconstruct(case)
    assert record["decisions"][0]["status"] == "failed"
    assert record["decisions"][0]["human_review_required"] == 1
