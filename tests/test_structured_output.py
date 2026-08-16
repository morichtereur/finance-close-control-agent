"""Structured output: valid objects parse, invalid ones fail safely."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from fcca.config import Settings
from fcca.errors import StructuredOutputError
from fcca.models import ControlDecision
from fcca.workflow.structured import extract_json_object, invoke_structured

VALID_DECISION = {
    "exception_id": "EXC-0001",
    "classification": "missing_supporting_documentation",
    "risk_level": "high",
    "finding": "Unsupported manual entry above materiality.",
    "recommended_action": "Escalate to the entity Financial Controller.",
    "action_category": "escalate_to_financial_controller",
    "requires_human_review": True,
    "confidence": 0.9,
    "policy_citations": [{"document": "Supporting Documentation Standard", "section": "4.2"}],
    "rationale": "CHK-01 and CHK-13 both fired.",
}


class ScriptedChatModel(BaseChatModel):
    """Returns a fixed list of responses, one per call."""

    responses: list[str] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _messages() -> list[BaseMessage]:
    return [HumanMessage(content="assess this case")]


# ------------------------------------------------------------------ extraction
def test_extracts_a_fenced_json_block() -> None:
    text = 'Here you go:\n```json\n{"a": 1}\n```\nthanks'
    assert extract_json_object(text) == {"a": 1}


def test_extracts_a_bare_object_with_trailing_prose() -> None:
    text = '{"a": {"b": 2}} — hope that helps'
    assert extract_json_object(text) == {"a": {"b": 2}}


def test_braces_inside_strings_do_not_confuse_the_extractor() -> None:
    text = '{"note": "use {curly} braces"} trailing'
    assert extract_json_object(text) == {"note": "use {curly} braces"}


def test_missing_json_raises_a_typed_error() -> None:
    with pytest.raises(StructuredOutputError):
        extract_json_object("I would rather explain it in prose.")


# ------------------------------------------------------------------ validation
def test_valid_response_produces_a_decision(settings: Settings) -> None:
    llm = ScriptedChatModel(responses=[json.dumps(VALID_DECISION)])
    result = invoke_structured(llm, _messages(), ControlDecision, settings)
    assert isinstance(result.value, ControlDecision)
    assert result.attempts == 1


def test_invalid_response_is_repaired_on_the_second_attempt(settings: Settings) -> None:
    llm = ScriptedChatModel(responses=["not json at all", json.dumps(VALID_DECISION)])
    result = invoke_structured(llm, _messages(), ControlDecision, settings)
    assert isinstance(result.value, ControlDecision)
    assert result.attempts == 2


def test_persistently_invalid_response_fails_rather_than_guessing(settings: Settings) -> None:
    llm = ScriptedChatModel(responses=["still not json"])
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, _messages(), ControlDecision, settings)


def test_out_of_vocabulary_values_are_rejected(settings: Settings) -> None:
    bad = {**VALID_DECISION, "risk_level": "catastrophic"}
    llm = ScriptedChatModel(responses=[json.dumps(bad)])
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, _messages(), ControlDecision, settings)


def test_confidence_outside_the_unit_interval_is_rejected(settings: Settings) -> None:
    bad = {**VALID_DECISION, "confidence": 1.4}
    llm = ScriptedChatModel(responses=[json.dumps(bad)])
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, _messages(), ControlDecision, settings)


def test_self_contradictory_decision_is_rejected(settings: Settings) -> None:
    """'No action' on a high-risk item must not validate."""
    bad = {**VALID_DECISION, "action_category": "no_action", "risk_level": "high"}
    llm = ScriptedChatModel(responses=[json.dumps(bad)])
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, _messages(), ControlDecision, settings)
