"""Provider abstraction.

The central claim of this repository is that the workflow does not know which
cloud it is talking to. These tests are what make that a claim rather than a
sentence in a README.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from fcca.shared.config import Settings
from fcca.shared.errors import ConfigurationError, ProviderError, ProviderNotInstalledError
from fcca.shared.providers.base import available_providers, describe_provider, get_llm


def test_mock_provider_returns_a_langchain_chat_model(settings: Settings) -> None:
    llm = get_llm("mock", settings=settings)
    assert isinstance(llm, BaseChatModel)


def test_unknown_provider_is_rejected(settings: Settings) -> None:
    with pytest.raises(ProviderError):
        get_llm("azure", settings=settings)  # type: ignore[arg-type]


def test_provider_can_be_described_without_credentials(settings: Settings) -> None:
    for provider in ("mock", "bedrock", "vertex"):
        spec = describe_provider(provider, settings=settings)  # type: ignore[arg-type]
        assert spec.model
        assert spec.requires_credentials == (provider != "mock")


def test_model_id_comes_only_from_settings(settings: Settings) -> None:
    assert settings.model_name_for("bedrock") == settings.bedrock_model_id
    assert settings.model_name_for("vertex") == settings.vertex_model_name
    assert settings.model_name_for("mock") == settings.mock_model_name


def test_model_can_be_overridden_per_call(settings: Settings) -> None:
    spec = describe_provider("bedrock", model_name="some.other.model-v1", settings=settings)
    assert spec.model == "some.other.model-v1"


def test_cloud_providers_fail_with_a_typed_error_when_unavailable(settings: Settings) -> None:
    """Absent dependency or absent configuration must not leak an SDK exception."""
    available = available_providers()

    if not available["bedrock"]:
        with pytest.raises(ProviderNotInstalledError):
            get_llm("bedrock", settings=settings)

    # Vertex validates configuration before it imports its SDK, so an
    # unconfigured environment surfaces ConfigurationError first. Either way the
    # caller sees an fcca error, never a google.* or botocore exception.
    without_project = settings.model_copy(update={"google_cloud_project": None})
    with pytest.raises((ConfigurationError, ProviderNotInstalledError)):
        get_llm("vertex", settings=without_project)

    if available["vertex"]:
        with pytest.raises(ConfigurationError):
            get_llm("vertex", settings=without_project)


def test_mock_provider_is_deterministic(settings: Settings) -> None:
    from langchain_core.messages import HumanMessage

    llm = get_llm("mock", settings=settings)
    prompt = [HumanMessage(content='```json\n{"exception_id": "EXC-0001"}\n```')]
    assert llm.invoke(prompt).content == llm.invoke(prompt).content
