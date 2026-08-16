"""Provider factory.

``get_llm`` is the only place in the codebase that knows a cloud exists. Every
other module works against ``langchain_core.language_models.BaseChatModel``, so
switching from a local stub to Bedrock to Vertex AI is a configuration change,
not a code change — and that claim is enforced by a test rather than asserted in
a README.

Cloud SDKs are imported lazily inside their adapters. Installing this package
without the ``[bedrock]`` or ``[vertex]`` extras leaves the mock path fully
functional, which is what keeps the repository free to run.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from fcca.config import ProviderName, Settings, get_settings
from fcca.errors import ProviderError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    """What is recorded in the audit trail about the model that decided a case."""

    provider: str
    model: str
    supports_native_structured_output: bool
    installed: bool
    requires_credentials: bool

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


#: Optional dependency that must be importable for each cloud adapter.
_REQUIRED_MODULE: dict[str, str] = {
    "mock": "fcca.providers.mock",
    "bedrock": "langchain_aws",
    "vertex": "langchain_google_vertexai",
}


def _build_mock(settings: Settings, model_name: str, **kwargs: Any) -> BaseChatModel:
    from fcca.providers.mock import MockChatModel

    return MockChatModel(model_name=model_name, **kwargs)


def _build_bedrock(settings: Settings, model_name: str, **kwargs: Any) -> BaseChatModel:
    from fcca.providers.bedrock import build_bedrock_chat_model

    return build_bedrock_chat_model(settings, model_name, **kwargs)


def _build_vertex(settings: Settings, model_name: str, **kwargs: Any) -> BaseChatModel:
    from fcca.providers.vertex import build_vertex_chat_model

    return build_vertex_chat_model(settings, model_name, **kwargs)


_REGISTRY: dict[str, Callable[..., BaseChatModel]] = {
    "mock": _build_mock,
    "bedrock": _build_bedrock,
    "vertex": _build_vertex,
}


def available_providers() -> dict[str, bool]:
    """Map provider name to whether its dependency is importable in this environment."""
    return {
        name: importlib.util.find_spec(module) is not None
        for name, module in _REQUIRED_MODULE.items()
    }


def describe_provider(
    provider: ProviderName | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> ProviderSpec:
    """Describe a provider without constructing a client (no credentials needed)."""
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    if provider not in _REGISTRY:
        raise ProviderError(f"unknown provider {provider!r}; expected one of {sorted(_REGISTRY)}")
    return ProviderSpec(
        provider=provider,
        model=model_name or settings.model_name_for(provider),
        # Both cloud adapters expose tool calling; the deterministic stub does not.
        supports_native_structured_output=provider in {"bedrock", "vertex"},
        installed=available_providers()[provider],
        requires_credentials=provider != "mock",
    )


def get_llm(
    provider: ProviderName | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model for ``provider``.

    Args:
        provider: ``mock``, ``bedrock`` or ``vertex``. Defaults to ``LLM_PROVIDER``.
        model_name: Overrides the configured model id for that provider.
        settings: Settings override, mainly for tests.
        **kwargs: Passed through to the underlying integration.

    Raises:
        ProviderError: unknown provider, missing optional dependency, or the
            client could not be constructed.
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    builder = _REGISTRY.get(provider)
    if builder is None:
        raise ProviderError(f"unknown provider {provider!r}; expected one of {sorted(_REGISTRY)}")

    resolved_model = model_name or settings.model_name_for(provider)
    logger.debug("constructing provider=%s model=%s", provider, resolved_model)
    return builder(settings, resolved_model, **kwargs)
