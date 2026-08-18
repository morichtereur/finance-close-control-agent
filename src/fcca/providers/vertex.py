"""Google Vertex AI adapter.

Uses ``ChatVertexAI`` from ``langchain-google-vertexai``. Authentication uses
Application Default Credentials (``gcloud auth application-default login``) or an
attached service account, so — as with Bedrock — no secret is ever held by this
project.

**Status:** implemented and importable, but not executed against a live Vertex
endpoint in this repository. Benchmark rows for Vertex are marked ``not_run``
until someone runs them with their own project.

**Deprecation, stated rather than hidden:** as of ``langchain-google-vertexai``
3.2.0, ``ChatVertexAI`` emits a deprecation warning and points at
``ChatGoogleGenerativeAI`` from ``langchain-google-genai``; removal is
scheduled for 4.0.0. It has deliberately not been swapped here. The two
differ in how they authenticate, and replacing one adapter that has never been
run against a live endpoint with another that has never been run against a
live endpoint is not an improvement — it only moves the untested surface. The
migration belongs with the first real Vertex run, where it can be verified in
the same pass.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from fcca.config import Settings, get_settings
from fcca.errors import ConfigurationError, ProviderError, ProviderNotInstalledError

logger = logging.getLogger(__name__)


def build_vertex_chat_model(
    settings: Settings | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Construct a Vertex AI chat model.

    Args:
        settings: Configuration; defaults to the process settings.
        model_name: Vertex model name, e.g. a Gemini model available to the project.
        **kwargs: Passed through to ``ChatVertexAI``.

    Raises:
        ConfigurationError: ``GOOGLE_CLOUD_PROJECT`` is not set.
        ProviderNotInstalledError: ``langchain-google-vertexai`` is not installed.
        ProviderError: the client could not be constructed.
    """
    settings = settings or get_settings()
    model = model_name or settings.vertex_model_name

    if not settings.google_cloud_project:
        raise ConfigurationError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT to be set (see .env.example)."
        )

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ProviderNotInstalledError(
            "Vertex AI support requires the optional dependency. Install it with:\n"
            '    pip install "finance-close-control-agent[vertex]"'
        ) from exc

    params: dict[str, Any] = {
        "model_name": model,
        "project": settings.google_cloud_project,
        "location": settings.vertex_location,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_tokens,
    }
    params.update(kwargs)

    try:
        client = ChatVertexAI(**params)
    except Exception as exc:
        raise ProviderError(
            f"could not construct a Vertex AI client for model {model!r} in "
            f"{settings.vertex_location!r}: {exc}. Check that the Vertex AI API is enabled "
            "and that Application Default Credentials are configured."
        ) from exc

    logger.info(
        "vertex client ready: model=%s project=%s location=%s",
        model,
        settings.google_cloud_project,
        settings.vertex_location,
    )
    return client
