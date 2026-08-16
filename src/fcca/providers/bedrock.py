"""AWS Bedrock adapter.

Uses ``ChatBedrockConverse`` from ``langchain-aws``. The Converse API is chosen
deliberately over the model-specific invoke APIs: it presents one request and
response shape across Bedrock model families, so changing ``BEDROCK_MODEL_ID``
from an Anthropic model to a Nova, Llama or Mistral model does not change any
code in this repository.

Credentials are never read from this project's configuration. Authentication uses
the standard AWS credential chain — environment variables, a shared credentials
file, an SSO profile, or an instance/task role. Nothing here can leak a key into
the audit log, because nothing here ever holds one.

**Status:** implemented and importable, but not executed against a live Bedrock
endpoint in this repository. Benchmark rows for Bedrock are marked ``not_run``
until someone runs them with their own account.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from fcca.config import Settings, get_settings
from fcca.errors import ProviderError, ProviderNotInstalledError

logger = logging.getLogger(__name__)


def build_bedrock_chat_model(
    settings: Settings | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Construct a Bedrock chat model.

    Args:
        settings: Configuration; defaults to the process settings.
        model_name: Bedrock model id. Any chat model enabled in the account works;
            inference-profile ids (``eu.``/``us.`` prefixed) are supported.
        **kwargs: Passed through to ``ChatBedrockConverse``.

    Raises:
        ProviderNotInstalledError: ``langchain-aws`` is not installed.
        ProviderError: the client could not be constructed.
    """
    settings = settings or get_settings()
    model_id = model_name or settings.bedrock_model_id

    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ProviderNotInstalledError(
            "AWS Bedrock support requires the optional dependency. Install it with:\n"
            '    pip install "finance-close-control-agent[bedrock]"'
        ) from exc

    params: dict[str, Any] = {
        "model": model_id,
        "region_name": settings.aws_region,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    params.update(kwargs)

    try:
        client = ChatBedrockConverse(**params)
    except Exception as exc:
        raise ProviderError(
            f"could not construct a Bedrock client for model {model_id!r} in region "
            f"{settings.aws_region!r}: {exc}. Check that the model is enabled for the "
            "account and that AWS credentials are configured."
        ) from exc

    logger.info("bedrock client ready: model=%s region=%s", model_id, settings.aws_region)
    return client
