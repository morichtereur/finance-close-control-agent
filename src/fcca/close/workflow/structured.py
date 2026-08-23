"""Structured output with validation, one repair attempt, and safe failure.

Two paths are supported and both produce a validated
:class:`~fcca.close.models.ControlDecision`:

``json_schema_prompt`` (default)
    Schema in the prompt, JSON extracted and validated with Pydantic. Portable:
    identical behaviour on every provider, including ones with no tool-calling
    support. Costs a little prompt budget and one occasional repair round trip.

``native_tools``
    Delegates to the provider's own structured-output API via LangChain's
    ``with_structured_output``. Usually more reliable per call, but the failure
    modes differ by provider, which is exactly what a portability claim has to
    survive.

The default is the portable path. A prototype that only works when the provider
has good tool calling has not demonstrated portability.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ValidationError

from fcca.close.workflow.prompts import repair_message
from fcca.shared.config import Settings, get_settings
from fcca.shared.errors import StructuredOutputError

logger = logging.getLogger(__name__)

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class StructuredInvocation:
    """Outcome of one structured call, including what it cost."""

    value: BaseModel
    raw_text: str
    attempts: int
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles fenced blocks and bare objects with trailing prose, which is the
    common real-world failure mode when a model is asked for JSON only.
    """
    fenced = _FENCED.search(text)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = text.find("{")
        if start == -1:
            raise StructuredOutputError("response contained no JSON object")
        depth = 0
        end = -1
        in_string = False
        escape = False
        for i, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise StructuredOutputError("response contained an unterminated JSON object")
        candidate = text[start:end]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError("response JSON was not an object")
    return parsed


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # provider-specific content blocks
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ]
        return "".join(parts)
    return str(content)


def _usage(message: BaseMessage) -> tuple[int | None, int | None]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    return None, None


def invoke_structured(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[BaseModel],
    settings: Settings | None = None,
) -> StructuredInvocation:
    """Call ``llm`` and return a validated instance of ``schema``.

    Raises:
        StructuredOutputError: the model did not produce a valid object within the
            configured number of attempts. Callers must treat this as a case that
            fails into human review, never as a pass.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    if settings.structured_output_mode == "native_tools":
        return _invoke_native(llm, messages, schema, started)

    conversation = list(messages)
    last_error = ""
    raw_text = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    for attempt in range(1, settings.max_parse_retries + 2):
        response = llm.invoke(conversation)
        raw_text = _message_text(response)
        attempt_in, attempt_out = _usage(response)
        input_tokens = _add(input_tokens, attempt_in)
        output_tokens = _add(output_tokens, attempt_out)
        try:
            payload = extract_json_object(raw_text)
            value = schema.model_validate(payload)
        except (StructuredOutputError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning("structured output attempt %d failed: %s", attempt, last_error)
            if attempt > settings.max_parse_retries:
                break
            conversation = [
                *conversation,
                AIMessage(content=raw_text),
                repair_message(last_error),
            ]
            continue
        return StructuredInvocation(
            value=value,
            raw_text=raw_text,
            attempts=attempt,
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    raise StructuredOutputError(
        f"model did not return a valid {schema.__name__} after "
        f"{settings.max_parse_retries + 1} attempt(s): {last_error}"
    )


def _invoke_native(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[BaseModel],
    started: float,
) -> StructuredInvocation:
    try:
        structured = llm.with_structured_output(schema)
        value = structured.invoke(messages)
    except NotImplementedError as exc:
        raise StructuredOutputError(
            f"{type(llm).__name__} does not support native structured output; "
            "set FCCA_STRUCTURED_OUTPUT_MODE=json_schema_prompt"
        ) from exc
    except ValidationError as exc:
        raise StructuredOutputError(f"native structured output failed validation: {exc}") from exc

    if not isinstance(value, schema):
        raise StructuredOutputError(
            f"native structured output returned {type(value).__name__}, expected {schema.__name__}"
        )
    return StructuredInvocation(
        value=value,
        raw_text=value.model_dump_json(),
        attempts=1,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _add(current: int | None, extra: int | None) -> int | None:
    if extra is None:
        return current
    return extra if current is None else current + extra
