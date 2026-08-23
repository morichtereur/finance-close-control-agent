"""Meaningful, layer-specific exceptions.

Each layer raises its own error type so the workflow can decide whether a
failure is recoverable (retry / degrade) or must fail the case safely. A case
that cannot be decided is never silently dropped: it becomes a human-review
item.
"""

from __future__ import annotations


class FCCAError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(FCCAError):
    """Settings are missing or mutually inconsistent."""


class DataNotFoundError(FCCAError):
    """A requested journal entry, exception or dataset does not exist."""


class ProviderError(FCCAError):
    """A model provider could not be constructed or invoked.

    Raised instead of leaking a cloud SDK exception, so that callers do not
    need to import ``botocore`` or ``google.api_core`` to handle failures.
    """


class ProviderNotInstalledError(ProviderError):
    """The optional dependency for a cloud provider is not installed."""


class RetrievalError(FCCAError):
    """The policy index is missing or could not be queried."""


class StructuredOutputError(FCCAError):
    """The model did not return an output that validates against the schema."""


class AuditError(FCCAError):
    """The audit trail could not be written or read."""
