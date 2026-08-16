"""Model providers.

One factory, three adapters, one interface. See :mod:`fcca.providers.base`.
"""

from fcca.providers.base import (
    ProviderSpec,
    available_providers,
    describe_provider,
    get_llm,
)

__all__ = ["ProviderSpec", "available_providers", "describe_provider", "get_llm"]
