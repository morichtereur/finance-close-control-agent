"""Model providers.

One factory, three adapters, one interface. See :mod:`fcca.shared.providers.base`.
"""

from fcca.shared.providers.base import (
    ProviderSpec,
    available_providers,
    describe_provider,
    get_llm,
)

__all__ = ["ProviderSpec", "available_providers", "describe_provider", "get_llm"]
