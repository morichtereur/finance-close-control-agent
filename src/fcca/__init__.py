"""Finance Close Control Agent.

An evidence-based, auditable control assistant for month-end close exceptions.

The package is deliberately layered:

``config``      central settings; no model id or threshold is hard-coded elsewhere
``models``      Pydantic domain and decision contracts
``analytics``   deterministic DuckDB queries over synthetic ERP data
``controls``    deterministic finance control checks producing typed signals
``retrieval``   LlamaIndex policy index and retriever (source attribution)
``providers``   provider-agnostic ``get_llm`` factory (mock / Bedrock / Vertex)
``workflow``    LangChain orchestration, review gate, citation grounding
``audit``       append-only decision log and reconstruction
``evaluation``  labelled dataset metrics and the multi-provider benchmark
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
