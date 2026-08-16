"""Policy retrieval built on LlamaIndex.

LlamaIndex earns its place here for a specific reason: policy passages must be
addressable. A recommendation is only defensible if the reviewer can open the
exact section it rests on. The index therefore carries document title, section
reference and a stable node id through to the decision record.
"""

from fcca.retrieval.index import build_policy_index, load_policy_nodes, policy_index_manifest
from fcca.retrieval.retriever import PolicyRetrievalService, PolicyRetriever

__all__ = [
    "PolicyRetrievalService",
    "PolicyRetriever",
    "build_policy_index",
    "load_policy_nodes",
    "policy_index_manifest",
]
