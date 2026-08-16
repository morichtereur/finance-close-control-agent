"""Policy retrieval and evidence assembly.

Retrieval is lexical (BM25) rather than embedding-based. That is a deliberate
engineering choice, not a shortcut:

* the corpus is small, controlled and jargon-dense, and the decisive tokens are
  exact ones — *materiality*, *suspense*, *segregation*, *50,000*. Lexical
  matching handles those precisely;
* it requires no embedding provider, so the whole repository runs and is testable
  with no cloud account and no cost;
* it is deterministic, which makes retrieval itself unit-testable and makes the
  audit trail reproducible.

The retriever is a LlamaIndex ``BaseRetriever``, so swapping in a vector or hybrid
index (with a Bedrock or Vertex embedding model) is a constructor change in this
module and touches nothing downstream.

The query is not the raw exception text. It is *expanded from the deterministic
control signals* that fired: a structured-to-lexical bridge that keeps retrieval
anchored to what the controls actually found.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from fcca.config import Settings, get_settings
from fcca.models import CloseException, ControlSignal, JournalEntry, PolicyEvidence
from fcca.retrieval.index import load_policy_nodes

_TOKEN = re.compile(r"[a-z0-9§\.]+")

_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the to was were
    with this these those which will shall may must not no than then there their
    """.split()  # noqa: SIM905 - readable as prose, not as a 40-line list
)

#: Query expansion terms per control check. Keeping this table next to the
#: retriever (rather than inside each check) means the control layer stays free of
#: retrieval concerns.
CHECK_QUERY_TERMS: dict[str, str] = {
    "CHK-01": "supporting documentation missing unsupported manual entry escalated attachment",
    "CHK-02": "approval threshold second-level approval financial controller approver",
    "CHK-03": "segregation of duties preparer approver control failure",
    "CHK-04": "late posting timeliness justification document date delay",
    "CHK-05": "outside normal business hours posting window risk indicator",
    "CHK-06": "high-risk account manual entries reviewed management override",
    "CHK-07": "duplicate posting suspected duplicate correcting entry investigate",
    "CHK-08": "unusual entry round amount narrative justification estimate",
    "CHK-09": "description narrative independent reviewer business event",
    "CHK-10": "period integrity document date earlier period cut-off recognised",
    "CHK-11": "account cost center combination master data unexpected preparer confirmation",
    "CHK-12": "user does not normally post high-risk account unusual characteristics",
    "CHK-13": "group materiality escalation group accounting clearly trivial threshold",
    "CHK-14": "reconciliation status open in progress incomplete sign-off review",
    "CHK-15": "reconciling difference investigation threshold materiality write off ageing",
    "CHK-16": "suspense clearing account cleared zero month-end residual balance",
    "CHK-17": "variance movement month on month explanation escalation trigger",
    "CHK-18": "splitting entries approval threshold same posting date aggregated",
}

EXCEPTION_QUERY_TERMS: dict[str, str] = {
    "missing_supporting_documentation": "supporting documentation standard missing support",
    "late_manual_posting": "manual posting control late posting justification",
    "out_of_hours_posting": "manual posting outside business hours",
    "threshold_breach": "journal entry approval threshold breach",
    "duplicate_posting": "duplicate posting journal entry policy",
    "unusual_journal_entry": "unusual journal entry characteristics",
    "reconciliation_mismatch": "account reconciliation difference mismatch",
    "incomplete_reconciliation": "reconciliation incomplete open status",
    "material_variance": "materiality escalation variance",
    "unexpected_account_cost_center": "account cost center combination close policy",
    "no_finding": "close exceptions triage disposition",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer with a small stopword list."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class _BM25Model:
    """Minimal Okapi BM25 over the policy nodes.

    Implemented here rather than pulled in as a dependency: it is roughly thirty
    lines, it removes a compiled dependency from the install, and it keeps
    scoring transparent for an audit conversation.
    """

    k1: float = 1.5
    b: float = 0.75
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_freq: Counter[str] = field(default_factory=Counter)
    avg_len: float = 0.0

    def fit(self, corpus: list[str]) -> _BM25Model:
        self.doc_tokens = [tokenize(text) for text in corpus]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        lengths = [len(t) for t in self.doc_tokens] or [1]
        self.avg_len = sum(lengths) / len(lengths)
        return self

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        n_docs = len(self.doc_tokens) or 1
        scores: list[float] = []
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            length = len(tokens) or 1
            total = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += idf * (tf * (self.k1 + 1)) / denom
            scores.append(total)
        return scores


class PolicyRetriever(BaseRetriever):
    """LlamaIndex retriever over the persisted policy nodes."""

    def __init__(self, nodes: list[TextNode], top_k: int = 4) -> None:
        self._nodes = nodes
        self._top_k = top_k
        # Section headings carry decisive terms ("Suspense and clearing accounts"),
        # so they are indexed alongside the body text.
        corpus = [f"{n.metadata.get('section', '')}\n{n.text}" for n in nodes]
        self._model = _BM25Model().fit(corpus)
        super().__init__()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raw = self._model.score(query_bundle.query_str)
        best = max(raw) if raw else 0.0
        if best <= 0:
            return []
        ranked = sorted(
            (
                NodeWithScore(node=node, score=round(score / best, 4))
                for node, score in zip(self._nodes, raw, strict=True)
                if score > 0
            ),
            key=lambda nws: nws.score or 0.0,
            reverse=True,
        )
        return ranked[: self._top_k * 3]  # over-fetch; the service diversifies


@lru_cache(maxsize=4)
def _cached_retriever(index_dir: str, top_k: int) -> PolicyRetriever:
    settings = get_settings()
    return PolicyRetriever(load_policy_nodes(settings), top_k=top_k)


class PolicyRetrievalService:
    """Turns a case into ranked, attributable policy evidence."""

    def __init__(self, settings: Settings | None = None, retriever: BaseRetriever | None = None):
        self.settings = settings or get_settings()
        self._retriever = retriever or _cached_retriever(
            str(self.settings.index_dir), self.settings.retrieval_top_k
        )

    def build_query(
        self,
        exception: CloseException,
        entry: JournalEntry,
        signals: list[ControlSignal],
    ) -> str:
        """Expand the case into a retrieval query anchored on the fired controls."""
        parts: list[str] = [
            exception.exception_type.replace("_", " "),
            EXCEPTION_QUERY_TERMS.get(exception.exception_type, ""),
        ]
        for signal in signals:
            if signal.triggered:
                parts.append(signal.name.replace("_", " "))
                parts.append(CHECK_QUERY_TERMS.get(signal.check_id, ""))
        if entry.manual_posting:
            parts.append("manual posting journal entry")
        return " ".join(p for p in parts if p).strip()

    def retrieve(self, query: str) -> list[PolicyEvidence]:
        """Retrieve evidence for a raw query string."""
        results = self._retriever.retrieve(QueryBundle(query_str=query))
        return self._diversify(results)

    def retrieve_for_case(
        self,
        exception: CloseException,
        entry: JournalEntry,
        signals: list[ControlSignal],
    ) -> list[PolicyEvidence]:
        return self.retrieve(self.build_query(exception, entry, signals))

    def _diversify(self, results: list[NodeWithScore]) -> list[PolicyEvidence]:
        """Keep the best chunk per section and at most two sections per document.

        Without this, a single verbose policy can occupy every evidence slot and
        the reviewer never sees the second relevant standard.
        """
        seen_sections: set[tuple[str, str]] = set()
        per_document: Counter[str] = Counter()
        evidence: list[PolicyEvidence] = []

        for result in results:
            score = float(result.score or 0.0)
            if score < self.settings.retrieval_min_score:
                continue
            metadata = result.node.metadata
            document = str(metadata.get("document", "Unknown policy"))
            section = str(metadata.get("section", "0"))
            key = (document, section)
            if key in seen_sections or per_document[document] >= 2:
                continue
            seen_sections.add(key)
            per_document[document] += 1
            evidence.append(
                PolicyEvidence(
                    document=document,
                    section=section,
                    passage=result.node.get_content().strip(),
                    score=score,
                    node_id=result.node.node_id,
                    source_path=str(metadata.get("source_path", "")),
                )
            )
            if len(evidence) >= self.settings.retrieval_top_k:
                break
        return evidence


def clear_retriever_cache() -> None:
    """Drop the cached retriever (used after rebuilding the index, and by tests)."""
    _cached_retriever.cache_clear()
