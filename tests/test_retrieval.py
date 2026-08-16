"""Policy retrieval: does the right section come back, with a usable citation?"""

from __future__ import annotations

from datetime import datetime

from fcca.config import Settings
from fcca.controls import journal_checks as jc
from fcca.retrieval.index import load_policy_nodes, policy_index_manifest
from fcca.retrieval.retriever import PolicyRetrievalService, tokenize
from tests.conftest import make_entry, make_exception


def test_index_covers_every_policy_document(settings: Settings) -> None:
    manifest = policy_index_manifest(settings)
    titles = {d["title"] for d in manifest["documents"]}  # type: ignore[index]
    assert titles == {
        "Journal Entry Policy",
        "Manual Posting Control",
        "Account Reconciliation Policy",
        "Month-End Close Policy",
        "Materiality and Escalation Policy",
        "Supporting Documentation Standard",
    }
    assert int(manifest["node_count"]) > 30  # type: ignore[arg-type]


def test_nodes_carry_document_and_section_metadata(settings: Settings) -> None:
    nodes = load_policy_nodes(settings)
    assert all(node.metadata.get("document") for node in nodes)
    assert all(node.metadata.get("section") for node in nodes)
    assert all(node.metadata.get("source_path", "").startswith("policies/") for node in nodes)


def test_missing_documentation_case_retrieves_the_documentation_standard(
    settings: Settings,
) -> None:
    service = PolicyRetrievalService(settings)
    entry = make_entry(manual_posting=True, supporting_document=None, amount=300_000.0)
    signals = [
        jc.check_supporting_document(entry, settings),
        jc.check_approval_threshold(entry, settings),
    ]
    evidence = service.retrieve_for_case(make_exception(), entry, signals)
    documents = {item.document for item in evidence}
    assert "Supporting Documentation Standard" in documents


def test_out_of_hours_case_retrieves_the_manual_posting_control(settings: Settings) -> None:
    service = PolicyRetrievalService(settings)
    entry = make_entry(posting_timestamp=datetime(2026, 7, 14, 2, 30))
    signals = [jc.check_business_hours(entry, settings)]
    evidence = service.retrieve_for_case(
        make_exception(exception_type="out_of_hours_posting"), entry, signals
    )
    assert "Manual Posting Control" in {item.document for item in evidence}


def test_suspense_query_finds_the_clearing_account_section(settings: Settings) -> None:
    service = PolicyRetrievalService(settings)
    evidence = service.retrieve("suspense and clearing account must be cleared to zero month-end")
    top = evidence[0]
    assert top.document == "Account Reconciliation Policy"
    assert top.section_number == "5"


def test_evidence_is_attributable_and_verifiable(settings: Settings) -> None:
    service = PolicyRetrievalService(settings)
    item = service.retrieve("second-level approval threshold journal entry")[0]
    assert item.node_id.startswith("pol-")
    assert item.passage.strip()
    assert 0.0 < item.score <= 1.0
    assert len(item.passage_sha256) == 16
    assert item.citation.startswith(item.document)


def test_evidence_is_diversified_across_documents(settings: Settings) -> None:
    service = PolicyRetrievalService(settings)
    evidence = service.retrieve("escalation materiality threshold approval documentation")
    per_document: dict[str, int] = {}
    for item in evidence:
        per_document[item.document] = per_document.get(item.document, 0) + 1
    assert max(per_document.values()) <= 2
    assert len(evidence) <= settings.retrieval_top_k


def test_retrieval_is_deterministic(settings: Settings) -> None:
    service = PolicyRetrievalService(settings)
    first = [e.node_id for e in service.retrieve("late posting justification")]
    second = [e.node_id for e in service.retrieve("late posting justification")]
    assert first == second


def test_tokenizer_keeps_numeric_policy_terms() -> None:
    tokens = tokenize("Entries at or above EUR 50,000 require approval")
    assert "50" in tokens and "000" in tokens
    assert "or" not in tokens  # stopword
