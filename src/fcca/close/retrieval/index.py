"""Build and persist the policy index.

Pipeline: markdown file -> LlamaIndex ``Document`` -> section-aware nodes ->
``SentenceSplitter`` for long sections -> persisted ``SimpleDocumentStore``.

Section awareness is the point. A generic character splitter would happily cut
"entries at or above EUR 50,000 require documented second-level approval" in
half, and a citation to a mid-sentence chunk is useless to a reviewer. Splitting
on numbered headings first means every node inherits a real, quotable reference
such as *Journal Entry Policy §4.2*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from fcca.shared.config import Settings, get_settings
from fcca.shared.errors import RetrievalError

logger = logging.getLogger(__name__)

DOCSTORE_FILE = "policy_docstore.json"
MANIFEST_FILE = "manifest.json"

_H1 = re.compile(r"^#\s+(.*)$")
_H2_H3 = re.compile(r"^(#{2,3})\s+(.*)$")


@dataclass(frozen=True)
class Section:
    """One addressable passage of a policy document."""

    document: str
    section: str
    text: str
    source_path: str


def _read_documents(settings: Settings) -> list[Document]:
    """Load every markdown policy file as a LlamaIndex ``Document``."""
    policy_dir = settings.policies_dir
    paths = sorted(policy_dir.glob("*.md"))
    if not paths:
        raise RetrievalError(f"no policy documents found in {policy_dir}")

    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        title = _document_title(text, fallback=path.stem.replace("_", " ").title())
        documents.append(
            Document(
                text=text,
                doc_id=path.stem,
                metadata={
                    "document": title,
                    "source_path": str(path.relative_to(settings.base_dir)),
                    "policy_id": path.stem,
                },
            )
        )
    return documents


def _document_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = _H1.match(line.strip())
        if match:
            return match.group(1).strip()
    return fallback


def split_into_sections(document: Document) -> list[Section]:
    """Split a markdown policy into numbered sections.

    Headings at level 2 and 3 open a new section. Level-3 headings (``### 4.2``)
    are kept as their own sections because the numbered sub-clause is exactly the
    granularity a control reviewer cites.
    """
    title = str(document.metadata.get("document", "Policy"))
    source_path = str(document.metadata.get("source_path", ""))
    sections: list[Section] = []
    current_heading = "0 Preamble"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        # Drop the standalone "illustrative policy" blockquote that opens each file:
        # it is repository framing, not policy content, and would pollute retrieval.
        if body.startswith(">") and len(body) < 250:
            return
        sections.append(Section(title, current_heading, body, source_path))

    for raw_line in document.text.splitlines():
        line = raw_line.rstrip()
        if _H1.match(line.strip()):
            continue
        match = _H2_H3.match(line.strip())
        if match:
            flush()
            buffer = []
            current_heading = match.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return sections


def _to_nodes(sections: list[Section], settings: Settings) -> list[TextNode]:
    """Convert sections into nodes, splitting only those that are too long."""
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes: list[TextNode] = []
    for section in sections:
        chunks = splitter.split_text(section.text)
        for position, chunk in enumerate(chunks):
            node_id = _node_id(section, position)
            nodes.append(
                TextNode(
                    id_=node_id,
                    text=chunk,
                    metadata={
                        "document": section.document,
                        "section": section.section,
                        "source_path": section.source_path,
                        "chunk_index": position,
                        "chunk_count": len(chunks),
                    },
                    excluded_embed_metadata_keys=["chunk_index", "chunk_count"],
                )
            )
    return nodes


def _node_id(section: Section, position: int) -> str:
    digest = hashlib.sha256(
        f"{section.source_path}|{section.section}|{position}".encode()
    ).hexdigest()[:12]
    return f"pol-{digest}"


def build_policy_index(settings: Settings | None = None) -> dict[str, object]:
    """Build the policy index and persist it under ``data/processed/policy_index``.

    Returns a manifest describing what was indexed, which is also written to disk
    so a decision can be tied to the exact policy version it was made against.
    """
    settings = settings or get_settings()
    documents = _read_documents(settings)
    sections = [section for document in documents for section in split_into_sections(document)]
    nodes = _to_nodes(sections, settings)

    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(persist_path=str(index_dir / DOCSTORE_FILE))

    manifest = {
        "documents": [
            {
                "title": str(d.metadata["document"]),
                "source_path": str(d.metadata["source_path"]),
                "sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest()[:16],
            }
            for d in documents
        ],
        "section_count": len(sections),
        "node_count": len(nodes),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("indexed %d nodes from %d policy documents", len(nodes), len(documents))
    return manifest


def load_policy_nodes(settings: Settings | None = None) -> list[TextNode]:
    """Load the persisted policy nodes."""
    settings = settings or get_settings()
    path = settings.index_dir / DOCSTORE_FILE
    if not path.exists():
        raise RetrievalError(f"policy index not found at {path}. Run `fcca ingest-policies` first.")
    docstore = SimpleDocumentStore.from_persist_path(str(path))
    nodes = [node for node in docstore.docs.values() if isinstance(node, TextNode)]
    if not nodes:
        raise RetrievalError(f"policy index at {path} contains no nodes")
    return nodes


def policy_index_manifest(settings: Settings | None = None) -> dict[str, object]:
    """Return the manifest written when the index was built."""
    settings = settings or get_settings()
    path: Path = settings.index_dir / MANIFEST_FILE
    if not path.exists():
        raise RetrievalError(f"policy index manifest not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
