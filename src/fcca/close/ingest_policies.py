"""Build the policy index from ``policies/*.md``."""

from __future__ import annotations

import argparse
import logging

from fcca.close.retrieval.index import build_policy_index
from fcca.close.retrieval.retriever import clear_retriever_cache
from fcca.shared.config import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca ingest-policies",
        description="Chunk, index and persist the finance policy knowledge base.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    settings = get_settings()
    manifest = build_policy_index(settings)
    clear_retriever_cache()

    documents = manifest["documents"]
    assert isinstance(documents, list)
    if not args.quiet:
        for document in documents:
            print(
                f"  {document['title']:42s} {document['source_path']}  sha256:{document['sha256']}"
            )
    print(
        f"indexed {manifest['node_count']} nodes from {len(documents)} documents "
        f"-> {settings.index_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
