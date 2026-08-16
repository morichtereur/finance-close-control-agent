"""Command line interface.

    fcca generate-data
    fcca ingest-policies
    fcca run-case --exception EXC-0042 --provider mock
    fcca evaluate --provider mock
    fcca audit --exception EXC-0042
    fcca review --exception EXC-0042 --action approved --reviewer u.klein
    fcca info

Every subcommand is also runnable as a module, e.g.
``python -m fcca.run_case --exception EXC-0042``, which is what the repository
uses in CI and in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from fcca import __version__
from fcca.config import get_settings
from fcca.errors import FCCAError
from fcca.models import ReviewRecord
from fcca.presentation import console

SUBCOMMANDS = {
    "generate-data": "fcca.generate_data",
    "ingest-policies": "fcca.ingest_policies",
    "run-case": "fcca.run_case",
    "evaluate": "fcca.evaluate",
}


def _audit(argv: list[str]) -> int:
    from fcca.audit.logger import AuditLog

    parser = argparse.ArgumentParser(
        prog="fcca audit", description="Inspect the decision audit trail."
    )
    parser.add_argument("--exception", help="Reconstruct everything recorded for one exception.")
    parser.add_argument("--register", action="store_true", help="List recent audit records.")
    parser.add_argument("--limit", type=int, default=20, help="Rows for --register.")
    parser.add_argument("--export", help="Export the whole register to a JSON Lines file.")
    args = parser.parse_args(argv)

    con = console()
    settings = get_settings()
    with AuditLog(settings) as log:
        if args.export:
            from pathlib import Path

            path = log.export_jsonl(Path(args.export))
            con.print(f"exported {log.count()} record(s) to {path}")
            return 0
        if args.exception:
            record = log.reconstruct(args.exception)
            print(json.dumps(record, indent=2, default=str))
            return 0
        # Default view: the register.
        rows = log.register(limit=args.limit)
        if not rows:
            con.print("audit trail is empty; run a case first", style="dim")
            return 0
        from rich.table import Table

        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        for column in (
            "id",
            "timestamp",
            "exception",
            "provider:model",
            "risk",
            "conf",
            "review",
            "ms",
        ):
            table.add_column(column, no_wrap=True)
        for row in rows:
            table.add_row(
                str(row.id),
                row.timestamp[:19],
                row.exception_id,
                f"{row.provider}:{row.model}",
                (row.risk_level or row.status).upper(),
                f"{row.confidence:.2f}" if row.confidence is not None else "-",
                "yes" if row.human_review_required else "no",
                str(row.latency_ms if row.latency_ms is not None else "-"),
            )
        con.print(table)
    return 0


def _review(argv: list[str]) -> int:
    from fcca.audit.logger import AuditLog

    parser = argparse.ArgumentParser(
        prog="fcca review",
        description="Record a human reviewer's disposition of an exception.",
    )
    parser.add_argument("--exception", required=True)
    parser.add_argument("--action", required=True, choices=["approved", "rejected", "escalated"])
    parser.add_argument("--reviewer", required=True, help="Reviewer user id.")
    parser.add_argument("--comment", default="", help="Free-text reviewer comment.")
    args = parser.parse_args(argv)

    record = ReviewRecord(
        exception_id=args.exception,
        reviewer=args.reviewer,
        action=args.action,
        comment=args.comment,
        reviewed_at=datetime.now(UTC),
    )
    with AuditLog(get_settings()) as log:
        # Confirm a decision exists before recording a review of it.
        log.reconstruct(args.exception)
        review_id = log.record_review(record)
    console().print(
        f"recorded review {review_id}: {args.exception} {args.action} by {args.reviewer}"
    )
    return 0


def _info(argv: list[str]) -> int:
    from fcca.providers.base import available_providers, describe_provider

    parser = argparse.ArgumentParser(prog="fcca info", description="Show the active configuration.")
    parser.parse_args(argv)

    con = console()
    settings = get_settings()
    con.print(f"finance-close-control-agent {__version__}", style="bold")
    con.print()
    con.print("Providers", style="bold")
    for name, installed in available_providers().items():
        spec = describe_provider(name, settings=settings)  # type: ignore[arg-type]
        active = " (active)" if name == settings.llm_provider else ""
        state = "installed" if installed else "not installed"
        con.print(f"  {name:8s} {spec.model:52s} {state}{active}", style="dim")
    con.print()
    con.print("Thresholds", style="bold")
    for key, value in settings.public_snapshot().items():
        con.print(f"  {key:32s} {value}", style="dim")
    con.print()

    try:
        from fcca.analytics import CloseAnalytics

        with CloseAnalytics(settings) as analytics:
            con.print("Dataset", style="bold")
            for key, value in analytics.dataset_summary().items():
                con.print(f"  {key:32s} {value}", style="dim")
    except FCCAError as exc:
        con.print(f"Dataset  {exc}", style="yellow")
    con.print()

    try:
        from fcca.retrieval.index import policy_index_manifest

        manifest = policy_index_manifest(settings)
        con.print("Policy index", style="bold")
        con.print(
            f"  {manifest['node_count']} nodes from {len(manifest['documents'])} documents",  # type: ignore[arg-type]
            style="dim",
        )
    except FCCAError as exc:
        con.print(f"Policy index  {exc}", style="yellow")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    if argv[0] in {"-V", "--version"}:
        print(__version__)
        return 0

    command, rest = argv[0], argv[1:]
    try:
        if command == "audit":
            return _audit(rest)
        if command == "review":
            return _review(rest)
        if command == "info":
            return _info(rest)
        module_name = SUBCOMMANDS.get(command)
        if module_name is None:
            print(f"unknown command {command!r}\n", file=sys.stderr)
            print(__doc__, file=sys.stderr)
            return 2
        import importlib

        module = importlib.import_module(module_name)
        return int(module.main(rest))
    except FCCAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
