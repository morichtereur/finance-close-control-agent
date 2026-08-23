"""Assess a single close exception and print the control review."""

from __future__ import annotations

import argparse
import logging
import sys

from fcca.close.models import CaseFailure
from fcca.close.presentation import console, render_case, render_failure
from fcca.shared.config import get_settings
from fcca.shared.errors import FCCAError

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca run-case",
        description="Assess one close exception and print the control review.",
    )
    parser.add_argument("--exception", required=True, help="Exception id, e.g. EXC-0042.")
    parser.add_argument(
        "--provider",
        choices=["mock", "bedrock", "vertex"],
        default=None,
        help="Model provider; defaults to LLM_PROVIDER.",
    )
    parser.add_argument("--model", default=None, help="Override the configured model id.")
    parser.add_argument(
        "--all-signals", action="store_true", help="Show every control check, not only triggered."
    )
    parser.add_argument("--json", action="store_true", help="Print the case result as JSON.")
    parser.add_argument("--no-audit", action="store_true", help="Do not write an audit record.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from fcca.close.workflow.control_agent import (
        ControlAgent,  # imported late for CLI start-up time
    )

    settings = get_settings()
    try:
        agent = ControlAgent.build(
            provider=args.provider,
            model_name=args.model,
            settings=settings,
            with_audit=not args.no_audit,
        )
    except FCCAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    con = console()
    try:
        result = agent.run_safe(args.exception)
    finally:
        agent.close()

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0 if not isinstance(result, CaseFailure) else 1

    if isinstance(result, CaseFailure):
        render_failure(result, con)
        return 1
    render_case(result, con, show_all_signals=args.all_signals)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
