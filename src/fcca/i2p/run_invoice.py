"""Run the deterministic pipeline for one invoice and print the result.

The default view is the trace, because the trace is the product: what a reviewer
needs is the sequence of steps that produced the outcome, not the outcome on its
own.
"""

from __future__ import annotations

import argparse
import json

from fcca.i2p.resolver import InvoiceResolver
from fcca.shared.config import get_settings
from fcca.shared.trace import read_trace, render_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca i2p-run",
        description="Run the deterministic invoice-to-pay pipeline for one invoice.",
    )
    parser.add_argument("--invoice", required=True, help="Invoice id, e.g. INV-00001.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--provider", default=None, help="Model provider for the agent layer.")
    args = parser.parse_args(argv)

    settings = get_settings()
    resolver = InvoiceResolver.build(provider=args.provider, settings=settings)
    resolved = resolver.resolve(args.invoice)
    result = resolved.result

    if args.json:
        print(json.dumps(resolved.model_dump(mode="json"), indent=2))
        return 0

    print(f"{result.invoice_id}  {result.category}  {result.currency} {result.document_value:,.2f}")
    print(f"outcome: {result.primary_exception}  ({len(result.findings)} finding(s))")
    for finding in result.findings:
        location = f"line {finding.line_no}" if finding.line_no else "header"
        print(f"  {finding.rule_id}  {finding.severity:<6} {location:<8} {finding.detail}")

    if resolved.assessment is not None:
        assessment = resolved.assessment.assessment
        print()
        print(
            f"model      {resolved.assessment.run.provider}:{resolved.assessment.run.model}  "
            f"proposes {assessment.proposed_action} at confidence {assessment.confidence:.2f}"
        )
        for citation in assessment.evidence:
            print(f"  evidence {citation.field_path}")
    else:
        print("\nmodel      not called; no rule fired")

    print(f"\ntier       {resolved.routing.tier.upper()}")
    for reason in resolved.routing.reasons:
        print(f"  - {reason}")
    print()
    print(render_trace(read_trace(settings.i2p_trace_path, case_id=args.invoice)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
