"""Export a resolved run as static JSON for the review interface.

The interface is a static site. It reads files; it does not call this codebase,
and there is no server between them. That is a deliberate consequence of the
non-goals: with no authentication and no multi-tenancy, a live API would be an
unauthenticated endpoint serving finance documents, and the honest version of
"no auth" is "no endpoint".

Two shapes are written:

``queue.json``
    One row per invoice — enough to render and sort the queue, and nothing more.

``invoices/<invoice_id>.json``
    Everything one review needs: the source document as received, the vendor
    master record it was checked against, the deterministic comparisons with the
    naive figure alongside the normalised one, the model's assessment where
    there was one, the routing decision with all its reasons, and the full trace.

The trace is exported verbatim rather than summarised. A reviewer who cannot see
the steps is being asked to trust the outcome, which is the thing this
repository is arguing against.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from fcca.i2p.evaluate import run_evaluation
from fcca.i2p.repository import I2PRepository
from fcca.i2p.resolver import ResolvedInvoice
from fcca.shared.config import ProviderName, Settings, get_settings
from fcca.shared.trace import read_trace

#: Default destination: inside the UI project, so `npm run build` picks it up
#: without a copy step and the site cannot be built against stale data.
DEFAULT_UI_DATA_DIR = "ui/data"


def queue_row(item: ResolvedInvoice, repository: I2PRepository) -> dict[str, Any]:
    invoice = repository.invoice(item.invoice_id)
    vendor = repository.vendor(invoice.vendor_id)
    finding = next(
        (f for f in item.result.findings if f.exception_type == item.result.primary_exception),
        None,
    )
    return {
        "invoice_id": item.invoice_id,
        "vendor_id": invoice.vendor_id,
        "vendor_name": vendor.name if vendor else invoice.vendor_id,
        "category": item.result.category,
        "currency": item.result.currency,
        "document_value": item.result.document_value,
        "invoice_date": invoice.invoice_date.isoformat(),
        "received_date": invoice.received_date.isoformat(),
        "vendor_reference": invoice.vendor_reference,
        "exception_type": item.result.primary_exception,
        "severity": finding.severity if finding else None,
        "findings": len(item.result.findings),
        "tier": item.routing.tier,
        "tier_before_model": item.result.routing.tier,
        "model_called": item.model_called,
        "confidence": (
            item.assessment.assessment.confidence if item.assessment is not None else None
        ),
        "lines": len(invoice.lines),
    }


def invoice_detail(
    item: ResolvedInvoice, repository: I2PRepository, settings: Settings
) -> dict[str, Any]:
    invoice = repository.invoice(item.invoice_id)
    vendor = repository.vendor(invoice.vendor_id)

    purchase_orders = {}
    for line in invoice.lines:
        if line.po_id and line.po_id not in purchase_orders:
            order = repository.purchase_order(line.po_id)
            if order is not None:
                purchase_orders[line.po_id] = order.model_dump(mode="json")

    receipts: list[dict[str, Any]] = []
    for line in invoice.lines:
        if line.po_id and line.po_line is not None:
            receipts += [
                r.model_dump(mode="json") for r in repository.receipts_for(line.po_id, line.po_line)
            ]

    trace = [
        record.model_dump(mode="json")
        for record in read_trace(settings.i2p_trace_path, case_id=item.invoice_id)
    ]

    return {
        "invoice_id": item.invoice_id,
        "invoice": invoice.model_dump(mode="json"),
        "vendor": vendor.model_dump(mode="json") if vendor else None,
        "purchase_orders": purchase_orders,
        "goods_receipts": receipts,
        "result": item.result.model_dump(mode="json"),
        "routing": item.routing.model_dump(mode="json"),
        "assessment": (
            item.assessment.model_dump(mode="json") if item.assessment is not None else None
        ),
        "model_called": item.model_called,
        # The payload that would post, for auto_clear invoices only. None
        # everywhere else, which is the point: it exists exactly where no person
        # is going to look at the invoice.
        "posting": item.posting.model_dump(mode="json") if item.posting is not None else None,
        "trace": trace,
    }


def export(
    destination: Path,
    provider: ProviderName | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the pipeline and write the static dataset the interface reads."""
    settings = settings or get_settings()

    # Start from an empty trace so the exported trace is this run's and not an
    # accumulation of every run since the file was created. The trace file
    # itself stays append-only; what changes here is which file we are writing.
    if settings.i2p_trace_path.exists():
        settings.i2p_trace_path.unlink()

    report, resolved = run_evaluation(provider=provider, settings=settings)
    repository = I2PRepository(settings)

    invoices_dir = destination / "invoices"
    if invoices_dir.exists():
        shutil.rmtree(invoices_dir)
    invoices_dir.mkdir(parents=True, exist_ok=True)

    rows = [queue_row(item, repository) for item in resolved]
    _write(destination / "queue.json", rows)

    for item in resolved:
        _write(
            invoices_dir / f"{item.invoice_id}.json",
            invoice_detail(item, repository, settings),
        )

    summary = report.model_dump(mode="json")
    summary["exact_agreement"] = report.exact_agreement
    summary["per_class"] = [
        {
            **metrics.model_dump(mode="json"),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "is_absent": metrics.is_absent,
        }
        for metrics in report.per_class
    ]
    _write(destination / "evaluation.json", summary)

    return {
        "invoices": len(rows),
        "destination": str(destination),
        "touchless_rate": report.touchless_rate,
        "false_auto_post_count": report.false_auto_post_count,
    }


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca i2p-export",
        description="Export a resolved run as static JSON for the review interface.",
    )
    parser.add_argument("--out", default=DEFAULT_UI_DATA_DIR, help="Destination directory.")
    parser.add_argument("--provider", default=None, help="mock | bedrock | vertex")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    destination = Path(args.out)
    if not destination.is_absolute():
        destination = settings.base_dir / destination

    summary = export(destination, provider=args.provider, settings=settings)
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
