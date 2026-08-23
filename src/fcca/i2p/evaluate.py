"""Run the invoice-to-pay pipeline over the whole labelled population and score it.

Writes ``results/i2p_evaluation_<provider>__<model>.json`` and an export of every
resolved invoice for the UI to read. The JSON is what the README quotes from, so
the table in the README cannot drift away from a run that actually happened.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fcca.i2p.evaluation import EvaluationReport, evaluate, render_report
from fcca.i2p.repository import I2PRepository
from fcca.i2p.resolver import InvoiceResolver, ResolvedInvoice
from fcca.shared.config import ProviderName, Settings, get_settings
from fcca.shared.providers.base import describe_provider


def run_slug(provider: str, model: str) -> str:
    """Filesystem-safe identifier for one provider/model pair."""
    return f"{provider}__{re.sub(r'[^A-Za-z0-9]+', '-', model).strip('-').lower()}"


def run_evaluation(
    provider: ProviderName | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
    with_extraction_noise: bool = False,
) -> tuple[EvaluationReport, list[ResolvedInvoice]]:
    """Run the pipeline over the labelled population and score it.

    ``with_extraction_noise`` swaps the synthetic source for a document source
    whose readings are degraded at the rates in ``config/thresholds.yaml``. It
    is off by default and the shipped metrics are measured without it, so the
    README's "structured JSON, no OCR" claim stays literally true. Turning it on
    is the demonstration: touchless rate falls, extraction_gated rises, and
    false_auto_post stays at zero — noise costs throughput, not safety.
    """
    settings = settings or get_settings()
    spec = describe_provider(provider, model_name, settings)
    repository = I2PRepository(settings)

    source = None
    if with_extraction_noise:
        from fcca.i2p.degrade import build_payloads
        from fcca.i2p.extraction import DocumentSource, SyntheticSource

        base = SyntheticSource(repository)
        source = DocumentSource(
            build_payloads(
                list(repository.invoices.values()),
                seed=settings.random_seed,
                dropout_rate=settings.i2p.extraction_dropout_rate,
                digit_confusion_rate=settings.i2p.extraction_digit_confusion_rate,
            ),
            base,
        )

    resolver = InvoiceResolver.build(
        provider=provider,
        model_name=model_name,
        settings=settings,
        repository=repository,
        source=source,
    )

    labels = {
        row["invoice_id"]: row["expected_exception"]
        for row in json.loads(Path(settings.i2p_labels_path).read_text(encoding="utf-8"))
    }

    invoice_ids = list(repository.invoices)
    if limit is not None:
        invoice_ids = invoice_ids[:limit]

    resolved = [resolver.resolve(invoice_id) for invoice_id in invoice_ids]
    report = evaluate(
        resolved,
        {invoice_id: labels[invoice_id] for invoice_id in invoice_ids},
        provider=spec.provider,
        model=spec.model,
        settings_snapshot={
            "price_tolerance_pct": settings.i2p.price_tolerance_pct,
            "price_tolerance_abs": settings.i2p.price_tolerance_abs,
            "quantity_tolerance_pct": settings.i2p.quantity_tolerance_pct,
            "gr_grace_days": settings.i2p.gr_grace_days,
            "auto_clear_max_value": settings.i2p.auto_clear_max_value,
            "max_ungrounded_citation_rate": settings.i2p.max_ungrounded_citation_rate,
            "auto_clear_min_confidence": settings.i2p.auto_clear_min_confidence,
            "propose_max_value": settings.i2p.propose_max_value,
            "duplicate_window_days": settings.i2p.duplicate_window_days,
            "random_seed": settings.random_seed,
        },
    )
    return report, resolved


def write_results(
    report: EvaluationReport,
    resolved: list[ResolvedInvoice],
    settings: Settings,
) -> tuple[Path, Path]:
    """Persist the report and the per-invoice detail the UI reads."""
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    slug = run_slug(report.provider, report.model)

    report_path = settings.results_dir / f"i2p_evaluation_{slug}.json"
    payload = report.model_dump(mode="json")
    # Derived properties are not fields; write them so a consumer need not
    # reimplement the arithmetic and risk disagreeing with it.
    payload["exact_agreement"] = report.exact_agreement
    payload["per_class"] = [
        {
            **metrics.model_dump(mode="json"),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        }
        for metrics in report.per_class
    ]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    detail_path = settings.i2p_results_path
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in resolved], indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return report_path, detail_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca i2p-evaluate",
        description="Score the invoice-to-pay pipeline against the labelled dataset.",
    )
    parser.add_argument("--provider", default=None, help="mock | bedrock | vertex")
    parser.add_argument("--model", default=None, help="Override the configured model id.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N invoices.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    report, resolved = run_evaluation(
        provider=args.provider, model_name=args.model, settings=settings, limit=args.limit
    )
    report_path, detail_path = write_results(report, resolved, settings)

    if not args.quiet:
        print(render_report(report))
        print()
        print(f"report written to {report_path}")
        print(f"per-invoice detail written to {detail_path}")

    # A nonzero false-auto-post count is a failure, not a metric. Exiting
    # nonzero means CI notices it even if nobody reads the table.
    if not report.is_safe:
        print(
            f"\nFAILED: {report.false_auto_post_count} invoice(s) were cleared without a "
            f"person that ground truth says were exceptions: "
            f"{', '.join(report.false_auto_post_ids)}",
        )
        return 1

    # An ungrounded citation never reaches a decision -- it is stripped first. The
    # limit is here because the rate is a measurement of the model, and a number
    # that is only ever printed is a number nobody notices moving.
    if not report.is_grounded:
        limit = float(report.settings_snapshot["max_ungrounded_citation_rate"])
        print(
            f"\nFAILED: {report.ungrounded_citation_invoices} of "
            f"{report.invoices_with_findings} assessed invoice(s) cited a field the model "
            f"was not given ({report.ungrounded_citation_rate:.1%}, limit {limit:.0%}). "
            f"The citations were stripped and routing is unaffected, but the rate is "
            f"outside what this configuration accepts.",
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
