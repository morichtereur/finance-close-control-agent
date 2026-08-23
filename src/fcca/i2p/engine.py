"""The deterministic invoice-to-pay pipeline.

Twelve named steps, in a fixed order, each appending one trace record:

.. code-block:: text

    intake -> classification -> duplicate_check -> master_data_resolution
           -> tax_code -> gl_derivation -> cost_center_derivation
           -> three_way_match -> quantity_check -> price_check
           -> tolerance_evaluation -> routing_decision

The order is fixed because in a control process the sequence of checks *is* the
control design. Every invoice receives the same checks in the same order, or the
population is not comparable and the exception queue is not evidence of
anything. Nothing here plans its own path, and nothing here calls a model.

Why the steps are this granular: a step that both computes and decides produces
one trace record for two acts, and when the two disagree there is nothing to
read. ``price_check`` produces a residual; ``tolerance_evaluation`` decides what
the residual means against a configured limit. They are separate records because
they can be wrong separately — the first for an arithmetic reason, the second
for a policy one.

The routing decision at the end is deterministic here and stays deterministic
when the agent layer is added: a model's opinion becomes one more input to a
rule, never the rule itself.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fcca.i2p import checks
from fcca.i2p.models import (
    EXCEPTION_PRECEDENCE,
    ExceptionFinding,
    Invoice,
    InvoiceLine,
    InvoiceResult,
    LineResolution,
    PriceComparison,
    PurchaseOrderLine,
    QuantityComparison,
)
from fcca.i2p.repository import I2PRepository
from fcca.shared.config import Settings, get_settings
from fcca.shared.routing import route
from fcca.shared.trace import TraceWriter

logger = logging.getLogger(__name__)

#: Rule identifiers for the orchestration steps themselves, as distinct from the
#: checks they call. A trace record must name something a reader can look up.
STEP_RULES: dict[str, str] = {
    "intake": "I2P-S-01",
    "classification": "I2P-S-02",
    "duplicate_check": "I2P-S-03",
    "master_data_resolution": "I2P-S-04",
    "tax_code": "I2P-S-05",
    "gl_derivation": "I2P-S-06",
    "cost_center_derivation": "I2P-S-07",
    "three_way_match": "I2P-S-08",
    "quantity_check": "I2P-S-09",
    "price_check": "I2P-S-10",
    "tolerance_evaluation": "I2P-S-11",
    "routing_decision": "I2P-S-12",
}


class InvoiceEngine:
    """Runs the deterministic pipeline for one invoice."""

    def __init__(
        self,
        repository: I2PRepository | None = None,
        settings: Settings | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or I2PRepository(self.settings)
        self.trace = trace or TraceWriter(self.settings.i2p_trace_path, module="i2p")
        self.config = self.settings.i2p

    # ------------------------------------------------------------------ steps
    def _step(
        self,
        case_id: str,
        step_name: str,
        inputs: Any,
        outcome: str,
        summary: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.trace.step(
            case_id=case_id,
            step_name=step_name,
            actor="rule",
            rule_id=STEP_RULES[step_name],
            inputs=inputs,
            outcome=outcome,
            summary=summary,
            detail=detail or {},
        )

    def run(self, invoice_id: str) -> InvoiceResult:
        """Assess one invoice end to end."""
        config = self.config
        repo = self.repository
        findings: list[ExceptionFinding] = []

        # ------------------------------------------------------------ intake
        invoice = repo.invoice(invoice_id)
        self._step(
            invoice_id,
            "intake",
            {"invoice_id": invoice_id},
            "loaded",
            (
                f"Received {invoice.invoice_id} from {invoice.vendor_id}, "
                f"{invoice.currency} {invoice.stated_total_gross:,.2f}, "
                f"{len(invoice.lines)} line(s)."
            ),
            {
                "vendor_reference": invoice.vendor_reference,
                "invoice_date": invoice.invoice_date.isoformat(),
                "received_date": invoice.received_date.isoformat(),
                "gross": invoice.stated_total_gross,
            },
        )

        # ---------------------------------------------------- classification
        category, reason = checks.classify_invoice(invoice)
        self._step(
            invoice_id,
            "classification",
            [line.po_id for line in invoice.lines],
            category,
            f"Classified as {category}. {reason}",
        )

        # ----------------------------------------------------- bank details
        # Run early and unconditionally. A payment-redirection attempt on an
        # otherwise clean invoice must not depend on a later step firing.
        vendor = repo.vendor(invoice.vendor_id)
        bank_finding = checks.check_bank_details(invoice, vendor)
        if bank_finding:
            findings.append(bank_finding)

        # --------------------------------------------------- duplicate check
        earlier = repo.earlier_invoices(invoice, config.duplicate_window_days)
        duplicates, duplicate_finding = checks.check_duplicate(invoice, earlier, config)
        if duplicate_finding:
            findings.append(duplicate_finding)
        self._step(
            invoice_id,
            "duplicate_check",
            {
                "vendor_reference": invoice.vendor_reference,
                "gross": invoice.stated_total_gross,
                "window_days": config.duplicate_window_days,
            },
            "duplicate" if duplicates else "unique",
            (
                f"Matches {len(duplicates)} earlier invoice(s): {', '.join(duplicates)}."
                if duplicates
                else f"No match among {len(earlier)} earlier invoice(s) from this vendor."
            ),
            {"candidates": duplicates, "compared_against": len(earlier)},
        )

        # ------------------------------------------------------- line passes
        resolutions: list[LineResolution] = []
        # Quantities already invoiced against a PO line by *this* invoice, so a
        # document with two lines against one order line cannot claim the same
        # received quantity twice.
        consumed: dict[tuple[str, int], float] = {}

        materials: dict[int, str | None] = {}
        tax_codes: dict[int, str | None] = {}
        gl_accounts: dict[int, tuple[str | None, str]] = {}
        cost_centers: dict[int, tuple[str | None, str]] = {}
        price_comparisons: dict[int, PriceComparison] = {}
        quantity_comparisons: dict[int, QuantityComparison] = {}

        # -------------------------------------------- master data resolution
        for line in invoice.lines:
            material_id, finding = checks.resolve_material(line)
            materials[line.line_no] = material_id
            if finding:
                findings.append(finding)
        resolved = sum(1 for v in materials.values() if v)
        self._step(
            invoice_id,
            "master_data_resolution",
            [line.supplier_item_no for line in invoice.lines],
            "resolved" if resolved == len(invoice.lines) else "partial",
            (
                f"Resolved {resolved} of {len(invoice.lines)} supplier item number(s) "
                "to material master."
            ),
            {"materials": materials},
        )

        # ----------------------------------------------------------- tax code
        for line in invoice.lines:
            tax_code, finding = checks.resolve_tax_code(line)
            tax_codes[line.line_no] = tax_code
            if finding:
                findings.append(finding)
        self._step(
            invoice_id,
            "tax_code",
            [line.tax_rate for line in invoice.lines],
            "resolved" if all(tax_codes.values()) else "unresolved",
            "Tax codes: "
            + ", ".join(f"line {no}: {code or 'unresolved'}" for no, code in tax_codes.items())
            + ".",
            {"tax_codes": tax_codes},
        )

        # ------------------------------------------------------ GL derivation
        for line in invoice.lines:
            account, source, finding = checks.derive_gl_account(line, materials[line.line_no])
            gl_accounts[line.line_no] = (account, source)
            if finding:
                findings.append(finding)
        derived = [no for no, (_, source) in gl_accounts.items() if source == "derived"]
        self._step(
            invoice_id,
            "gl_derivation",
            {no: line.gl_account for no, line in _by_line(invoice)},
            "derived" if derived else "stated",
            (
                f"GL account derived from material group on line(s) {derived}."
                if derived
                else "GL account stated on every line."
            ),
            {"gl_accounts": {no: value for no, value in gl_accounts.items()}},
        )

        # --------------------------------------------- cost centre derivation
        for line in invoice.lines:
            po_line = self._po_line_for(line)
            centre, source, finding = checks.derive_cost_center(line, invoice, po_line)
            cost_centers[line.line_no] = (centre, source)
            if finding:
                findings.append(finding)
        unresolved_cc = [no for no, (value, _) in cost_centers.items() if value is None]
        self._step(
            invoice_id,
            "cost_center_derivation",
            {no: line.cost_center for no, line in _by_line(invoice)},
            "unresolved" if unresolved_cc else "resolved",
            (
                f"Cost centre could not be resolved from structured data on line(s) "
                f"{unresolved_cc}; free text is present and is not read here."
                if unresolved_cc
                else "Cost centre resolved on every line."
            ),
            {"cost_centers": {no: value for no, value in cost_centers.items()}},
        )

        # ------------------------------------------------------ three-way match
        matched_lines = 0
        receipted: dict[int, float] = {}
        for line in invoice.lines:
            po_line = self._po_line_for(line)
            material_id = materials[line.line_no]
            if po_line is None or line.po_id is None or material_id is None:
                continue
            matched_lines += 1
            cutoff = invoice.received_date + timedelta(days=config.gr_grace_days)
            received_by_cutoff = repo.received_base_quantity(
                line.po_id, po_line.po_line, material_id, as_of=cutoff
            )
            received_total = repo.received_base_quantity(line.po_id, po_line.po_line, material_id)
            receipted[line.line_no] = received_total
            finding = checks.check_goods_receipt(line, received_by_cutoff, received_total, config)
            if finding:
                findings.append(finding)
        self._step(
            invoice_id,
            "three_way_match",
            {"category": category, "matched_lines": matched_lines},
            "matched" if matched_lines else "not_applicable",
            (
                f"Three-way match applicable to {matched_lines} line(s): purchase order, "
                "goods receipt and invoice."
                if matched_lines
                else "No purchase order referenced; three-way match does not apply."
            ),
        )

        # ----------------------------------------------------- quantity check
        for line in invoice.lines:
            po_line = self._po_line_for(line)
            material_id = materials[line.line_no]
            if po_line is None or line.po_id is None or material_id is None:
                continue
            key = (line.po_id, po_line.po_line)
            received = receipted.get(line.line_no, 0.0)
            comparison = checks.check_quantity(
                line, material_id, received, consumed.get(key, 0.0), config
            )
            consumed[key] = consumed.get(key, 0.0) + comparison.invoiced_base_qty
            quantity_comparisons[line.line_no] = comparison
        self._step(
            invoice_id,
            "quantity_check",
            {no: c.model_dump(mode="json") for no, c in quantity_comparisons.items()},
            "computed" if quantity_comparisons else "not_applicable",
            (
                "; ".join(
                    f"line {no}: invoiced {c.invoiced_base_qty:,.3f} of "
                    f"{c.open_base_qty:,.3f} available"
                    for no, c in quantity_comparisons.items()
                )
                or "No quantity comparison applicable."
            ),
        )

        # -------------------------------------------------------- price check
        for line in invoice.lines:
            po_line = self._po_line_for(line)
            material_id = materials[line.line_no]
            if po_line is None or material_id is None:
                continue
            matched = quantity_comparisons.get(line.line_no)
            matched_qty = matched.invoiced_base_qty if matched else 0.0
            price_comparisons[line.line_no] = checks.compare_price(
                line, po_line, material_id, matched_qty, config
            )
        self._step(
            invoice_id,
            "price_check",
            {no: c.model_dump(mode="json") for no, c in price_comparisons.items()},
            "computed" if price_comparisons else "not_applicable",
            (
                "; ".join(
                    f"line {no}: normalised {c.invoice_unit_price_normalised:,.6f} vs "
                    f"{c.po_unit_price_normalised:,.6f} ({c.residual_pct:+.2f}%), "
                    f"naive comparison would show {c.naive_residual_abs:+,.4f}"
                    for no, c in price_comparisons.items()
                )
                or "No price comparison applicable."
            ),
        )

        # ------------------------------------------------ tolerance evaluation
        breached: list[int] = []
        for line in invoice.lines:
            price_result = price_comparisons.get(line.line_no)
            if price_result is not None:
                price_finding = checks.price_variance_finding(line, price_result, config)
                if price_finding:
                    findings.append(price_finding)
                    breached.append(line.line_no)
            quantity_result = quantity_comparisons.get(line.line_no)
            # A line with no goods receipt at all is already reported as a
            # missing receipt. Raising over-billing on top of it would be true
            # but useless: it would say the same thing twice, and the resolution
            # is to post the receipt, not to dispute the quantity. Over-billing
            # is reported only where goods were actually received.
            if quantity_result is not None and receipted.get(line.line_no, 0.0) > 0:
                qty_finding = checks.quantity_variance_finding(line, quantity_result)
                if qty_finding:
                    findings.append(qty_finding)
        self._step(
            invoice_id,
            "tolerance_evaluation",
            {
                "price_tolerance_pct": config.price_tolerance_pct,
                "price_tolerance_abs": config.price_tolerance_abs,
                "quantity_tolerance_pct": config.quantity_tolerance_pct,
            },
            "breach" if breached else "within_tolerance",
            (
                f"Price tolerance breached on line(s) {breached}."
                if breached
                else (
                    f"All residuals inside tolerance ({config.price_tolerance_pct}% or "
                    f"{config.price_tolerance_abs:,.2f})."
                )
            ),
            {"breached_lines": breached},
        )

        # --------------------------------------------------------- assemble
        for line in invoice.lines:
            gl_account, gl_source = gl_accounts[line.line_no]
            centre, centre_source = cost_centers[line.line_no]
            resolutions.append(
                LineResolution(
                    line_no=line.line_no,
                    material_id=materials[line.line_no],
                    material_source=(
                        "supplier_item_no" if materials[line.line_no] else "unresolved"
                    ),
                    tax_code=tax_codes[line.line_no],
                    gl_account=gl_account,
                    gl_source=gl_source,  # type: ignore[arg-type]
                    cost_center=centre,
                    cost_center_source=centre_source,  # type: ignore[arg-type]
                    price=price_comparisons.get(line.line_no),
                    quantity=quantity_comparisons.get(line.line_no),
                )
            )

        # ------------------------------------------------- deterministic routing
        # Routed here with no model input at all, so the trace records the tier
        # the rules alone assign. The agent layer re-routes afterwards with the
        # model's confidence added, and the two records side by side show what
        # the model changed — which, by construction, can only be to tighten.
        primary = _primary_exception(findings)
        routing = route(
            exception_type=primary,
            is_exception=bool(findings),
            document_value=invoice.stated_total_gross,
            auto_clear_max_value=config.auto_clear_max_value,
            propose_max_value=config.propose_max_value,
            auto_clear_min_confidence=config.auto_clear_min_confidence,
            model_confidence=None,
            severity=_primary_severity(findings, primary),
        )

        result = InvoiceResult(
            invoice_id=invoice.invoice_id,
            category=category,  # type: ignore[arg-type]
            document_value=invoice.stated_total_gross,
            currency=invoice.currency,
            resolutions=tuple(resolutions),
            findings=tuple(findings),
            duplicate_candidates=tuple(duplicates),
            routing=routing,
            evaluated_at=datetime.now(UTC),
        )

        # -------------------------------------------------- routing decision
        # The deterministic outcome. The agent layer refines this for cases that
        # are exceptions; a clean invoice never reaches a model at all.
        self._step(
            invoice_id,
            "routing_decision",
            {
                "findings": [f.rule_id for f in findings],
                "document_value": result.document_value,
            },
            routing.tier,
            (
                # The full reason is in `detail`; the summary is one line a
                # reviewer scans, and the trace schema caps it deliberately.
                f"{len(findings)} finding(s); primary exception "
                f"{result.primary_exception}. Rules alone route to {routing.tier}."
                if findings
                else (
                    "No finding. Invoice matches on price, quantity and coding. "
                    f"Rules alone route to {routing.tier}."
                )
            ),
            {
                "tier": routing.tier,
                "routing_reasons": routing.reasons,
                "findings": [
                    {
                        "rule_id": f.rule_id,
                        "exception_type": f.exception_type,
                        "line_no": f.line_no,
                        "severity": f.severity,
                        "detail": f.detail,
                    }
                    for f in findings
                ],
            },
        )
        return result

    # ---------------------------------------------------------------- helpers
    def _po_line_for(self, line: InvoiceLine) -> PurchaseOrderLine | None:
        if line.po_id is None or line.po_line is None:
            return None
        po = self.repository.purchase_order(line.po_id)
        if po is None:
            return None
        return po.line(line.po_line)


def _by_line(invoice: Invoice) -> list[tuple[int, InvoiceLine]]:
    return [(line.line_no, line) for line in invoice.lines]


def _primary_exception(findings: list[ExceptionFinding]) -> str:
    """Same precedence the result object uses, available before it is built."""
    if not findings:
        return "no_exception"
    rank = {"high": 0, "medium": 1, "low": 2}
    return min(
        findings,
        key=lambda f: (rank[f.severity], EXCEPTION_PRECEDENCE.index(f.exception_type)),
    ).exception_type


def _primary_severity(
    findings: list[ExceptionFinding], primary: str
) -> Literal["low", "medium", "high"] | None:
    """Severity of the finding that decides the routing, not the worst overall."""
    match = next((f for f in findings if f.exception_type == primary), None)
    return match.severity if match else None


__all__ = ["STEP_RULES", "InvoiceEngine"]
