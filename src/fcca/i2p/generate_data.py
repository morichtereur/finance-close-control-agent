"""Seeded synthetic invoice-to-pay dataset with ground-truth labels.

Roughly 300 invoices, each generated from a named scenario whose expected
outcome is known before the deterministic engine ever sees it. That is what
makes the evaluation in :mod:`fcca.i2p.evaluate` report real numbers rather
than plausible ones — and it is also the honest limitation, because a label
derived from a scenario definition validates the pipeline, not anyone's finance
judgement.

**The messiness is the point.** A generator that emitted clean invoices with one
obvious defect each would produce a three-way match that passes trivially and
tells you nothing. These invoices carry, on ordinary lines that are *not*
exceptions:

* cascading percentage discounts, sequential and not summable;
* absolute per-unit surcharges applied after the cascade;
* price units — a purchase order per 100 PCE against an invoice per PCE;
* units of measure that differ between order and invoice;
* partial deliveries, so the received quantity is rarely the ordered quantity;
* mixed tax rates within one document;
* the vendor's own item numbers rather than ours.

Every one of those makes a naive price comparison disagree with the purchase
order. None of them is an exception. The dataset is built this way so that a
system which merely subtracts printed prices scores badly on it, and the
normalisation has something to prove.

Seeded throughout: the same seed produces byte-identical files, which is why the
numbers in the README can be checked rather than believed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fcca.i2p.masterdata import (
    COMPANY_CODES,
    COST_CENTERS,
    GL_ACCOUNTS,
    GL_BY_MATERIAL_GROUP,
    MATERIALS,
    MATERIALS_BY_ID,
    SUPPLIER_ITEM_NUMBERS,
    TAX_CODES,
    UOM_CONVERSIONS,
    UOM_FACTORS,
    VENDORS,
    VENDORS_BY_ID,
    cost_centers_for,
)
from fcca.i2p.models import (
    ExceptionType,
    GoodsReceipt,
    Invoice,
    InvoiceLine,
    PriceElements,
    PurchaseOrder,
    PurchaseOrderLine,
    UnitOfMeasure,
)
from fcca.i2p.pricing import normalise_unit_price, to_base_quantity, uom_factor
from fcca.shared.config import Settings, get_settings

#: Scenario mix. Weights are chosen so the dataset resembles an accounts-payable
#: population rather than a balanced test set: most invoices are ordinary, and
#: the exception classes are unevenly common, because that is what makes a
#: touchless-rate figure mean anything.
SCENARIO_WEIGHTS: dict[str, int] = {
    "clean_mm": 92,
    "clean_mm_messy_pricing": 58,
    "clean_fi": 34,
    "price_variance_within_tolerance": 22,
    "price_variance_outside_tolerance": 24,
    "missing_goods_receipt": 20,
    "delayed_goods_receipt": 12,
    "duplicate_invoice": 14,
    "cost_center_in_free_text": 16,
    "gl_account_not_stated": 16,
    "bank_details_mismatch": 12,
}

#: What each scenario is expected to produce. `no_exception` is a real answer,
#: not an absence of one: the within-tolerance price scenario deliberately
#: carries a genuine price difference that the tolerance clears.
SCENARIO_LABELS: dict[str, ExceptionType] = {
    "clean_mm": "no_exception",
    "clean_mm_messy_pricing": "no_exception",
    "clean_fi": "no_exception",
    "price_variance_within_tolerance": "no_exception",
    "price_variance_outside_tolerance": "price_variance",
    "missing_goods_receipt": "missing_or_delayed_goods_receipt",
    "delayed_goods_receipt": "missing_or_delayed_goods_receipt",
    "duplicate_invoice": "duplicate_invoice",
    "cost_center_in_free_text": "cost_center_missing",
    "gl_account_not_stated": "gl_account_missing",
    "bank_details_mismatch": "bank_details_mismatch",
}

PERIOD_START = date(2026, 4, 1)
PERIOD_DAYS = 120

FREE_TEXT_TEMPLATES: tuple[str, ...] = (
    "Please book to {alias} as agreed with the plant manager.",
    "Charge to {alias}; requisition raised verbally, PO to follow.",
    "For {alias} — replacement parts after the June shutdown.",
    "Cost to be borne by {alias}. Contact the site coordinator with queries.",
    "{alias} budget. Approved by the department head over email.",
)

BANK_MISMATCH_NOTES: tuple[str, ...] = (
    "Please note our new bank account, effective immediately.",
    "Updated banking details below — kindly use for this and all future payments.",
    "Our previous account is closed. Remit to the account shown on this invoice.",
)


@dataclass
class GeneratedCase:
    """One invoice plus everything it refers to, and what it is supposed to be."""

    invoice: Invoice
    scenario: str
    expected_exception: ExceptionType
    notes: str


class _Builder:
    """Holds the seeded RNG and the running id counters."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # A dedicated Random instance rather than the module-level functions, so
        # that generating this dataset cannot perturb any other seeded process.
        self.rng = random.Random(settings.random_seed + 17)
        self.purchase_orders: dict[str, PurchaseOrder] = {}
        self.goods_receipts: list[GoodsReceipt] = []
        self.cases: list[GeneratedCase] = []
        self._po_seq = 0
        self._gr_seq = 0
        self._inv_seq = 0

    # ------------------------------------------------------------------- ids
    def next_po_id(self) -> str:
        self._po_seq += 1
        return f"PO-45{self._po_seq:05d}"

    def next_gr_id(self) -> str:
        self._gr_seq += 1
        return f"GR-50{self._gr_seq:05d}"

    def next_invoice_id(self) -> str:
        self._inv_seq += 1
        return f"INV-{self._inv_seq:05d}"

    # ------------------------------------------------------------- primitives
    def a_date(self, offset_days: int = 0) -> date:
        return PERIOD_START + timedelta(days=self.rng.randrange(PERIOD_DAYS) + offset_days)

    def a_uom_for(self, material_id: str) -> UnitOfMeasure:
        """Sometimes the line's UoM is an alternative one, which forces conversion."""
        material = MATERIALS_BY_ID[material_id]
        alternatives: list[UnitOfMeasure] = [u for (m, u) in UOM_FACTORS if m == material_id]
        if alternatives and self.rng.random() < 0.35:
            return self.rng.choice(alternatives)
        return material.base_uom

    def a_discount_cascade(self) -> tuple[float, ...]:
        """Zero to three sequential percentage discounts."""
        count = self.rng.choices([0, 1, 2, 3], weights=[30, 30, 25, 15])[0]
        return tuple(round(self.rng.uniform(1.0, 7.5), 2) for _ in range(count))

    def a_surcharge(self) -> float:
        if self.rng.random() < 0.35:
            return round(self.rng.uniform(0.05, 2.50), 4)
        return 0.0

    # ------------------------------------------------------- purchase orders
    def build_purchase_order(
        self,
        vendor_id: str,
        company_code: str,
        line_count: int,
        po_date: date,
        min_line_value: float = 0.0,
    ) -> PurchaseOrder:
        vendor = VENDORS_BY_ID[vendor_id]
        cost_centres = cost_centers_for(company_code) or COST_CENTERS
        lines: list[PurchaseOrderLine] = []
        for index in range(1, line_count + 1):
            material = self.rng.choice(MATERIALS)
            uom = self.a_uom_for(material.material_id)
            factor = uom_factor(material.material_id, uom)
            # Price the line in whatever UoM it is quoted in, keeping the money
            # per base unit close to the material's standard price.
            price_unit = self.rng.choice([1, 1, 1, 10, 100]) if factor == 1.0 else 1
            list_price = round(
                material.standard_price / material.price_unit * factor * price_unit, 4
            )
            quantity = float(self.rng.randrange(5, 400))
            if min_line_value > 0:
                # Order enough of it that the line is worth something. Used by the
                # outside-tolerance scenario: the configured rule clears a line
                # inside EITHER limit, so a percentage breach on a line worth
                # forty euros is correctly not an exception. Making the variance
                # enormous instead would breach both limits but stop resembling a
                # pricing dispute anyone has ever had.
                per_unit = (
                    normalise_unit_price(
                        PriceElements(
                            list_price=list_price,
                            price_unit=price_unit,
                            discount_pct=(),
                            surcharge_per_unit=0.0,
                        ),
                        material.material_id,
                        uom,
                    )
                    * factor
                )
                if per_unit > 0:
                    quantity = max(quantity, min_line_value / per_unit)
            lines.append(
                PurchaseOrderLine(
                    po_line=index * 10,
                    material_id=material.material_id,
                    supplier_item_no=SUPPLIER_ITEM_NUMBERS[material.material_id],
                    quantity=round(quantity, 3),
                    uom=uom,
                    price=PriceElements(
                        list_price=list_price,
                        price_unit=price_unit,
                        discount_pct=self.a_discount_cascade(),
                        surcharge_per_unit=self.a_surcharge(),
                    ),
                    tax_code=self.rng.choices(list(TAX_CODES), weights=[70, 20, 10])[0],
                    gl_account=GL_BY_MATERIAL_GROUP[material.material_group],
                    cost_center=self.rng.choice(cost_centres).cost_center,
                )
            )
        po = PurchaseOrder(
            po_id=self.next_po_id(),
            vendor_id=vendor_id,
            company_code=company_code,
            po_date=po_date,
            currency=vendor.currency,
            lines=tuple(lines),
        )
        self.purchase_orders[po.po_id] = po
        return po

    def post_goods_receipts(
        self, po: PurchaseOrder, receipt_date: date, coverage: float = 1.0
    ) -> None:
        """Receive `coverage` of each ordered line, sometimes across two receipts.

        Partial delivery is the normal case. A dataset where received always
        equals ordered would let a quantity check pass by doing nothing.
        """
        for line in po.lines:
            total = round(line.quantity * coverage, 3)
            if total <= 0:
                continue
            if self.rng.random() < 0.30 and total > 2:
                first = round(total * self.rng.uniform(0.3, 0.7), 3)
                splits = [first, round(total - first, 3)]
            else:
                splits = [total]
            for offset, quantity in enumerate(splits):
                self.goods_receipts.append(
                    GoodsReceipt(
                        gr_id=self.next_gr_id(),
                        po_id=po.po_id,
                        po_line=line.po_line,
                        receipt_date=receipt_date + timedelta(days=offset * 2),
                        quantity=quantity,
                        uom=line.uom,
                    )
                )

    # --------------------------------------------------------------- invoices
    def invoice_line_from_po(
        self,
        line_no: int,
        po: PurchaseOrder,
        po_line: PurchaseOrderLine,
        *,
        quantity: float | None = None,
        price_factor: float = 1.0,
        restate_pricing: bool = False,
        omit_gl: bool = False,
        omit_cost_center: bool = False,
    ) -> InvoiceLine:
        """Build the vendor's version of a purchase-order line.

        ``restate_pricing`` is the interesting path: the vendor expresses the
        *same* net price differently — a different price unit, the discount
        cascade already applied, the surcharge folded in. The printed numbers
        disagree with the purchase order; the normalised ones do not.
        """
        material = MATERIALS_BY_ID[po_line.material_id]
        uom = po_line.uom
        price = po_line.price

        if restate_pricing:
            net_per_line_uom = normalise_unit_price(
                po_line.price, po_line.material_id, po_line.uom
            ) * uom_factor(po_line.material_id, po_line.uom)
            # Same money, quoted per single unit with no discount schedule.
            price = PriceElements(
                list_price=round(net_per_line_uom * price_factor, 6),
                price_unit=1,
                discount_pct=(),
                surcharge_per_unit=0.0,
            )
        elif price_factor != 1.0:
            price = PriceElements(
                list_price=round(price.list_price * price_factor, 6),
                price_unit=price.price_unit,
                discount_pct=price.discount_pct,
                surcharge_per_unit=price.surcharge_per_unit,
            )

        return InvoiceLine(
            line_no=line_no,
            description=material.description,
            supplier_item_no=po_line.supplier_item_no,
            quantity=quantity if quantity is not None else po_line.quantity,
            uom=uom,
            price=price,
            tax_rate=TAX_CODES[po_line.tax_code],
            po_id=po.po_id,
            po_line=po_line.po_line,
            gl_account=None if omit_gl else po_line.gl_account,
            cost_center=None if omit_cost_center else po_line.cost_center,
        )

    def assemble(
        self,
        *,
        vendor_id: str,
        company_code: str,
        category: str,
        invoice_date: date,
        lines: list[InvoiceLine],
        bank_iban: str | None = None,
        free_text: str = "",
        vendor_reference: str | None = None,
        invoice_id: str | None = None,
    ) -> Invoice:
        """Total the invoice the way a vendor's billing system would.

        The stated totals are computed from the stated lines, including any
        defect: an invoice with a wrong unit price has totals that are
        internally consistent with that wrong price. Totals that disagreed with
        their own lines would be a different exception, and one this dataset
        does not claim to cover.
        """
        vendor = VENDORS_BY_ID[vendor_id]
        net = 0.0
        tax = 0.0
        for line in lines:
            material_id = self._material_for_line(line)
            unit = normalise_unit_price(line.price, material_id, line.uom)
            base_qty = to_base_quantity(line.quantity, material_id, line.uom)
            line_net = unit * base_qty
            net += line_net
            tax += line_net * line.tax_rate / 100.0
        net, tax = round(net, 2), round(tax, 2)
        return Invoice(
            invoice_id=invoice_id or self.next_invoice_id(),
            vendor_id=vendor_id,
            company_code=company_code,
            category=category,  # type: ignore[arg-type]
            invoice_date=invoice_date,
            received_date=invoice_date + timedelta(days=self.rng.randrange(0, 6)),
            vendor_reference=vendor_reference
            or f"{vendor.country}{self.rng.randrange(10_000, 99_999)}",
            currency=vendor.currency,
            lines=tuple(lines),
            stated_bank_iban=bank_iban or vendor.bank_iban,
            free_text=free_text,
            stated_total_net=net,
            stated_total_tax=tax,
            stated_total_gross=round(net + tax, 2),
        )

    @staticmethod
    def _material_for_line(line: InvoiceLine) -> str:
        from fcca.i2p.masterdata import MATERIAL_BY_SUPPLIER_ITEM

        return MATERIAL_BY_SUPPLIER_ITEM[line.supplier_item_no]

    def record(
        self, invoice: Invoice, scenario: str, notes: str, expected: ExceptionType | None = None
    ) -> None:
        self.cases.append(
            GeneratedCase(
                invoice=invoice,
                scenario=scenario,
                expected_exception=expected or SCENARIO_LABELS[scenario],
                notes=notes,
            )
        )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _mm_case(
    builder: _Builder,
    scenario: str,
    *,
    price_factor: float = 1.0,
    restate: bool = False,
    gr_coverage: float = 1.0,
    gr_offset_days: int = -4,
    post_gr: bool = True,
    omit_gl: bool = False,
    omit_cost_center: bool = False,
    free_text: str = "",
    bank_iban: str | None = None,
    min_line_value: float = 0.0,
    notes: str = "",
) -> Invoice:
    """Build one PO-based invoice with its purchase order and goods receipts."""
    vendor = builder.rng.choice(VENDORS)
    company_code = builder.rng.choice(COMPANY_CODES)
    po_date = builder.a_date()
    po = builder.build_purchase_order(
        vendor.vendor_id,
        company_code,
        builder.rng.choices([1, 2, 3], weights=[55, 30, 15])[0],
        po_date,
        min_line_value=min_line_value,
    )
    invoice_date = po_date + timedelta(days=builder.rng.randrange(7, 25))
    if post_gr:
        builder.post_goods_receipts(po, invoice_date + timedelta(days=gr_offset_days), gr_coverage)

    lines = [
        builder.invoice_line_from_po(
            index,
            po,
            po_line,
            quantity=round(po_line.quantity * gr_coverage, 3) if gr_coverage < 1.0 else None,
            price_factor=price_factor,
            restate_pricing=restate,
            omit_gl=omit_gl,
            omit_cost_center=omit_cost_center,
        )
        for index, po_line in enumerate(po.lines, start=1)
    ]
    invoice = builder.assemble(
        vendor_id=vendor.vendor_id,
        company_code=company_code,
        category="MM",
        invoice_date=invoice_date,
        lines=lines,
        free_text=free_text,
        bank_iban=bank_iban,
    )
    builder.record(invoice, scenario, notes)
    return invoice


def _fi_case(builder: _Builder, scenario: str, **kwargs: Any) -> Invoice:
    """A non-PO invoice: services, no three-way match to perform."""
    vendor = builder.rng.choice(VENDORS)
    company_code = builder.rng.choice(COMPANY_CODES)
    cost_centres = cost_centers_for(company_code) or COST_CENTERS
    material = MATERIALS_BY_ID["MAT-900001"]
    omit_gl = bool(kwargs.get("omit_gl"))
    omit_cc = bool(kwargs.get("omit_cost_center"))
    line = InvoiceLine(
        line_no=1,
        description=material.description,
        supplier_item_no=SUPPLIER_ITEM_NUMBERS[material.material_id],
        quantity=float(builder.rng.randrange(8, 160)),
        uom="HR",
        price=PriceElements(
            list_price=round(material.standard_price * builder.rng.uniform(0.95, 1.05), 2),
            price_unit=1,
            discount_pct=builder.a_discount_cascade(),
            surcharge_per_unit=0.0,
        ),
        tax_rate=TAX_CODES["V1"],
        po_id=None,
        po_line=None,
        gl_account=None if omit_gl else GL_BY_MATERIAL_GROUP[material.material_group],
        cost_center=None if omit_cc else builder.rng.choice(cost_centres).cost_center,
    )
    invoice = builder.assemble(
        vendor_id=vendor.vendor_id,
        company_code=company_code,
        category="FI",
        invoice_date=builder.a_date(),
        lines=[line],
        free_text=str(kwargs.get("free_text", "")),
        bank_iban=kwargs.get("bank_iban"),
    )
    builder.record(invoice, scenario, str(kwargs.get("notes", "")))
    return invoice


def _free_text_cost_centre(builder: _Builder, company_code: str) -> tuple[str, str]:
    """A note naming a cost centre by an alias rather than by its code."""
    candidates = cost_centers_for(company_code) or COST_CENTERS
    centre = builder.rng.choice(candidates)
    alias = builder.rng.choice(centre.aliases or (centre.name,))
    return builder.rng.choice(FREE_TEXT_TEMPLATES).format(alias=alias), centre.cost_center


def _mutate_iban(builder: _Builder, iban: str) -> str:
    """A different, structurally plausible account at the same bank prefix.

    Deliberately similar to the account of record. A fraudulent redirection that
    looked obviously wrong would not test anything: the control has to be the
    comparison to vendor master, not a reviewer noticing an odd string.
    """
    prefix, tail = iban[:8], iban[8:]
    digits = "".join(str(builder.rng.randrange(10)) for _ in tail)
    return prefix + digits


def _build_scenario(builder: _Builder, scenario: str) -> None:
    settings = builder.settings
    tolerance_pct = settings.i2p.price_tolerance_pct

    if scenario == "clean_mm":
        _mm_case(builder, scenario, notes="Ordinary PO invoice, matched and received.")

    elif scenario == "clean_mm_messy_pricing":
        _mm_case(
            builder,
            scenario,
            restate=True,
            notes=(
                "Vendor restates the same net price without the discount schedule. "
                "Printed prices disagree with the PO; normalised prices do not."
            ),
        )

    elif scenario == "clean_fi":
        _fi_case(builder, scenario, notes="Non-PO service invoice, fully coded.")

    elif scenario == "price_variance_within_tolerance":
        # Deliberately inside the band but not trivially so.
        factor = 1.0 + builder.rng.uniform(0.2, 0.8) * tolerance_pct / 100.0
        _mm_case(
            builder,
            scenario,
            restate=True,
            price_factor=factor,
            notes=f"Genuine price difference of {(factor - 1) * 100:.2f}%, inside tolerance.",
        )

    elif scenario == "price_variance_outside_tolerance":
        factor = 1.0 + builder.rng.uniform(1.8, 9.0) * tolerance_pct / 100.0
        _mm_case(
            builder,
            scenario,
            restate=True,
            price_factor=factor,
            # Large enough that the smallest realistic percentage breach is also
            # material in absolute terms, so the label holds under the OR rule.
            min_line_value=settings.i2p.price_tolerance_abs * 60,
            notes=(
                f"Price difference of {(factor - 1) * 100:.2f}%, breaching both the "
                "percentage and the absolute limit."
            ),
        )

    elif scenario == "missing_goods_receipt":
        _mm_case(
            builder,
            scenario,
            post_gr=False,
            notes="Invoice received; no goods receipt posted against the order.",
        )

    elif scenario == "delayed_goods_receipt":
        _mm_case(
            builder,
            scenario,
            gr_offset_days=settings.i2p.gr_grace_days + builder.rng.randrange(4, 20),
            notes="Goods receipt exists but is posted well after the invoice arrived.",
        )

    elif scenario == "duplicate_invoice":
        original = _mm_case(
            builder,
            "clean_mm",
            notes="Original of a subsequently duplicated invoice.",
        )
        # The resubmission: same vendor, same amount, near date, reference the
        # same document with different punctuation or spacing.
        reference = original.vendor_reference
        variants = (
            f"{reference[:2]}-{reference[2:]}",
            f"{reference[:2]} {reference[2:]}",
            reference.lower(),
            f"{reference}/2",
        )
        duplicate = original.model_copy(
            update={
                "invoice_id": builder.next_invoice_id(),
                "vendor_reference": builder.rng.choice(variants),
                "invoice_date": original.invoice_date
                + timedelta(days=builder.rng.randrange(1, 21)),
                "received_date": original.received_date
                + timedelta(days=builder.rng.randrange(1, 21)),
            }
        )
        builder.record(
            duplicate,
            scenario,
            "Resubmission of an already-received invoice under a punctuation variant "
            "of the same vendor reference.",
        )

    elif scenario == "cost_center_in_free_text":
        vendor = builder.rng.choice(VENDORS)
        company_code = builder.rng.choice(COMPANY_CODES)
        text, _expected_cc = _free_text_cost_centre(builder, company_code)
        _fi_case(
            builder,
            scenario,
            omit_cost_center=True,
            free_text=text,
            notes="Cost centre absent from the coding block; named by alias in free text.",
        )

    elif scenario == "gl_account_not_stated":
        _mm_case(
            builder,
            scenario,
            omit_gl=True,
            notes="GL account not stated; derivable from the material group.",
        )

    elif scenario == "bank_details_mismatch":
        vendor = builder.rng.choice(VENDORS)
        _mm_case(
            builder,
            scenario,
            bank_iban=_mutate_iban(builder, vendor.bank_iban),
            free_text=builder.rng.choice(BANK_MISMATCH_NOTES),
            notes="Bank details on the invoice differ from the vendor master record.",
        )

    else:  # pragma: no cover - guarded by the weights table
        raise ValueError(f"unknown scenario {scenario!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(settings: Settings | None = None) -> dict[str, Any]:
    """Generate the dataset and write it as structured JSON.

    Returns a summary dict, which the CLI prints and the tests assert on.
    """
    settings = settings or get_settings()
    builder = _Builder(settings)

    plan: list[str] = []
    for scenario, weight in SCENARIO_WEIGHTS.items():
        plan.extend([scenario] * weight)
    # Shuffle so that ids do not encode the scenario — an evaluation that could
    # read the answer off the invoice number would measure nothing.
    builder.rng.shuffle(plan)

    for scenario in plan:
        _build_scenario(builder, scenario)

    out = settings.i2p_data_dir
    out.mkdir(parents=True, exist_ok=True)
    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)

    _dump(out / "vendors.json", [v.model_dump(mode="json") for v in VENDORS])
    _dump(out / "materials.json", [m.model_dump(mode="json") for m in MATERIALS])
    _dump(out / "gl_accounts.json", [g.model_dump(mode="json") for g in GL_ACCOUNTS])
    _dump(out / "cost_centers.json", [c.model_dump(mode="json") for c in COST_CENTERS])
    _dump(out / "uom_conversions.json", [u.model_dump(mode="json") for u in UOM_CONVERSIONS])
    _dump(
        out / "purchase_orders.json",
        [po.model_dump(mode="json") for po in builder.purchase_orders.values()],
    )
    _dump(
        out / "goods_receipts.json",
        [gr.model_dump(mode="json") for gr in builder.goods_receipts],
    )
    _dump(
        out / "invoices.json",
        [case.invoice.model_dump(mode="json") for case in builder.cases],
    )
    _dump(
        settings.i2p_labels_path,
        [
            {
                "invoice_id": case.invoice.invoice_id,
                "scenario": case.scenario,
                "expected_exception": case.expected_exception,
                "notes": case.notes,
            }
            for case in builder.cases
        ],
    )

    counts: dict[str, int] = {}
    for case in builder.cases:
        counts[case.expected_exception] = counts.get(case.expected_exception, 0) + 1

    return {
        "invoices": len(builder.cases),
        "purchase_orders": len(builder.purchase_orders),
        "goods_receipts": len(builder.goods_receipts),
        "vendors": len(VENDORS),
        "materials": len(MATERIALS),
        "by_expected_exception": dict(sorted(counts.items())),
        "seed": settings.random_seed,
        "output_dir": str(out),
    }


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca i2p-generate-data",
        description="Generate the seeded synthetic invoice-to-pay dataset.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = generate()
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
