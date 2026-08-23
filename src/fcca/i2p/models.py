"""Invoice-to-pay domain contracts.

Three families of type live here, and the boundary between them is the point of
the module:

* **Master data and source documents** — :class:`Vendor`, :class:`Material`,
  :class:`PurchaseOrder`, :class:`GoodsReceipt`, :class:`Invoice`. These come
  from the synthetic generator and stand in for what an ERP extract would
  supply. Nothing infers them.
* **Deterministic results** — :class:`PriceComparison`, :class:`QuantityComparison`,
  :class:`ExceptionFinding`. Computed in Python, unit-tested, and never produced
  by a model.
* **Closed vocabularies** — :data:`ExceptionType` and :data:`ResolutionAction`.
  These bound what the deterministic layer may raise and what any downstream
  agent may propose, so neither can invent a category the organisation has no
  process for.

The arithmetic types are frozen. A price comparison that could be mutated after
the fact is not evidence.

Note what an :class:`Invoice` does *not* have: a PDF, a scan, an image. This
module starts from structured JSON. Extraction from a document image is a real
and separate problem, and pretending to solve it here would make every number
downstream unfalsifiable.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fcca.shared.models import RiskLevel
from fcca.shared.routing import RoutingDecision

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: MM invoices reference a purchase order and are subject to three-way match.
#: FI invoices have no PO — there is nothing to match against, so they are
#: governed by coding completeness and approval instead.
InvoiceCategory = Literal["MM", "FI"]

#: The exception classes the deterministic layer can raise, plus the explicit
#: "nothing wrong" case. This is a closed vocabulary because it is the axis of
#: the confusion matrix in the evaluation: a free-text classification could not
#: be scored against ground truth at all.
ExceptionType = Literal[
    "no_exception",
    "missing_or_delayed_goods_receipt",
    "duplicate_invoice",
    "price_variance",
    "quantity_variance",
    "cost_center_missing",
    "gl_account_missing",
    "bank_details_mismatch",
]

#: What the system proposes doing about an exception. Closed, because an agent
#: that can invent a remedy can invent one the organisation has no process for.
ResolutionAction = Literal[
    "post_invoice",
    "derive_and_post",
    "request_goods_receipt",
    "block_for_price_review",
    "propose_credit_memo",
    "reject_as_duplicate",
    "route_to_requisitioner",
    "escalate_to_ap_manager",
    "refer_to_vendor_master_team",
]

#: Unit of measure. Deliberately small and explicit rather than a free string:
#: an unrecognised UoM must fail loudly, because silently treating CAR as PCE
#: would multiply a price by twelve.
UnitOfMeasure = Literal["PCE", "BOX", "CAR", "KG", "L", "M", "HR"]


# ---------------------------------------------------------------------------
# Master data
# ---------------------------------------------------------------------------


class Vendor(BaseModel):
    """Vendor master record — the authoritative source for bank details."""

    model_config = ConfigDict(frozen=True)

    vendor_id: str
    name: str
    country: str
    currency: str
    payment_terms: str
    tax_id: str
    bank_iban: str = Field(description="The bank account of record. Invoices are checked to it.")
    bank_name: str
    is_blocked: bool = False


class Material(BaseModel):
    """Material master record.

    ``material_group`` is what makes GL derivation deterministic: an invoice
    that omits the GL account is not a judgement call, it is a lookup.
    """

    model_config = ConfigDict(frozen=True)

    material_id: str
    description: str
    material_group: str
    base_uom: UnitOfMeasure
    standard_price: float = Field(gt=0)
    price_unit: int = Field(
        default=1,
        gt=0,
        description="Number of base units the price refers to, e.g. 100 for 'per 100 PCE'.",
    )


class GLAccount(BaseModel):
    """A general-ledger account and the material group that maps to it."""

    model_config = ConfigDict(frozen=True)

    gl_account: str
    name: str
    material_group: str | None = Field(
        default=None,
        description="Material group this account is derived from; None for non-material accounts.",
    )


class CostCenter(BaseModel):
    """A cost centre, with the aliases people actually type into free text."""

    model_config = ConfigDict(frozen=True)

    cost_center: str
    name: str
    company_code: str
    aliases: tuple[str, ...] = Field(
        default=(),
        description=(
            "Names a requisitioner might write instead of the code — 'Plant 2 Maintenance' "
            "for CC-4200. Resolution matches against these, so a proposed cost centre is "
            "always one that exists rather than one that was invented."
        ),
    )


class UomConversion(BaseModel):
    """How many base units are in an alternative unit of measure."""

    model_config = ConfigDict(frozen=True)

    material_id: str
    alternative_uom: UnitOfMeasure
    factor: float = Field(gt=0, description="Base units per alternative unit, e.g. 12 PCE per BOX.")


# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


class PriceElements(BaseModel):
    """The pricing of one line, before normalisation.

    Kept as separate components rather than a single net price because that is
    how the two sides actually differ: a purchase order carries a list price and
    a discount schedule, and an invoice carries whatever the vendor's system
    printed. Comparing the printed numbers is exactly the mistake this module
    exists to avoid.
    """

    model_config = ConfigDict(frozen=True)

    list_price: float = Field(
        ge=0, description="Gross price for `price_unit` units of the line's unit of measure."
    )
    price_unit: int = Field(
        default=1,
        gt=0,
        description="How many units of the line's UoM the list price covers, e.g. 100.",
    )
    discount_pct: tuple[float, ...] = Field(
        default=(),
        description=(
            "Percentage discounts applied in sequence, each to the running result. "
            "Order matters and the sequence is not commutative with the surcharge."
        ),
    )
    surcharge_per_unit: float = Field(
        default=0.0,
        description=(
            "Absolute amount per unit of the line's UoM, applied after the percentage "
            "cascade and before any UoM conversion. Negative values are absolute discounts."
        ),
    )

    @model_validator(mode="after")
    def _discounts_are_percentages(self) -> PriceElements:
        for pct in self.discount_pct:
            if not 0.0 <= pct < 100.0:
                raise ValueError(f"discount_pct entries must be in [0, 100); got {pct}")
        return self


class PurchaseOrderLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    po_line: int
    material_id: str
    supplier_item_no: str = Field(
        description="The vendor's own item number, which differs from ours and is what they invoice."
    )
    quantity: float = Field(gt=0)
    uom: UnitOfMeasure
    price: PriceElements
    tax_code: str
    gl_account: str
    cost_center: str


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    po_id: str
    vendor_id: str
    company_code: str
    po_date: date
    currency: str
    lines: tuple[PurchaseOrderLine, ...]

    def line(self, po_line: int) -> PurchaseOrderLine | None:
        return next((line for line in self.lines if line.po_line == po_line), None)


class GoodsReceipt(BaseModel):
    """A receipt against one purchase-order line.

    Partial deliveries are the normal case, not an edge case: a line may have
    several receipts, and the quantity available to invoice is their sum.
    """

    model_config = ConfigDict(frozen=True)

    gr_id: str
    po_id: str
    po_line: int
    receipt_date: date
    quantity: float = Field(gt=0)
    uom: UnitOfMeasure


class InvoiceLine(BaseModel):
    """One line as the vendor stated it.

    Every field here is the *vendor's* view. ``supplier_item_no`` is their code,
    not ours; ``uom`` may not be the purchase order's; ``price`` is whatever
    their billing system produced. Resolving those to our master data is the
    deterministic engine's job.
    """

    model_config = ConfigDict(frozen=True)

    line_no: int
    description: str
    supplier_item_no: str
    quantity: float = Field(gt=0)
    uom: UnitOfMeasure
    price: PriceElements
    tax_rate: float = Field(ge=0, description="Percentage, e.g. 19.0.")

    # Structured coding. Absent on the exception cases that have to be derived.
    po_id: str | None = None
    po_line: int | None = None
    gl_account: str | None = None
    cost_center: str | None = None


class Invoice(BaseModel):
    """A vendor invoice as received, in structured form.

    ``free_text`` is where the messy real-world information lives — the cost
    centre someone typed into a description field instead of coding it properly.
    It is treated as untrusted input everywhere downstream.
    """

    model_config = ConfigDict(frozen=True)

    invoice_id: str
    vendor_id: str
    company_code: str
    category: InvoiceCategory
    invoice_date: date
    received_date: date
    vendor_reference: str = Field(description="The vendor's own invoice number.")
    currency: str
    lines: tuple[InvoiceLine, ...]
    stated_bank_iban: str = Field(description="Bank details printed on the invoice.")
    free_text: str = Field(default="", description="Unstructured note accompanying the invoice.")
    stated_total_net: float = Field(description="Net total as printed by the vendor.")
    stated_total_tax: float
    stated_total_gross: float

    @property
    def po_ids(self) -> tuple[str, ...]:
        return tuple({line.po_id for line in self.lines if line.po_id})


# ---------------------------------------------------------------------------
# Deterministic results
# ---------------------------------------------------------------------------


class PriceComparison(BaseModel):
    """The result of comparing one invoice line's price to its purchase order.

    Both sides are normalised to a net price for one base unit before anything
    is compared. The residual is reported in absolute and percentage terms
    because the tolerance is evaluated against both, and because a reviewer
    reading the queue needs to know whether 4% is four cents or four thousand.

    ``naive_residual_abs`` is carried deliberately: it is what a comparison of
    the printed unit prices would have produced. Showing it next to the
    normalised residual is what demonstrates that the normalisation is doing
    work rather than being decorative.
    """

    model_config = ConfigDict(frozen=True)

    po_unit_price_normalised: float
    invoice_unit_price_normalised: float
    residual_abs: float = Field(description="Invoice minus purchase order, per base unit.")
    residual_pct: float = Field(description="Residual as a percentage of the purchase-order price.")
    line_residual_abs: float = Field(description="Residual multiplied by the matched quantity.")
    naive_residual_abs: float = Field(
        description="What comparing the two printed list prices would have shown."
    )
    within_tolerance: bool
    tolerance_pct: float
    tolerance_abs: float


class QuantityComparison(BaseModel):
    """Invoiced quantity against goods received, in base units."""

    model_config = ConfigDict(frozen=True)

    invoiced_base_qty: float
    received_base_qty: float
    open_base_qty: float = Field(description="Received minus already invoiced elsewhere.")
    residual_base_qty: float = Field(
        description="Invoiced minus available. Positive is over-billing."
    )
    within_tolerance: bool
    tolerance_pct: float


class ExceptionFinding(BaseModel):
    """One deterministic finding against one invoice.

    Findings are produced by rules, never by a model. The model's job begins
    only once a finding exists, and it may classify and propose — it may not
    add, remove or re-rate one.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str
    exception_type: ExceptionType
    line_no: int | None = Field(default=None, description="None for header-level findings.")
    severity: RiskLevel
    detail: str
    evidence: dict[str, object] = Field(
        default_factory=dict, description="The values the rule actually compared."
    )


class LineResolution(BaseModel):
    """What the deterministic layer worked out about one invoice line.

    Every field records not just the value but where it came from. ``stated``
    means the vendor supplied it; ``derived`` means a rule computed it from
    master data; ``unresolved`` means neither, which is an exception. A reviewer
    reading the queue must be able to tell a value that was on the document from
    one the system worked out.
    """

    model_config = ConfigDict(frozen=True)

    line_no: int
    material_id: str | None = None
    material_source: Literal["supplier_item_no", "unresolved"] = "unresolved"
    tax_code: str | None = None
    gl_account: str | None = None
    gl_source: Literal["stated", "derived", "unresolved"] = "unresolved"
    cost_center: str | None = None
    cost_center_source: Literal["stated", "derived_from_po", "unresolved"] = "unresolved"
    price: PriceComparison | None = None
    quantity: QuantityComparison | None = None


class InvoiceResult(BaseModel):
    """The complete deterministic outcome for one invoice.

    This object is what the agent layer receives and what the UI renders. It
    contains no model output: everything in it was computed, and the separation
    is what lets the evaluation attribute a mistake to the right layer.
    """

    invoice_id: str
    category: InvoiceCategory
    document_value: float = Field(description="Gross value in document currency.")
    currency: str
    resolutions: tuple[LineResolution, ...]
    findings: tuple[ExceptionFinding, ...]
    duplicate_candidates: tuple[str, ...] = ()
    routing: RoutingDecision = Field(
        description="The tier the deterministic layer alone would assign, before any model."
    )
    evaluated_at: datetime

    @property
    def is_exception(self) -> bool:
        return bool(self.findings)

    @property
    def primary_exception(self) -> ExceptionType:
        """The finding that decides how the invoice is handled.

        Where several rules fire, the most severe wins, and ties are broken by
        the fixed order below rather than by whichever rule happened to run
        first. Routing must not depend on iteration order.
        """
        if not self.findings:
            return "no_exception"
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        return min(
            self.findings,
            key=lambda f: (severity_rank[f.severity], EXCEPTION_PRECEDENCE.index(f.exception_type)),
        ).exception_type


#: Fixed tie-break order for :attr:`InvoiceResult.primary_exception`. Ordered by
#: how much money a wrong answer costs: paying a fraudster outranks paying twice,
#: which outranks paying too much, which outranks paying for goods not yet
#: booked in. Coding gaps come last — they are wrong, but they are recoverable.
EXCEPTION_PRECEDENCE: tuple[ExceptionType, ...] = (
    "bank_details_mismatch",
    "duplicate_invoice",
    "quantity_variance",
    "price_variance",
    "missing_or_delayed_goods_receipt",
    "cost_center_missing",
    "gl_account_missing",
    "no_exception",
)


__all__ = [
    "EXCEPTION_PRECEDENCE",
    "CostCenter",
    "ExceptionFinding",
    "ExceptionType",
    "GLAccount",
    "GoodsReceipt",
    "Invoice",
    "InvoiceCategory",
    "InvoiceLine",
    "InvoiceResult",
    "LineResolution",
    "Material",
    "PriceComparison",
    "PriceElements",
    "PurchaseOrder",
    "PurchaseOrderLine",
    "QuantityComparison",
    "ResolutionAction",
    "UnitOfMeasure",
    "UomConversion",
    "Vendor",
]
