"""The deterministic checks.

Every function here is a pure function of documents, master data and
configuration. None of them calls a model, and none of them can: no provider is
imported in this module and nothing in it is asked to interpret anything. That
is the architectural rule the whole repository rests on — arithmetic, matching
and tolerance decisions are Python with unit tests, and a language model is
never in the position of deciding whether 51,000 exceeds 50,000.

Each check returns typed evidence and, where something is wrong, an
:class:`~fcca.i2p.models.ExceptionFinding` carrying the values it actually
compared. A finding whose evidence a reviewer cannot check is a finding they
have to take on trust, which defeats the purpose.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict

from fcca.i2p.extraction import FieldProvenance
from fcca.i2p.masterdata import (
    COST_CENTERS,
    GL_BY_MATERIAL_GROUP,
    MATERIAL_BY_SUPPLIER_ITEM,
    MATERIALS_BY_ID,
    TAX_CODES,
)
from fcca.i2p.models import (
    ExceptionFinding,
    Invoice,
    InvoiceLine,
    PriceComparison,
    PurchaseOrderLine,
    QuantityComparison,
    Vendor,
)
from fcca.i2p.pricing import (
    naive_unit_price,
    normalise_unit_price,
    residual_percentage,
    to_base_quantity,
)
from fcca.shared.config import I2PConfig

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_invoice(invoice: Invoice) -> tuple[str, str]:
    """MM or FI, with the reason.

    Not a judgement: an invoice is MM if any line references a purchase order.
    The distinction decides which checks are even applicable — there is no
    three-way match to perform on a document with nothing to match against — so
    getting it from the data rather than from a stated field means a vendor
    cannot opt out of matching by mislabelling their invoice.
    """
    if any(line.po_id for line in invoice.lines):
        return "MM", "At least one line references a purchase order."
    return "FI", "No line references a purchase order; coding and approval govern instead."


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def normalise_reference(reference: str) -> str:
    """Strip punctuation and case so that 'INV-4471' and 'inv 4471' compare equal.

    Vendors are not consistent about how they print their own invoice numbers,
    and a duplicate check that required an exact string match would miss the
    resubmissions that actually occur.
    """
    return re.sub(r"[^a-z0-9]", "", reference.lower())


def reference_similarity(left: str, right: str) -> float:
    """Similarity of two vendor references after normalisation, in [0, 1]."""
    return SequenceMatcher(None, normalise_reference(left), normalise_reference(right)).ratio()


def check_duplicate(
    invoice: Invoice,
    earlier: list[Invoice],
    config: I2PConfig,
) -> tuple[list[str], ExceptionFinding | None]:
    """Look for an earlier invoice that is the same document.

    Four signals, all required: same vendor (guaranteed by the caller's window),
    same gross amount within a cent, a date inside the look-back window, and a
    vendor reference that is similar enough after normalisation. Requiring all
    four is what keeps this from flagging a vendor's ordinary monthly invoice for
    the same recurring amount.
    """
    matches: list[str] = []
    evidence: list[dict[str, object]] = []
    for other in earlier:
        amount_delta = abs(invoice.stated_total_gross - other.stated_total_gross)
        if amount_delta > config.duplicate_amount_tolerance:
            continue
        similarity = reference_similarity(invoice.vendor_reference, other.vendor_reference)
        if similarity < config.duplicate_reference_similarity:
            continue
        matches.append(other.invoice_id)
        evidence.append(
            {
                "candidate": other.invoice_id,
                "candidate_reference": other.vendor_reference,
                "reference_similarity": round(similarity, 4),
                "amount_delta": round(amount_delta, 2),
                "days_apart": (invoice.invoice_date - other.invoice_date).days,
            }
        )

    if not matches:
        return [], None

    return matches, ExceptionFinding(
        rule_id="I2P-R-002",
        exception_type="duplicate_invoice",
        severity="high",
        detail=(
            f"Gross amount and vendor reference match {len(matches)} earlier invoice(s) "
            f"from the same vendor: {', '.join(matches)}."
        ),
        evidence={
            "this_reference": invoice.vendor_reference,
            "gross": invoice.stated_total_gross,
            "candidates": evidence,
        },
    )


# ---------------------------------------------------------------------------
# Master data resolution
# ---------------------------------------------------------------------------


def resolve_material(line: InvoiceLine) -> tuple[str | None, ExceptionFinding | None]:
    """Map the vendor's item number to our material id.

    A lookup, not a guess. If the supplier item number is unknown the line is
    unresolvable and everything downstream — GL derivation, UoM conversion, the
    price comparison — has no basis, so it stops here rather than proceeding on
    an assumption.
    """
    material_id = MATERIAL_BY_SUPPLIER_ITEM.get(line.supplier_item_no)
    if material_id is not None:
        return material_id, None
    return None, ExceptionFinding(
        rule_id="I2P-R-003",
        exception_type="gl_account_missing",
        line_no=line.line_no,
        severity="medium",
        detail=(
            f"Supplier item number {line.supplier_item_no!r} does not correspond to any "
            "material in the material master."
        ),
        evidence={"supplier_item_no": line.supplier_item_no},
    )


def resolve_tax_code(line: InvoiceLine) -> tuple[str | None, ExceptionFinding | None]:
    """Map the stated tax rate to a tax code.

    Rates are mixed within a document, so this is per line. An unrecognised rate
    is a finding rather than a rounding: posting an invoice under the wrong tax
    code is a reportable error, not an inconvenience.
    """
    for code, rate in TAX_CODES.items():
        if abs(rate - line.tax_rate) < 1e-9:
            return code, None
    return None, ExceptionFinding(
        rule_id="I2P-R-004",
        exception_type="gl_account_missing",
        line_no=line.line_no,
        severity="medium",
        detail=f"Stated tax rate {line.tax_rate}% matches no configured tax code.",
        evidence={"tax_rate": line.tax_rate, "known_rates": sorted(TAX_CODES.values())},
    )


def derive_gl_account(
    line: InvoiceLine, material_id: str | None
) -> tuple[str | None, str, ExceptionFinding | None]:
    """Return (account, source, finding).

    Where the vendor stated an account, that is used. Where they did not, the
    material group determines it — a table lookup with exactly one answer. This
    is the case the specification calls out, and it is worth being explicit
    about why no model is involved: there is nothing here to infer.
    """
    if line.gl_account:
        return line.gl_account, "stated", None
    if material_id is None:
        return None, "unresolved", None  # the material finding already covers this
    material = MATERIALS_BY_ID[material_id]
    account = GL_BY_MATERIAL_GROUP.get(material.material_group)
    if account is None:
        return (
            None,
            "unresolved",
            ExceptionFinding(
                rule_id="I2P-R-005",
                exception_type="gl_account_missing",
                line_no=line.line_no,
                severity="medium",
                detail=(f"No GL account is mapped to material group {material.material_group}."),
                evidence={"material_group": material.material_group},
            ),
        )
    return (
        account,
        "derived",
        ExceptionFinding(
            rule_id="I2P-R-005",
            exception_type="gl_account_missing",
            line_no=line.line_no,
            severity="low",
            detail=(
                f"GL account not stated; derived {account} from material group "
                f"{material.material_group}."
            ),
            evidence={
                "material_id": material_id,
                "material_group": material.material_group,
                "derived_gl_account": account,
            },
        ),
    )


def derive_cost_center(
    line: InvoiceLine,
    invoice: Invoice,
    po_line: PurchaseOrderLine | None,
) -> tuple[str | None, str, ExceptionFinding | None]:
    """Return (cost centre, source, finding).

    Order matters and it is not arbitrary. A stated cost centre is used. Failing
    that, the purchase order's cost centre is authoritative — the requisition
    already carried the account assignment, so an MM invoice rarely needs
    anything else.

    What this function deliberately does *not* do is read the free-text note.
    Extracting a cost centre from prose is the one part of this process that is
    genuinely a language problem, and it is handled in the agent layer, on a
    closed list of cost centres that exist, with the proposal routed to a person.
    A regex over free text would be a worse version of the same thing wearing a
    deterministic costume.
    """
    if line.cost_center:
        return line.cost_center, "stated", None
    if po_line is not None:
        return po_line.cost_center, "derived_from_po", None
    return (
        None,
        "unresolved",
        ExceptionFinding(
            rule_id="I2P-R-006",
            exception_type="cost_center_missing",
            line_no=line.line_no,
            severity="medium",
            detail=(
                "No cost centre in the coding block and no purchase order to inherit one "
                "from."
                + (
                    " A free-text note accompanies the invoice."
                    if invoice.free_text
                    else " No free-text note accompanies the invoice."
                )
            ),
            evidence={
                "free_text": invoice.free_text,
                "valid_cost_centers": [
                    c.cost_center for c in COST_CENTERS if c.company_code == invoice.company_code
                ],
            },
        ),
    )


# ---------------------------------------------------------------------------
# Three-way match
# ---------------------------------------------------------------------------


def check_goods_receipt(
    line: InvoiceLine,
    received_by_cutoff: float,
    received_total: float,
    config: I2PConfig,
) -> ExceptionFinding | None:
    """Was a goods receipt posted, and was it posted in time?

    ``received_by_cutoff`` is the quantity confirmed received by the end of the
    grace period — the invoice's received date plus ``gr_grace_days``. Invoices
    routinely arrive before the receiving clerk posts the receipt, and treating
    that ordinary timing as an exception would bury the queue.

    Past the cutoff the two cases are deliberately not distinguished in the
    outcome: a receipt that never existed and a receipt posted three weeks late
    are the same control failure, because in both the organisation contemplated
    paying for goods it had not confirmed receiving. They are distinguished in
    the *detail*, because the remedy differs.
    """
    if received_by_cutoff > 0:
        return None

    eventually = received_total > 0
    return ExceptionFinding(
        rule_id="I2P-R-007",
        exception_type="missing_or_delayed_goods_receipt",
        line_no=line.line_no,
        severity="high",
        detail=(
            (
                "Goods receipt was posted only after the grace period; nothing was "
                "confirmed received when the invoice fell due for processing."
            )
            if eventually
            else "No goods receipt has been posted against this purchase-order line."
        ),
        evidence={
            "received_by_cutoff": received_by_cutoff,
            "received_total": received_total,
            "grace_days": config.gr_grace_days,
            "receipt_eventually_posted": eventually,
        },
    )


def check_quantity(
    line: InvoiceLine,
    material_id: str,
    received_base_qty: float,
    already_invoiced_base_qty: float,
    config: I2PConfig,
) -> QuantityComparison:
    """Is the invoiced quantity covered by what was received?

    Both sides are converted to base units first, for the same reason prices are:
    an invoice in cartons against a receipt in pieces is not a comparison until
    they are in the same unit.
    """
    invoiced = to_base_quantity(line.quantity, material_id, line.uom)
    open_qty = round(received_base_qty - already_invoiced_base_qty, 6)
    residual = round(invoiced - open_qty, 6)
    allowance = open_qty * config.quantity_tolerance_pct / 100.0
    within = residual <= allowance + 1e-9

    return QuantityComparison(
        invoiced_base_qty=invoiced,
        received_base_qty=received_base_qty,
        open_base_qty=open_qty,
        residual_base_qty=residual,
        within_tolerance=within,
        tolerance_pct=config.quantity_tolerance_pct,
    )


def quantity_variance_finding(
    line: InvoiceLine, comparison: QuantityComparison
) -> ExceptionFinding | None:
    """Turn a quantity residual into an exception, or not."""
    if comparison.within_tolerance:
        return None
    return ExceptionFinding(
        rule_id="I2P-R-009",
        exception_type="quantity_variance",
        line_no=line.line_no,
        severity="high",
        detail=(
            f"Invoiced {comparison.invoiced_base_qty:,.3f} base units against "
            f"{comparison.open_base_qty:,.3f} available; over-billed by "
            f"{comparison.residual_base_qty:,.3f}."
        ),
        evidence=comparison.model_dump(mode="json"),
    )


def compare_price(
    line: InvoiceLine,
    po_line: PurchaseOrderLine,
    material_id: str,
    matched_base_qty: float,
    config: I2PConfig,
) -> PriceComparison:
    """Compare the invoice's price to the purchase order's, normalised.

    This is the check the module exists for, so it is worth stating exactly what
    it does and does not do.

    Both sides are reduced to a net price for one base unit — discounts applied
    in sequence, price unit divided out, per-unit surcharge added, unit of
    measure converted — and only then subtracted. The two sides are normalised
    the *same way* by the *same function*; a comparison where each side is
    prepared differently is not a comparison.

    The residual is reported three ways because a reviewer needs all three: per
    unit in absolute terms, as a percentage, and multiplied up to the line. Four
    percent is not a decision until you know whether it is four cents or four
    thousand.

    ``naive_residual_abs`` records what subtracting the printed prices would have
    produced. It decides nothing. It is carried so the reviewer can see what the
    normalisation changed, and so the test suite can assert the two differ on the
    cases where it matters.
    """
    po_unit = normalise_unit_price(po_line.price, material_id, po_line.uom)
    invoice_unit = normalise_unit_price(line.price, material_id, line.uom)

    residual_abs = round(invoice_unit - po_unit, 6)
    residual_pct = round(residual_percentage(po_unit, invoice_unit), 6)
    line_residual = round(residual_abs * matched_base_qty, 2)
    naive_residual = round(naive_unit_price(line.price) - naive_unit_price(po_line.price), 6)

    within = _within_price_tolerance(residual_abs, residual_pct, line_residual, config)

    return PriceComparison(
        po_unit_price_normalised=po_unit,
        invoice_unit_price_normalised=invoice_unit,
        residual_abs=residual_abs,
        residual_pct=residual_pct,
        line_residual_abs=line_residual,
        naive_residual_abs=naive_residual,
        within_tolerance=within,
        tolerance_pct=config.price_tolerance_pct,
        tolerance_abs=config.price_tolerance_abs,
    )


def price_variance_finding(
    line: InvoiceLine, comparison: PriceComparison, config: I2PConfig
) -> ExceptionFinding | None:
    """Turn a price residual into an exception, or not.

    Separate from :func:`compare_price` because computing a residual and
    deciding what it means are different acts, taken against different inputs:
    the residual is arithmetic over two documents, the decision is arithmetic
    over the residual and a configured tolerance that a process owner can
    change without touching the comparison. They are separate pipeline steps
    and separate trace records for the same reason.
    """
    if comparison.within_tolerance:
        return None
    line_residual = comparison.line_residual_abs
    return ExceptionFinding(
        rule_id="I2P-R-010",
        exception_type="price_variance",
        line_no=line.line_no,
        severity="medium" if abs(line_residual) < config.price_tolerance_abs * 20 else "high",
        detail=(
            f"Normalised unit price {comparison.invoice_unit_price_normalised:,.6f} against "
            f"purchase order {comparison.po_unit_price_normalised:,.6f}: "
            f"{comparison.residual_pct:+.2f}% ({line_residual:+,.2f} on the line), "
            f"outside a tolerance of {config.price_tolerance_pct}% "
            f"or {config.price_tolerance_abs:,.2f}."
        ),
        evidence=comparison.model_dump(mode="json"),
    )


def _within_price_tolerance(
    residual_abs: float,
    residual_pct: float,
    line_residual_abs: float,
    config: I2PConfig,
) -> bool:
    """A line passes if it is inside *either* limit.

    Both limits alone are wrong in opposite directions. The percentage alone
    blocks a 0.40 difference on a 1.20 unit price, which is 33% and immaterial.
    The absolute alone waves through 400 on a 20,000 line, which is 2% and real
    money. Requiring both to be breached is what makes the exception queue about
    pricing disputes rather than about rounding — and it is a configuration
    choice, recorded in config/thresholds.yaml where a process owner can argue
    with it.
    """
    if abs(residual_pct) <= config.price_tolerance_pct:
        return True
    return abs(line_residual_abs) <= config.price_tolerance_abs


# ---------------------------------------------------------------------------
# Vendor bank details
# ---------------------------------------------------------------------------


def check_bank_details(invoice: Invoice, vendor: Vendor | None) -> ExceptionFinding | None:
    """Does the invoice's bank account match the vendor master record?

    A payment-redirection fraud is not a data-quality problem and it does not
    become one because the rest of the invoice is clean. This check compares to
    the vendor master and to nothing else, and what happens to its finding is
    fixed in the routing layer rather than left to the model's confidence.
    """
    if vendor is None:
        return ExceptionFinding(
            rule_id="I2P-R-001",
            exception_type="bank_details_mismatch",
            severity="high",
            detail=f"Vendor {invoice.vendor_id} is not in the vendor master.",
            evidence={"vendor_id": invoice.vendor_id},
        )
    if invoice.stated_bank_iban == vendor.bank_iban:
        return None
    return ExceptionFinding(
        rule_id="I2P-R-001",
        exception_type="bank_details_mismatch",
        severity="high",
        detail=(
            "Bank details on the invoice differ from the vendor master record. "
            "Payment must not be made to the stated account without verification "
            "through an independently held contact."
        ),
        evidence={
            "vendor_id": vendor.vendor_id,
            "vendor_name": vendor.name,
            "master_iban": vendor.bank_iban,
            "stated_iban": invoice.stated_bank_iban,
            "free_text": invoice.free_text,
        },
    )


__all__ = [
    "check_bank_details",
    "check_duplicate",
    "check_goods_receipt",
    "check_quantity",
    "classify_invoice",
    "compare_price",
    "derive_cost_center",
    "derive_gl_account",
    "normalise_reference",
    "price_variance_finding",
    "quantity_variance_finding",
    "reference_similarity",
    "resolve_material",
    "resolve_tax_code",
]


# ---------------------------------------------------------------------------
# Extraction confidence
# ---------------------------------------------------------------------------

#: Logical load-bearing field names, expanded to the concrete paths they cover.
#: Logical names because a controller reviewing the configuration should be able
#: to read "unit_price" rather than "lines[3].price.list_price", and because the
#: number of lines is a property of the document, not of the policy.
LOAD_BEARING_EXPANSION: dict[str, tuple[str, ...]] = {
    "stated_total_gross": ("stated_total_gross",),
    "stated_total_net": ("stated_total_net",),
    "stated_total_tax": ("stated_total_tax",),
    "currency": ("currency",),
    "vendor_id": ("vendor_id",),
    "stated_bank_iban": ("stated_bank_iban",),
    "vendor_reference": ("vendor_reference",),
    "company_code": ("company_code",),
}

#: Load-bearing names that occur once per line rather than once per document.
LOAD_BEARING_LINE_FIELDS: dict[str, str] = {
    "quantity": "quantity",
    "unit_price": "price.list_price",
    "po_reference": "po_id",
    "tax_rate": "tax_rate",
}


class ExtractionGateResult(BaseModel):
    """Whether any load-bearing field was read too weakly to compute on.

    Deliberately not an :class:`~fcca.i2p.models.ExceptionFinding`. A weak
    reading is not a finding about the *invoice* — the vendor may have done
    nothing wrong and the amounts may all be correct. It is a statement about
    our own confidence in what we read, and conflating the two would put
    "the scan was blurry" into the same confusion matrix as "the vendor
    overcharged us", which measures nothing.
    """

    model_config = ConfigDict(frozen=True)

    gated: bool
    reasons: tuple[str, ...] = ()
    weakest_field: str | None = None
    weakest_confidence: float | None = None
    fields_checked: int = 0
    fields_extracted: int = 0
    threshold: float = 0.0

    @property
    def summary(self) -> str:
        if not self.fields_extracted:
            return (
                f"No extracted fields; {self.fields_checked} load-bearing field(s) are "
                "synthetic or master data and carry no confidence to gate on."
            )
        if not self.gated:
            return (
                f"All {self.fields_extracted} extracted load-bearing field(s) at or above "
                f"{self.threshold:.2f}; weakest {self.weakest_field} at "
                f"{self.weakest_confidence:.2f}."
            )
        return (
            f"{len(self.reasons)} load-bearing field(s) below {self.threshold:.2f}; "
            f"weakest {self.weakest_field} at {self.weakest_confidence:.2f}. "
            "Escalated before any model call."
        )


def load_bearing_paths(invoice: Invoice, names: tuple[str, ...]) -> dict[str, str]:
    """Concrete field paths to check, mapped back to their logical name.

    Returns path -> logical name, because the reason a gate fires must name the
    thing a person configured, not the index of the line it happened to hit.
    """
    paths: dict[str, str] = {}
    for name in names:
        for path in LOAD_BEARING_EXPANSION.get(name, ()):
            paths[path] = name
        line_field = LOAD_BEARING_LINE_FIELDS.get(name)
        if line_field is not None:
            for index in range(len(invoice.lines)):
                paths[f"lines[{index}].{line_field}"] = name
    return paths


def extraction_confidence_gate(
    invoice: Invoice,
    provenance: dict[str, FieldProvenance],
    threshold: float,
    load_bearing: tuple[str, ...],
) -> ExtractionGateResult:
    """Refuse to compute on a value we did not read well enough.

    The argument for gating *before* classification rather than after: every
    step downstream — the three-way match, the price normalisation, the
    tolerance evaluation — is arithmetic. Arithmetic on a misread number is not
    less accurate, it is unrelated to the document, and a tolerance evaluated
    against it produces a confident answer to a question nobody asked. Ordering
    the gate second means no such number reaches a comparison, and no model is
    ever asked to explain one.

    Only *extracted* fields are gated. A synthetic or master-data value has no
    confidence, and inventing one so that the gate has something to compare
    would be exactly the kind of fabricated number this module exists to keep
    out of the arithmetic.
    """
    paths = load_bearing_paths(invoice, load_bearing)
    reasons: list[str] = []
    weakest: tuple[str, float] | None = None
    extracted = 0

    for path in sorted(paths):
        record = provenance.get(path)
        if record is None or record.source != "extracted" or record.confidence is None:
            continue
        extracted += 1
        if weakest is None or record.confidence < weakest[1]:
            weakest = (paths[path], record.confidence)
        if record.confidence < threshold:
            reasons.append(f"low_confidence_field:{paths[path]}")

    # Deduplicated and ordered: three weak line quantities are one reason to
    # escalate, not three, and a reviewer reading the queue wants the field name
    # rather than a count of how many lines it appeared on.
    unique = tuple(sorted(set(reasons)))
    return ExtractionGateResult(
        gated=bool(unique),
        reasons=unique,
        weakest_field=weakest[0] if weakest else None,
        weakest_confidence=weakest[1] if weakest else None,
        fields_checked=len(paths),
        fields_extracted=extracted,
        threshold=threshold,
    )
