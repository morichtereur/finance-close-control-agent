"""ERP posting: the port, the payload shape, and the ledger that makes it idempotent.

Like :mod:`fcca.i2p.extraction`, this is a seam and not an integration. Nothing
here opens a socket.

What is actually worth building without a connection
----------------------------------------------------
A demo that posts to a sandbox ERP proves that a sandbox accepted a request. The
parts that are hard, and that a reviewer can actually check, are the parts that
have nothing to do with the network:

* **the mapping** — which invoice field becomes which SAP field, what gets
  derived, and what has no home in the target schema at all;
* **idempotency** — whether replaying the same document twice produces one
  posting or two, which is the difference between a control and a hazard;
* **the gate** — that nothing outside ``auto_clear`` ever produces a payload.

All three are implemented here and all three are tested. The connection is the
one part that would be straightforward on the day someone has credentials.

.. warning::

   :class:`SapODataTarget` **cannot dispatch**. ``dry_run`` is not a constructor
   argument, not a setting and not in the YAML; it is a read-only property that
   returns ``True``, and :meth:`SapODataTarget.dispatch` raises
   unconditionally. There is deliberately no code path — not even a disabled
   one — that sends an HTTP request to an ERP, because a flag that can be
   flipped is a flag that gets flipped. Making it real means writing the
   transport, and that is a decision someone should have to make explicitly.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from fcca.i2p.extraction import FieldProvenance
from fcca.i2p.models import Invoice, InvoiceResult
from fcca.shared.errors import FCCAError


class PostingBlocked(FCCAError):
    """A posting was attempted that the design does not permit.

    Raised for two different mistakes, both of which should fail loudly: asking
    a dry-run adapter to dispatch, and building a payload for an invoice that
    was not routed ``auto_clear``.
    """


def fiscal_year(document_date: date) -> int:
    """Fiscal year for a document.

    Calendar year, which is what the synthetic company codes use. A real
    deployment reads this from the company code's fiscal-year variant, and that
    is exactly the kind of thing this function exists to make findable rather
    than leaving ``document_date.year`` inline at the call site.
    """
    return document_date.year


def posting_key(vendor_id: str, vendor_reference: str, document_date: date) -> str:
    """The deterministic identity of a posting.

    Vendor, the vendor's own document number, and fiscal year — the same triple
    SAP uses to refuse a duplicate supplier invoice, and for the same reason:
    it is the only combination the *vendor* controls end to end, so it survives
    our side re-reading, re-extracting or re-numbering the document.

    Deliberately not a hash of the payload. Two extraction passes over the same
    scan can differ in a rounding or a whitespace and would hash differently,
    which would make a replay look like a new document — the precise failure the
    key exists to prevent. Normalised for case and internal whitespace because
    "INV 4471" and "inv-4471" are the same invoice.
    """
    reference = "".join(vendor_reference.split()).upper().replace("-", "")
    return f"{vendor_id.upper()}|{reference}|{fiscal_year(document_date)}"


class PostingPayload(BaseModel):
    """What would be sent, and where every value in it came from.

    ``document`` is the body an ERP would receive. ``mapping`` is the audit of
    how it was built: for each target field, the invoice field it came from and
    that field's provenance. The mapping is the part with review value — it is
    where someone who knows the target system says "that is the wrong field".
    """

    model_config = ConfigDict(frozen=True)

    invoice_id: str
    posting_key: str
    target: str = Field(description="Adapter that built this, e.g. 'sap-odata'.")
    service: str = Field(description="Target service or API name.")
    dry_run: bool
    document: dict[str, Any] = Field(description="The body that would be dispatched.")
    mapping: tuple[FieldMapping, ...]
    built_at: datetime

    @property
    def line_count(self) -> int:
        items = self.document.get("to_SuplrInvcItemPurOrdRef", {})
        results = items.get("results", []) if isinstance(items, dict) else []
        return len(results)


class FieldMapping(BaseModel):
    """One target field, and the invoice field behind it."""

    model_config = ConfigDict(frozen=True)

    target_field: str = Field(description="Field in the ERP schema, e.g. 'InvoiceGrossAmount'.")
    source_path: str | None = Field(
        default=None,
        description="Invoice field path it came from; None where the value is derived or constant.",
    )
    value: Any
    source_value: Any = Field(
        default=None, description="The value on the invoice, before any transformation."
    )
    provenance: FieldProvenance | None = None
    note: str = ""

    @property
    def transformed(self) -> bool:
        """Whether the posted value differs from what was on the document.

        This is what the detail pane's diff column reads. A transformation is
        not suspicious in itself — a date reformatted for the target schema is
        fine — but a *value* that changed on the way to the ledger is the thing
        a reviewer should be shown rather than have to go looking for.
        """
        return self.source_value is not None and self.value != self.source_value


PostingPayload.model_rebuild()


@runtime_checkable
class PostingTarget(Protocol):
    """Somewhere a cleared invoice would go."""

    name: str

    @property
    def dry_run(self) -> bool:
        """Whether this target only ever builds payloads."""
        ...

    def build(
        self,
        invoice: Invoice,
        result: InvoiceResult,
        provenance: dict[str, FieldProvenance],
    ) -> PostingPayload:
        """Map one cleared invoice to a payload. Must not dispatch."""
        ...

    def dispatch(self, payload: PostingPayload) -> None:
        """Send a payload. Simulators raise."""
        ...


def _require_auto_clear(result: InvoiceResult) -> None:
    """No payload exists for an invoice a person still has to look at.

    Enforced in the adapters rather than at the call site, so a future caller
    cannot forget it. This is the invariant behind the ``false_auto_post``
    metric: if nothing outside ``auto_clear`` can produce a payload, a false
    auto-post requires a routing error rather than merely a plumbing one.
    """
    if result.routing.tier != "auto_clear":
        raise PostingBlocked(
            f"{result.invoice_id} is routed {result.routing.tier!r}, not 'auto_clear'; "
            "no posting payload may be built for it"
        )


class SimulatedPosting:
    """Records that a posting would have happened. Builds no ERP-specific shape.

    The behaviour the repository had before there was a port: the tier is
    recorded, the trace shows why, and nothing else occurs.
    """

    name = "simulated"

    @property
    def dry_run(self) -> bool:
        return True

    def build(
        self,
        invoice: Invoice,
        result: InvoiceResult,
        provenance: dict[str, FieldProvenance],
    ) -> PostingPayload:
        _require_auto_clear(result)
        key = posting_key(invoice.vendor_id, invoice.vendor_reference, invoice.invoice_date)
        return PostingPayload(
            invoice_id=invoice.invoice_id,
            posting_key=key,
            target=self.name,
            service="none",
            dry_run=True,
            document={
                "invoice_id": invoice.invoice_id,
                "vendor_id": invoice.vendor_id,
                "gross": invoice.stated_total_gross,
                "currency": invoice.currency,
            },
            mapping=(),
            built_at=datetime.now(UTC),
        )

    def dispatch(self, payload: PostingPayload) -> None:
        raise PostingBlocked(
            "SimulatedPosting never dispatches. Nothing in this repository writes to a "
            "financial system."
        )


class SapODataTarget:
    """Maps an invoice to the shape ``API_SUPPLIERINVOICE_PROCESS_SRV`` expects.

    Written against the real S/4HANA service so the mapping is reviewable by
    someone who knows it: ``A_SupplierInvoice`` header fields, with purchase-order
    references in the ``to_SuplrInvcItemPurOrdRef`` navigation property. Field
    names, the ``/Date(ms)/`` serialisation and the string-typed amounts are the
    service's conventions, not this repository's preferences.

    Two things a real integration needs that are deliberately absent, because
    guessing at them would be worse than their absence: a CSRF token fetch, and
    the company-code-specific fiscal-year variant. Both are noted in the mapping
    output rather than silently defaulted.
    """

    name = "sap-odata"
    service = "API_SUPPLIERINVOICE_PROCESS_SRV"

    #: Not a constructor argument, not a setting, not in the YAML. See the
    #: module warning: a flag that can be flipped is a flag that gets flipped.
    @property
    def dry_run(self) -> bool:
        return True

    def build(
        self,
        invoice: Invoice,
        result: InvoiceResult,
        provenance: dict[str, FieldProvenance],
    ) -> PostingPayload:
        _require_auto_clear(result)

        def prov(path: str) -> FieldProvenance | None:
            return provenance.get(path)

        mapping: list[FieldMapping] = [
            FieldMapping(
                target_field="CompanyCode",
                source_path="company_code",
                value=invoice.company_code,
                source_value=invoice.company_code,
                provenance=prov("company_code"),
            ),
            FieldMapping(
                target_field="DocumentDate",
                source_path="invoice_date",
                value=_odata_date(invoice.invoice_date),
                source_value=invoice.invoice_date.isoformat(),
                provenance=prov("invoice_date"),
                note="Serialised to the OData /Date(ms)/ form the service requires.",
            ),
            FieldMapping(
                target_field="PostingDate",
                source_path="received_date",
                value=_odata_date(invoice.received_date),
                source_value=invoice.received_date.isoformat(),
                provenance=prov("received_date"),
                note="Posting date is the date of receipt, not the vendor's document date.",
            ),
            FieldMapping(
                target_field="InvoicingParty",
                source_path="vendor_id",
                value=invoice.vendor_id,
                source_value=invoice.vendor_id,
                provenance=prov("vendor_id"),
            ),
            FieldMapping(
                target_field="DocumentCurrency",
                source_path="currency",
                value=invoice.currency,
                source_value=invoice.currency,
                provenance=prov("currency"),
            ),
            FieldMapping(
                target_field="InvoiceGrossAmount",
                source_path="stated_total_gross",
                value=f"{invoice.stated_total_gross:.2f}",
                source_value=invoice.stated_total_gross,
                provenance=prov("stated_total_gross"),
                note="The service types amounts as strings.",
            ),
            FieldMapping(
                target_field="SupplierInvoiceIDByInvcgParty",
                source_path="vendor_reference",
                value=invoice.vendor_reference,
                source_value=invoice.vendor_reference,
                provenance=prov("vendor_reference"),
                note="The vendor's own number. Part of the duplicate key on both sides.",
            ),
            FieldMapping(
                target_field="FiscalYear",
                source_path=None,
                value=str(fiscal_year(invoice.invoice_date)),
                note=(
                    "Derived as the calendar year. A real deployment reads the fiscal-year "
                    "variant from the company code; that lookup does not exist here."
                ),
            ),
        ]

        items: list[dict[str, Any]] = []
        for position, (line, resolution) in enumerate(
            zip(invoice.lines, result.resolutions, strict=False), start=1
        ):
            items.append(
                {
                    "SupplierInvoiceItem": f"{position}",
                    "PurchaseOrder": line.po_id,
                    "PurchaseOrderItem": f"{line.po_line}" if line.po_line else None,
                    "DocumentCurrency": invoice.currency,
                    "SupplierInvoiceItemAmount": f"{_line_net(line):.2f}",
                    "QuantityInPurchaseOrderUnit": f"{line.quantity:.3f}",
                    "PurchaseOrderQuantityUnit": line.uom,
                    "TaxCode": resolution.tax_code,
                    "GLAccount": resolution.gl_account,
                    "CostCenter": resolution.cost_center,
                }
            )
            index = position - 1
            mapping.append(
                FieldMapping(
                    target_field=f"item[{position}].QuantityInPurchaseOrderUnit",
                    source_path=f"lines[{index}].quantity",
                    value=f"{line.quantity:.3f}",
                    source_value=line.quantity,
                    provenance=prov(f"lines[{index}].quantity"),
                )
            )
            mapping.append(
                FieldMapping(
                    target_field=f"item[{position}].GLAccount",
                    source_path=f"lines[{index}].gl_account",
                    value=resolution.gl_account,
                    source_value=line.gl_account,
                    provenance=prov(f"lines[{index}].gl_account"),
                    note=(
                        "Derived from the material group; the document did not state it."
                        if resolution.gl_source == "derived"
                        else ""
                    ),
                )
            )

        document: dict[str, Any] = {
            entry.target_field: entry.value for entry in mapping if "[" not in entry.target_field
        }
        document["to_SuplrInvcItemPurOrdRef"] = {"results": items}

        return PostingPayload(
            invoice_id=invoice.invoice_id,
            posting_key=posting_key(
                invoice.vendor_id, invoice.vendor_reference, invoice.invoice_date
            ),
            target=self.name,
            service=self.service,
            dry_run=True,
            document=document,
            mapping=tuple(mapping),
            built_at=datetime.now(UTC),
        )

    def dispatch(self, payload: PostingPayload) -> None:
        raise PostingBlocked(
            "SapODataTarget is payload-shape only and has no transport. There is no HTTP "
            "client here, no credential handling and no CSRF token fetch — not disabled, "
            "absent. Posting to a real ERP means writing that transport deliberately."
        )


def _line_net(line: Any) -> float:
    """Net amount for one line, as the target schema wants it."""
    return round(line.quantity * line.price.list_price / line.price.price_unit, 2)


def _odata_date(value: date) -> str:
    """``/Date(1719360000000)/`` — the OData v2 date literal the service uses."""
    epoch_ms = int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)
    return f"/Date({epoch_ms})/"


class PostedKeyLedger:
    """Append-only record of every posting key that has been emitted.

    This is what makes the duplicate control credible. A scan of the current
    batch catches a vendor who sends the same invoice twice in one file; it does
    not catch the far more common case of the same invoice arriving next week,
    or the same file being processed twice after a failed run. The ledger
    persists across runs, so the second sighting is a duplicate whenever it
    happens.

    JSONL for the same reason the trace is: append-only, greppable, and
    diffable in review. Nothing here rewrites a line.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def keys(self) -> dict[str, str]:
        """Posting key to the invoice that first claimed it."""
        claimed: dict[str, str] = {}
        for row in self.entries():
            claimed.setdefault(row["posting_key"], row["invoice_id"])
        return claimed

    def seen(self, key: str) -> str | None:
        """The invoice that already claimed this key, or None."""
        return self.keys().get(key)

    def record(self, payload: PostingPayload) -> bool:
        """Claim a key. Returns False if it was already claimed.

        Returning rather than raising: a repeat is an ordinary business event,
        not an error. The caller decides what it means — the pipeline raises a
        duplicate finding, a replay of the same run does nothing.
        """
        existing = self.seen(payload.posting_key)
        if existing is not None:
            return False
        row = {
            "posting_key": payload.posting_key,
            "invoice_id": payload.invoice_id,
            "target": payload.target,
            "dry_run": payload.dry_run,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return True

    def clear(self) -> None:
        """Remove the ledger. Only for tests and a deliberate dataset rebuild."""
        self.path.unlink(missing_ok=True)


def get_target(name: str) -> PostingTarget:
    """Resolve a posting target by name."""
    targets: dict[str, PostingTarget] = {
        "simulated": SimulatedPosting(),
        "sap-odata": SapODataTarget(),
    }
    try:
        return targets[name]
    except KeyError:
        raise FCCAError(
            f"unknown posting target {name!r}; available: {', '.join(sorted(targets))}"
        ) from None


__all__ = [
    "FieldMapping",
    "PostedKeyLedger",
    "PostingBlocked",
    "PostingPayload",
    "PostingTarget",
    "SapODataTarget",
    "SimulatedPosting",
    "fiscal_year",
    "get_target",
    "posting_key",
]
