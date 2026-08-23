"""In-memory access to the invoice-to-pay dataset.

Deliberately not a database. The close module uses DuckDB because it runs
population-level analytics — variance against a prior-period average, duplicate
candidates across a whole ledger. The invoice side asks pointed questions about
one document and the handful of records it references, and loading a few hundred
structured documents into memory answers those directly, keeps the
deterministic layer free of SQL, and makes every check a pure function of
objects a test can construct by hand.

The one population-level question — has this invoice been seen before — is
answered by an index built once at load time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from functools import cached_property
from pathlib import Path

from fcca.i2p.models import (
    CostCenter,
    GoodsReceipt,
    Invoice,
    Material,
    PurchaseOrder,
    Vendor,
)
from fcca.i2p.pricing import to_base_quantity
from fcca.shared.config import Settings, get_settings
from fcca.shared.errors import DataNotFoundError


class I2PRepository:
    """Loads and indexes one generated dataset."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.directory = self.settings.i2p_data_dir
        if not (self.directory / "invoices.json").exists():
            raise DataNotFoundError(
                f"no invoice dataset at {self.directory}; run `fcca i2p-generate-data` first"
            )

    # ------------------------------------------------------------------ load
    def _load(self, name: str) -> list[dict[str, object]]:
        path: Path = self.directory / name
        if not path.exists():
            raise DataNotFoundError(f"missing dataset file {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        return payload

    @cached_property
    def invoices(self) -> dict[str, Invoice]:
        rows = [Invoice.model_validate(row) for row in self._load("invoices.json")]
        return {row.invoice_id: row for row in rows}

    @cached_property
    def purchase_orders(self) -> dict[str, PurchaseOrder]:
        rows = [PurchaseOrder.model_validate(row) for row in self._load("purchase_orders.json")]
        return {row.po_id: row for row in rows}

    @cached_property
    def vendors(self) -> dict[str, Vendor]:
        rows = [Vendor.model_validate(row) for row in self._load("vendors.json")]
        return {row.vendor_id: row for row in rows}

    @cached_property
    def materials(self) -> dict[str, Material]:
        rows = [Material.model_validate(row) for row in self._load("materials.json")]
        return {row.material_id: row for row in rows}

    @cached_property
    def cost_centers(self) -> dict[str, CostCenter]:
        rows = [CostCenter.model_validate(row) for row in self._load("cost_centers.json")]
        return {row.cost_center: row for row in rows}

    @cached_property
    def goods_receipts(self) -> dict[tuple[str, int], list[GoodsReceipt]]:
        """Receipts grouped by purchase-order line.

        A list, not a single record: partial delivery is the normal case, and a
        quantity check that assumed one receipt per line would understate what
        is available to invoice.
        """
        grouped: dict[tuple[str, int], list[GoodsReceipt]] = defaultdict(list)
        for row in self._load("goods_receipts.json"):
            receipt = GoodsReceipt.model_validate(row)
            grouped[(receipt.po_id, receipt.po_line)].append(receipt)
        for receipts in grouped.values():
            receipts.sort(key=lambda r: r.receipt_date)
        return dict(grouped)

    @cached_property
    def _invoices_by_vendor(self) -> dict[str, list[Invoice]]:
        grouped: dict[str, list[Invoice]] = defaultdict(list)
        for invoice in self.invoices.values():
            grouped[invoice.vendor_id].append(invoice)
        for group in grouped.values():
            group.sort(key=lambda i: (i.invoice_date, i.invoice_id))
        return dict(grouped)

    # ---------------------------------------------------------------- lookups
    def invoice(self, invoice_id: str) -> Invoice:
        try:
            return self.invoices[invoice_id]
        except KeyError:
            raise DataNotFoundError(f"no invoice {invoice_id!r}") from None

    def purchase_order(self, po_id: str) -> PurchaseOrder | None:
        return self.purchase_orders.get(po_id)

    def vendor(self, vendor_id: str) -> Vendor | None:
        return self.vendors.get(vendor_id)

    def receipts_for(self, po_id: str, po_line: int) -> list[GoodsReceipt]:
        return self.goods_receipts.get((po_id, po_line), [])

    def received_base_quantity(
        self,
        po_id: str,
        po_line: int,
        material_id: str,
        as_of: date | None = None,
    ) -> float:
        """Total received against a purchase-order line, in **base units**.

        Converted rather than summed raw. Receipts are posted in whatever unit
        the receiving clerk used, which is not necessarily the material's base
        unit and not necessarily the invoice's — summing them as printed and
        comparing to an invoiced base quantity compares two different things and
        manufactures a quantity variance on every line where the units differ.

        ``as_of`` matters: a receipt posted after the invoice fell due does not
        make the invoice matched at the time payment was contemplated, and
        treating it as though it did would hide the delayed-receipt exception.
        """
        receipts = self.receipts_for(po_id, po_line)
        if as_of is not None:
            receipts = [r for r in receipts if r.receipt_date <= as_of]
        return round(sum(to_base_quantity(r.quantity, material_id, r.uom) for r in receipts), 6)

    def earlier_invoices(self, invoice: Invoice, window_days: int) -> list[Invoice]:
        """Other invoices from the same vendor within the look-back window.

        Ordered and filtered to documents that arrived *before* this one, so a
        duplicate pair produces one finding on the resubmission rather than two
        findings that each accuse the other.
        """
        candidates = self._invoices_by_vendor.get(invoice.vendor_id, [])
        result = []
        for other in candidates:
            if other.invoice_id == invoice.invoice_id:
                continue
            delta = (invoice.invoice_date - other.invoice_date).days
            if 0 <= delta <= window_days:
                # Same-day pairs are ordered by id so exactly one of them is "earlier".
                if delta == 0 and other.invoice_id > invoice.invoice_id:
                    continue
                result.append(other)
        return result

    def summary(self) -> dict[str, int]:
        return {
            "invoices": len(self.invoices),
            "purchase_orders": len(self.purchase_orders),
            "goods_receipt_lines": len(self.goods_receipts),
            "vendors": len(self.vendors),
            "materials": len(self.materials),
            "cost_centers": len(self.cost_centers),
        }


__all__ = ["I2PRepository"]
