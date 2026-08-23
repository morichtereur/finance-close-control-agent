"""Tests for the synthetic invoice-to-pay dataset.

Two things are worth testing about a data generator. The first is that it is
reproducible, because every measured number downstream depends on it. The second
is that it is actually difficult — a generator that quietly emitted clean,
uniform invoices would let the engine score well by doing nothing, and no test
of the engine would catch it. The messiness assertions below exist to fail if
the dataset ever becomes easy.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fcca.i2p.generate_data import SCENARIO_LABELS, SCENARIO_WEIGHTS, generate
from fcca.i2p.masterdata import (
    GL_BY_MATERIAL_GROUP,
    MATERIAL_BY_SUPPLIER_ITEM,
    MATERIALS_BY_ID,
    VENDORS_BY_ID,
)
from fcca.i2p.models import GoodsReceipt, Invoice, PurchaseOrder
from fcca.i2p.pricing import naive_unit_price, normalise_unit_price
from fcca.shared.config import Settings


@pytest.fixture(scope="module")
def dataset(sandbox: Settings) -> dict[str, object]:
    """The generated dataset, read back from disk the way a consumer would."""
    directory = sandbox.i2p_data_dir

    def load(name: str) -> list[dict[str, object]]:
        return json.loads((directory / name).read_text())

    invoices = [Invoice.model_validate(row) for row in load("invoices.json")]
    orders = {
        row["po_id"]: PurchaseOrder.model_validate(row) for row in load("purchase_orders.json")
    }
    receipts = [GoodsReceipt.model_validate(row) for row in load("goods_receipts.json")]
    labels = {
        row["invoice_id"]: row for row in json.loads(Path(sandbox.i2p_labels_path).read_text())
    }
    return {"invoices": invoices, "orders": orders, "receipts": receipts, "labels": labels}


# ------------------------------------------------------------------- shape
class TestDatasetShape:
    def test_generates_roughly_three_hundred_invoices(self, dataset: dict[str, object]) -> None:
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        assert 280 <= len(invoices) <= 360

    def test_every_invoice_has_a_label(self, dataset: dict[str, object]) -> None:
        invoices = dataset["invoices"]
        labels = dataset["labels"]
        assert isinstance(invoices, list) and isinstance(labels, dict)
        assert {invoice.invoice_id for invoice in invoices} == set(labels)

    def test_every_seeded_exception_type_is_present(self, dataset: dict[str, object]) -> None:
        """All six specified exception types must actually occur, or the evaluation is hollow."""
        labels = dataset["labels"]
        assert isinstance(labels, dict)
        seen = {row["expected_exception"] for row in labels.values()}
        assert seen == {
            "no_exception",
            "missing_or_delayed_goods_receipt",
            "duplicate_invoice",
            "price_variance",
            "cost_center_missing",
            "gl_account_missing",
            "bank_details_mismatch",
        }

    def test_both_invoice_categories_are_present(self, dataset: dict[str, object]) -> None:
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        categories = Counter(invoice.category for invoice in invoices)
        assert categories["MM"] > 50
        assert categories["FI"] > 20

    def test_the_population_is_mostly_ordinary(self, dataset: dict[str, object]) -> None:
        """An accounts-payable population is not a balanced test set."""
        labels = dataset["labels"]
        assert isinstance(labels, dict)
        clean = sum(1 for row in labels.values() if row["expected_exception"] == "no_exception")
        assert clean / len(labels) > 0.5

    def test_scenario_table_and_label_table_agree(self) -> None:
        assert set(SCENARIO_WEIGHTS) == set(SCENARIO_LABELS)


# ----------------------------------------------------------- reproducibility
class TestReproducibility:
    def test_the_same_seed_produces_an_identical_dataset(
        self, tmp_path: Path, sandbox: Settings
    ) -> None:
        """Every measured number in the README depends on this holding."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        digests = []
        for base in (first, second):
            settings = sandbox.model_copy(update={"base_dir": base})
            generate(settings)
            digests.append((settings.i2p_data_dir / "invoices.json").read_text())
        assert digests[0] == digests[1]

    def test_a_different_seed_produces_a_different_dataset(
        self, tmp_path: Path, sandbox: Settings
    ) -> None:
        settings_a = sandbox.model_copy(update={"base_dir": tmp_path / "a", "random_seed": 1})
        settings_b = sandbox.model_copy(update={"base_dir": tmp_path / "b", "random_seed": 2})
        generate(settings_a)
        generate(settings_b)
        assert (settings_a.i2p_data_dir / "invoices.json").read_text() != (
            settings_b.i2p_data_dir / "invoices.json"
        ).read_text()


# ------------------------------------------------------------------ messiness
class TestMessiness:
    """If these ever fail, the dataset has become too easy to be worth scoring."""

    @staticmethod
    def _lines(dataset: dict[str, object]) -> list[object]:
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        return [line for invoice in invoices for line in invoice.lines]

    def test_cascading_discounts_of_three_steps_occur(self, dataset: dict[str, object]) -> None:
        lines = self._lines(dataset)
        assert sum(1 for line in lines if len(line.price.discount_pct) >= 3) >= 10

    def test_per_unit_surcharges_occur(self, dataset: dict[str, object]) -> None:
        lines = self._lines(dataset)
        assert sum(1 for line in lines if line.price.surcharge_per_unit) >= 20

    def test_price_units_other_than_one_occur(self, dataset: dict[str, object]) -> None:
        lines = self._lines(dataset)
        assert sum(1 for line in lines if line.price.price_unit > 1) >= 20

    def test_alternative_units_of_measure_occur(self, dataset: dict[str, object]) -> None:
        lines = self._lines(dataset)
        alternative = sum(
            1
            for line in lines
            if line.uom
            != MATERIALS_BY_ID[MATERIAL_BY_SUPPLIER_ITEM[line.supplier_item_no]].base_uom
        )
        assert alternative >= 20

    def test_tax_rates_are_mixed_within_documents(self, dataset: dict[str, object]) -> None:
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        assert sum(1 for i in invoices if len({line.tax_rate for line in i.lines}) > 1) >= 10

    def test_supplier_item_numbers_differ_from_our_material_codes(
        self, dataset: dict[str, object]
    ) -> None:
        lines = self._lines(dataset)
        assert all(
            line.supplier_item_no != MATERIAL_BY_SUPPLIER_ITEM[line.supplier_item_no]
            for line in lines
        )

    def test_partial_deliveries_occur(self, dataset: dict[str, object]) -> None:
        receipts = dataset["receipts"]
        orders = dataset["orders"]
        assert isinstance(receipts, list) and isinstance(orders, dict)
        by_line: Counter[tuple[str, int]] = Counter()
        for receipt in receipts:
            by_line[(receipt.po_id, receipt.po_line)] += 1
        assert sum(1 for count in by_line.values() if count > 1) >= 10


# ---------------------------------------------------- the naive-comparison trap
class TestTheNaiveComparisonTrap:
    """The dataset's reason for existing: printed prices disagree, real prices do not."""

    def test_many_non_exception_lines_have_a_misleading_printed_price(
        self, dataset: dict[str, object]
    ) -> None:
        invoices = dataset["invoices"]
        orders = dataset["orders"]
        labels = dataset["labels"]
        assert isinstance(invoices, list) and isinstance(orders, dict)
        assert isinstance(labels, dict)

        misleading = 0
        for invoice in invoices:
            if labels[invoice.invoice_id]["expected_exception"] != "no_exception":
                continue
            for line in invoice.lines:
                if line.po_id is None or line.po_line is None:
                    continue
                po_line = orders[line.po_id].line(line.po_line)
                assert po_line is not None
                if abs(naive_unit_price(line.price) - naive_unit_price(po_line.price)) > 1e-6:
                    misleading += 1

        assert misleading >= 40, (
            "The dataset must contain many clean lines whose printed prices disagree; "
            "otherwise normalisation has nothing to prove."
        )

    def test_clean_restated_lines_normalise_to_the_purchase_order_price(
        self, dataset: dict[str, object]
    ) -> None:
        """Same money, different presentation, and the normalised prices agree exactly."""
        invoices = dataset["invoices"]
        orders = dataset["orders"]
        labels = dataset["labels"]
        assert isinstance(invoices, list) and isinstance(orders, dict)
        assert isinstance(labels, dict)

        checked = 0
        for invoice in invoices:
            if labels[invoice.invoice_id]["scenario"] != "clean_mm_messy_pricing":
                continue
            for line in invoice.lines:
                assert line.po_id is not None and line.po_line is not None
                po_line = orders[line.po_id].line(line.po_line)
                assert po_line is not None
                material_id = MATERIAL_BY_SUPPLIER_ITEM[line.supplier_item_no]
                po_price = normalise_unit_price(po_line.price, material_id, po_line.uom)
                invoice_price = normalise_unit_price(line.price, material_id, line.uom)
                assert invoice_price == pytest.approx(po_price, abs=1e-6)
                checked += 1
        assert checked >= 20


# ------------------------------------------------------------ referential sanity
class TestReferentialIntegrity:
    def test_every_mm_line_points_at_a_real_purchase_order_line(
        self, dataset: dict[str, object]
    ) -> None:
        invoices = dataset["invoices"]
        orders = dataset["orders"]
        assert isinstance(invoices, list) and isinstance(orders, dict)
        for invoice in invoices:
            if invoice.category != "MM":
                continue
            for line in invoice.lines:
                assert line.po_id in orders
                assert orders[line.po_id].line(line.po_line) is not None

    def test_every_invoice_names_a_known_vendor(self, dataset: dict[str, object]) -> None:
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        assert all(invoice.vendor_id in VENDORS_BY_ID for invoice in invoices)

    def test_stated_gl_accounts_are_the_ones_the_material_group_implies(
        self, dataset: dict[str, object]
    ) -> None:
        """Where a GL account is stated it must be right, or derivation cannot be scored."""
        invoices = dataset["invoices"]
        assert isinstance(invoices, list)
        for invoice in invoices:
            for line in invoice.lines:
                if line.gl_account is None:
                    continue
                material = MATERIALS_BY_ID[MATERIAL_BY_SUPPLIER_ITEM[line.supplier_item_no]]
                assert line.gl_account == GL_BY_MATERIAL_GROUP[material.material_group]

    def test_bank_mismatch_cases_really_differ_from_vendor_master(
        self, dataset: dict[str, object]
    ) -> None:
        invoices = dataset["invoices"]
        labels = dataset["labels"]
        assert isinstance(invoices, list) and isinstance(labels, dict)
        checked = 0
        for invoice in invoices:
            if labels[invoice.invoice_id]["expected_exception"] != "bank_details_mismatch":
                continue
            assert invoice.stated_bank_iban != VENDORS_BY_ID[invoice.vendor_id].bank_iban
            checked += 1
        assert checked >= 5

    def test_non_mismatch_cases_use_the_vendor_master_account(
        self, dataset: dict[str, object]
    ) -> None:
        invoices = dataset["invoices"]
        labels = dataset["labels"]
        assert isinstance(invoices, list) and isinstance(labels, dict)
        for invoice in invoices:
            if labels[invoice.invoice_id]["expected_exception"] == "bank_details_mismatch":
                continue
            assert invoice.stated_bank_iban == VENDORS_BY_ID[invoice.vendor_id].bank_iban
