"""Tests for price and quantity normalisation.

The test this module exists for is
:meth:`TestTheCaseThisModuleExistsFor.test_cascading_discounts_plus_surcharge_is_not_a_variance`.
Everything else supports it.
"""

from __future__ import annotations

import pytest

from fcca.i2p.checks import compare_price, price_variance_finding
from fcca.i2p.models import InvoiceLine, PriceElements, PurchaseOrderLine
from fcca.i2p.pricing import (
    UomConversionError,
    apply_discount_cascade,
    naive_unit_price,
    normalise_unit_price,
    residual_percentage,
    to_base_quantity,
    uom_factor,
)
from fcca.shared.config import I2PConfig

CONFIG = I2PConfig()


def po_line(price: PriceElements, **overrides: object) -> PurchaseOrderLine:
    defaults: dict[str, object] = {
        "po_line": 10,
        "material_id": "MAT-100045",
        "supplier_item_no": "NW-8840-SS",
        "quantity": 1000.0,
        "uom": "PCE",
        "price": price,
        "tax_code": "V1",
        "gl_account": "400100",
        "cost_center": "CC-1000",
    }
    defaults.update(overrides)
    return PurchaseOrderLine.model_validate(defaults)


def invoice_line(price: PriceElements, **overrides: object) -> InvoiceLine:
    defaults: dict[str, object] = {
        "line_no": 1,
        "description": "Hex bolt M8x40, stainless",
        "supplier_item_no": "NW-8840-SS",
        "quantity": 1000.0,
        "uom": "PCE",
        "price": price,
        "tax_rate": 19.0,
        "po_id": "PO-4500001",
        "po_line": 10,
        "gl_account": "400100",
        "cost_center": "CC-1000",
    }
    defaults.update(overrides)
    return InvoiceLine.model_validate(defaults)


# ===========================================================================
class TestTheCaseThisModuleExistsFor:
    """Three cascading discounts and a per-unit surcharge, presented two ways."""

    #: The purchase order: 38.50 per 100 pieces, less 5%, less 3%, less 2%,
    #: plus 0.04 per piece handling.
    PO_PRICE = PriceElements(
        list_price=38.50,
        price_unit=100,
        discount_pct=(5.0, 3.0, 2.0),
        surcharge_per_unit=0.04,
    )

    #: The vendor's invoice for exactly the same money, expressed the way their
    #: billing system prints it: one net price per piece, no discount schedule,
    #: surcharge already absorbed.
    #:
    #: 38.50 x 0.95 x 0.97 x 0.98 = 34.768195 per 100
    #: / 100                      =  0.34768195 per piece
    #: + 0.04                     =  0.38768195 per piece
    INVOICE_PRICE = PriceElements(
        list_price=0.3876819,
        price_unit=1,
        discount_pct=(),
        surcharge_per_unit=0.0,
    )

    def test_the_two_presentations_normalise_to_the_same_price(self) -> None:
        po = normalise_unit_price(self.PO_PRICE, "MAT-100045", "PCE")
        invoice = normalise_unit_price(self.INVOICE_PRICE, "MAT-100045", "PCE")
        assert po == pytest.approx(0.3876819, abs=1e-6)
        assert invoice == pytest.approx(po, abs=1e-6)

    def test_the_naive_comparison_flags_a_false_exception(self) -> None:
        """Subtracting the printed prices reports a variance of nearly 100%.

        This is the failure mode. 38.50 against 0.3876 looks like the vendor has
        billed a hundredth of the agreed price, and it is entirely an artefact of
        the price unit and the discount schedule. Either direction is a false
        exception; this one would also stop a correct invoice from being paid.
        """
        naive_difference = naive_unit_price(self.INVOICE_PRICE) - naive_unit_price(self.PO_PRICE)
        assert naive_difference == pytest.approx(-38.1123181, abs=1e-6)
        naive_pct = residual_percentage(
            naive_unit_price(self.PO_PRICE), naive_unit_price(self.INVOICE_PRICE)
        )
        assert abs(naive_pct) > CONFIG.price_tolerance_pct
        assert abs(naive_pct) > 98.0

    def test_cascading_discounts_plus_surcharge_is_not_a_variance(self) -> None:
        """The required case: normalised, this line raises no exception at all.

        A naive comparison flags it. The normalised comparison does not, because
        there is nothing wrong with it — the vendor billed the agreed price.
        """
        comparison = compare_price(
            invoice_line(self.INVOICE_PRICE),
            po_line(self.PO_PRICE),
            material_id="MAT-100045",
            matched_base_qty=1000.0,
            config=CONFIG,
        )

        assert comparison.residual_abs == pytest.approx(0.0, abs=1e-6)
        assert comparison.residual_pct == pytest.approx(0.0, abs=1e-3)
        assert comparison.within_tolerance is True
        assert price_variance_finding(invoice_line(self.INVOICE_PRICE), comparison, CONFIG) is None

        # And the naive figure is carried alongside, so a reviewer can see what
        # the normalisation did rather than having to trust that it did anything.
        assert abs(comparison.naive_residual_abs) > 38.0

    def test_a_real_variance_on_the_same_line_is_still_caught(self) -> None:
        """Normalisation must not be a way of making every difference disappear."""
        overbilled = PriceElements(
            list_price=self.INVOICE_PRICE.list_price * 1.06,
            price_unit=1,
        )
        # 20,000 pieces, so that 6% is 465 rather than 23 — otherwise the
        # absolute limit clears it, correctly, and this test would be asserting
        # the wrong thing about the wrong rule.
        comparison = compare_price(
            invoice_line(overbilled, quantity=20_000.0),
            po_line(self.PO_PRICE, quantity=20_000.0),
            material_id="MAT-100045",
            matched_base_qty=20_000.0,
            config=CONFIG,
        )
        assert comparison.residual_pct == pytest.approx(6.0, abs=0.05)
        assert comparison.within_tolerance is False
        assert abs(comparison.line_residual_abs) > CONFIG.price_tolerance_abs
        finding = price_variance_finding(invoice_line(overbilled), comparison, CONFIG)
        assert finding is not None
        assert finding.exception_type == "price_variance"
        assert finding.rule_id == "I2P-R-010"


# ===========================================================================
class TestDiscountCascade:
    def test_discounts_are_sequential_not_additive(self) -> None:
        """5 + 3 + 2 is not 10. Summing them would manufacture exceptions."""
        cascaded = apply_discount_cascade(100.0, (5.0, 3.0, 2.0))
        assert cascaded == pytest.approx(90.307, abs=1e-3)
        assert cascaded != pytest.approx(90.0, abs=1e-3)

    def test_an_empty_cascade_leaves_the_price_alone(self) -> None:
        assert apply_discount_cascade(42.0, ()) == 42.0

    def test_order_does_not_matter_among_percentages(self) -> None:
        """Multiplication commutes; the test records that we rely on it."""
        assert apply_discount_cascade(100.0, (5.0, 3.0)) == pytest.approx(
            apply_discount_cascade(100.0, (3.0, 5.0))
        )


class TestOrderOfOperations:
    """The steps that do *not* commute, which is why the order is fixed."""

    def test_the_surcharge_is_not_discounted(self) -> None:
        """Adding the surcharge before the cascade would shrink it by the discount."""
        with_surcharge = normalise_unit_price(
            PriceElements(list_price=100.0, discount_pct=(10.0,), surcharge_per_unit=5.0),
            "MAT-300121",
            "PCE",
        )
        assert with_surcharge == pytest.approx(95.0)  # 90 + 5, not (100 + 5) * 0.9 = 94.5

    def test_the_surcharge_is_not_scaled_by_the_price_unit(self) -> None:
        """Applying the surcharge before dividing would multiply it by the price unit."""
        price = normalise_unit_price(
            PriceElements(list_price=100.0, price_unit=10, surcharge_per_unit=2.0),
            "MAT-300121",
            "PCE",
        )
        assert price == pytest.approx(12.0)  # 10 + 2, not (100 + 2) / 10 = 10.2


class TestUnitOfMeasure:
    def test_the_base_unit_needs_no_conversion(self) -> None:
        assert uom_factor("MAT-100045", "PCE") == 1.0

    def test_an_alternative_unit_converts(self) -> None:
        assert uom_factor("MAT-100045", "BOX") == 250.0

    def test_a_price_per_box_becomes_a_price_per_piece(self) -> None:
        per_box = normalise_unit_price(PriceElements(list_price=250.0), "MAT-100045", "BOX")
        assert per_box == pytest.approx(1.0)

    def test_quantities_convert_the_same_way(self) -> None:
        assert to_base_quantity(3.0, "MAT-100045", "BOX") == 750.0

    def test_an_unknown_conversion_raises_rather_than_assuming_one(self) -> None:
        """Silently treating a carton as a piece would give a confident wrong answer."""
        with pytest.raises(UomConversionError, match="no conversion"):
            uom_factor("MAT-100045", "KG")

    def test_an_unknown_material_raises(self) -> None:
        with pytest.raises(UomConversionError, match="unknown material"):
            uom_factor("MAT-NOPE", "PCE")


class TestResidualPercentage:
    def test_ordinary_case(self) -> None:
        assert residual_percentage(100.0, 106.0) == pytest.approx(6.0)

    def test_sign_is_preserved(self) -> None:
        assert residual_percentage(100.0, 94.0) == pytest.approx(-6.0)

    def test_a_zero_expected_price_does_not_divide_by_zero(self) -> None:
        assert residual_percentage(0.0, 0.0) == 0.0
        assert residual_percentage(0.0, 1.0) == 100.0


class TestToleranceRule:
    """A line passes if it is inside *either* limit. Both halves are load-bearing."""

    def test_a_large_percentage_on_a_tiny_line_is_cleared_by_the_absolute_limit(self) -> None:
        """0.40 on a 1.20 unit price is 33% and immaterial. Blocking it is noise."""
        comparison = compare_price(
            invoice_line(PriceElements(list_price=1.60), quantity=10.0),
            po_line(PriceElements(list_price=1.20), quantity=10.0),
            material_id="MAT-300121",
            matched_base_qty=10.0,
            config=CONFIG,
        )
        assert abs(comparison.residual_pct) > CONFIG.price_tolerance_pct
        assert abs(comparison.line_residual_abs) <= CONFIG.price_tolerance_abs
        assert comparison.within_tolerance is True

    def test_a_small_percentage_on_a_large_line_is_caught_by_the_percentage_limit(self) -> None:
        """A 2.5% variance worth thousands is real money and must not pass."""
        comparison = compare_price(
            invoice_line(PriceElements(list_price=410.0), quantity=100.0),
            po_line(PriceElements(list_price=400.0), quantity=100.0),
            material_id="MAT-200310",
            matched_base_qty=100.0,
            config=CONFIG,
        )
        assert comparison.residual_pct == pytest.approx(2.5)
        assert comparison.line_residual_abs == pytest.approx(1000.0)
        assert comparison.within_tolerance is False

    def test_the_tolerance_in_force_is_recorded_on_the_comparison(self) -> None:
        """A reviewer reading an old case must see the limits it was judged against."""
        comparison = compare_price(
            invoice_line(PriceElements(list_price=1.0)),
            po_line(PriceElements(list_price=1.0)),
            material_id="MAT-300121",
            matched_base_qty=1.0,
            config=CONFIG,
        )
        assert comparison.tolerance_pct == CONFIG.price_tolerance_pct
        assert comparison.tolerance_abs == CONFIG.price_tolerance_abs

    def test_a_tightened_tolerance_changes_the_outcome(self) -> None:
        """The rule is configuration, and the test proves configuration reaches it."""
        strict = I2PConfig(price_tolerance_pct=0.1, price_tolerance_abs=0.01)
        line = invoice_line(PriceElements(list_price=1.60), quantity=10.0)
        order = po_line(PriceElements(list_price=1.20), quantity=10.0)
        assert compare_price(line, order, "MAT-300121", 10.0, CONFIG).within_tolerance is True
        assert compare_price(line, order, "MAT-300121", 10.0, strict).within_tolerance is False
