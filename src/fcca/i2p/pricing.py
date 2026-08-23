"""Price and quantity normalisation.

This is the module the whole invoice-to-pay side rests on, and it contains no
model call, no heuristic and no tolerance of its own. It answers one question:
*what is the net price of one base unit?* — for a purchase-order line and for an
invoice line, computed the same way, so that the two numbers can be subtracted.

The naive comparison is to subtract the two printed unit prices. It is wrong
often enough to be worse than useless, because the two sides legitimately
express the same price differently:

* a **price unit** — the purchase order quotes 38.50 per 100 PCE, the invoice
  quotes 0.3850 per PCE;
* a **unit of measure** — the purchase order is in PCE, the invoice is in BOX of
  250;
* **cascading percentage discounts** — 5%, then 3% on the result, then 2% on
  that. These do not sum: 5+3+2 is not 10, it is 9.703;
* an **absolute per-unit surcharge** — a small freight or handling amount added
  after the percentage cascade, which no percentage arithmetic will reproduce.

Each of those alone produces a difference the naive comparison reports as a
price variance. Together they produce a difference large enough that a reviewer
would believe it. The whole point of this module is that the two sides are
reduced to the same unit before anything is compared, so that what remains is a
real disagreement about price.

Order of operations is fixed and it matters:

1. apply the percentage discounts in sequence, each to the running result;
2. divide by the price unit;
3. add the absolute per-unit surcharge;
4. convert from the line's unit of measure to the material's base unit.

Steps 1 and 3 do not commute — a surcharge added before the cascade would be
discounted by it — and step 3 before step 2 would scale the surcharge by the
price unit. Both are tested.
"""

from __future__ import annotations

from fcca.i2p.masterdata import MATERIALS_BY_ID, UOM_FACTORS
from fcca.i2p.models import PriceElements, UnitOfMeasure
from fcca.shared.errors import FCCAError

#: Unit prices are compared at this precision. Six decimals is far below any
#: currency's smallest unit and far above float noise, so it removes spurious
#: residuals without hiding real ones — a 1/100 cent difference on a unit price
#: multiplied by 10,000 units is a euro, and must still be visible.
UNIT_PRICE_PRECISION = 6


class UomConversionError(FCCAError):
    """No conversion factor exists for this material and unit of measure.

    Raised rather than defaulted to 1.0. Silently treating a carton as a piece
    would multiply a price by a hundred and produce a confident, wrong answer —
    exactly the failure mode this module exists to prevent.
    """


def uom_factor(material_id: str, uom: UnitOfMeasure) -> float:
    """Base units per one unit of ``uom`` for this material.

    Returns 1.0 when ``uom`` is already the material's base unit.
    """
    material = MATERIALS_BY_ID.get(material_id)
    if material is None:
        raise UomConversionError(f"unknown material {material_id!r}")
    if uom == material.base_uom:
        return 1.0
    factor = UOM_FACTORS.get((material_id, uom))
    if factor is None:
        raise UomConversionError(
            f"no conversion from {uom} to {material.base_uom} for material {material_id}"
        )
    return factor


def apply_discount_cascade(gross: float, discount_pct: tuple[float, ...]) -> float:
    """Apply percentage discounts in sequence, each to the running result.

    Sequential, not summed. 5% then 3% then 2% leaves 90.297% of the gross
    price, not 90%. The difference is small per unit and is precisely the size
    that a percentage tolerance is set to catch, which is why summing them would
    manufacture exceptions.
    """
    net = gross
    for pct in discount_pct:
        net *= 1.0 - pct / 100.0
    return net


def normalise_unit_price(
    price: PriceElements,
    material_id: str,
    uom: UnitOfMeasure,
) -> float:
    """Net price of one base unit of ``material_id``.

    See the module docstring for why the order of operations is what it is.
    """
    net = apply_discount_cascade(price.list_price, price.discount_pct)
    net /= price.price_unit
    net += price.surcharge_per_unit
    net /= uom_factor(material_id, uom)
    return round(net, UNIT_PRICE_PRECISION)


def naive_unit_price(price: PriceElements) -> float:
    """What a comparison of the printed prices would use.

    Carried through to the reviewer's screen next to the normalised figure. It
    is not used for any decision; it exists so that a person can see what the
    normalisation changed, and so that the test suite can assert the two differ
    on the cases where it matters.
    """
    return round(price.list_price, UNIT_PRICE_PRECISION)


def to_base_quantity(quantity: float, material_id: str, uom: UnitOfMeasure) -> float:
    """Convert a quantity in ``uom`` to the material's base unit."""
    return round(quantity * uom_factor(material_id, uom), UNIT_PRICE_PRECISION)


def residual_percentage(expected: float, actual: float) -> float:
    """Signed difference as a percentage of ``expected``.

    A zero expected price is a master-data failure, not a division by zero, so
    it returns 0.0 for an exact match and 100.0 otherwise rather than raising:
    the caller is a control that must still produce a finding.
    """
    if expected == 0.0:
        return 0.0 if actual == 0.0 else 100.0
    return (actual - expected) / expected * 100.0


__all__ = [
    "UNIT_PRICE_PRECISION",
    "UomConversionError",
    "apply_discount_cascade",
    "naive_unit_price",
    "normalise_unit_price",
    "residual_percentage",
    "to_base_quantity",
    "uom_factor",
]
