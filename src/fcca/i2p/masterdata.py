"""Synthetic master data for the invoice-to-pay module.

Fixed rather than generated, because master data is the thing the process
*resolves against*: if it moved with the seed, a test asserting that material
group 3100 derives GL account 480000 would be asserting nothing. The
transactional data is seeded and varied; the world it refers to is stable.

No real vendor, bank, material or account appears here. The IBANs are
structurally plausible and deliberately invalid as payment instructions.
"""

from __future__ import annotations

from fcca.i2p.models import (
    CostCenter,
    GLAccount,
    Material,
    UnitOfMeasure,
    UomConversion,
    Vendor,
)

COMPANY_CODES: tuple[str, ...] = ("DE10", "NL30", "FR20")

#: Reporting currency for the group. Document currencies convert to it.
REPORTING_CURRENCY = "EUR"

#: Static conversion rates. Fixed, not sampled: an exchange rate that moved with
#: the seed would make every amount in the README irreproducible.
FX_TO_EUR: dict[str, float] = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "CHF": 1.05}


VENDORS: tuple[Vendor, ...] = (
    Vendor(
        vendor_id="V-10010",
        name="Nordwerk Industrieteile GmbH",
        country="DE",
        currency="EUR",
        payment_terms="NT30",
        tax_id="DE811907980",
        bank_iban="DE44500105175407324931",
        bank_name="Rheinische Handelsbank",
    ),
    Vendor(
        vendor_id="V-10011",
        name="Delta Componenten B.V.",
        country="NL",
        currency="EUR",
        payment_terms="NT45",
        tax_id="NL004495445B01",
        bank_iban="NL91ABNA0417164300",
        bank_name="Amstel Zakenbank",
    ),
    Vendor(
        vendor_id="V-10012",
        name="Atelier Mécanique Roussel SAS",
        country="FR",
        currency="EUR",
        payment_terms="NT60",
        tax_id="FR40303265045",
        bank_iban="FR7630006000011234567890189",
        bank_name="Banque de la Sarthe",
    ),
    Vendor(
        vendor_id="V-10013",
        name="Halstead Precision Ltd",
        country="GB",
        currency="GBP",
        payment_terms="NT30",
        tax_id="GB980780684",
        bank_iban="GB29NWBK60161331926819",
        bank_name="Northgate Commercial",
    ),
    Vendor(
        vendor_id="V-10014",
        name="Brunner Werkstoffe AG",
        country="CH",
        currency="CHF",
        payment_terms="NT14",
        tax_id="CHE116281812",
        bank_iban="CH9300762011623852957",
        bank_name="Kantonalbank Aargau",
    ),
    Vendor(
        vendor_id="V-10015",
        name="Meridian Facility Services",
        country="DE",
        currency="EUR",
        payment_terms="NT30",
        tax_id="DE290517341",
        bank_iban="DE02120300000000202051",
        bank_name="Rheinische Handelsbank",
    ),
    Vendor(
        vendor_id="V-10016",
        name="Okabe Sensors Europe N.V.",
        country="NL",
        currency="EUR",
        payment_terms="NT45",
        tax_id="NL810876123B01",
        bank_iban="NL39RABO0300065264",
        bank_name="Amstel Zakenbank",
    ),
    Vendor(
        vendor_id="V-10017",
        name="Castellane Chimie SA",
        country="FR",
        currency="EUR",
        payment_terms="NT60",
        tax_id="FR89552081317",
        bank_iban="FR1420041010050500013M02606",
        bank_name="Banque de la Sarthe",
    ),
)


#: Material group -> GL account. This mapping is what makes a missing GL account
#: a lookup rather than a judgement, and it is why no model is asked to guess one.
GL_ACCOUNTS: tuple[GLAccount, ...] = (
    GLAccount(gl_account="400100", name="Raw materials", material_group="1000"),
    GLAccount(gl_account="400200", name="Components and assemblies", material_group="2000"),
    GLAccount(gl_account="400300", name="Consumables", material_group="3000"),
    GLAccount(gl_account="415000", name="Maintenance materials", material_group="4000"),
    GLAccount(gl_account="420000", name="Laboratory and chemicals", material_group="5000"),
    GLAccount(gl_account="470000", name="Facility services", material_group="9000"),
    GLAccount(gl_account="480000", name="Professional services", material_group=None),
    GLAccount(gl_account="199000", name="Goods received / invoice received", material_group=None),
)


COST_CENTERS: tuple[CostCenter, ...] = (
    CostCenter(
        cost_center="CC-1000",
        name="Production Line 1",
        company_code="DE10",
        aliases=("Production Line 1", "Line 1", "Prod L1"),
    ),
    CostCenter(
        cost_center="CC-1100",
        name="Production Line 2",
        company_code="DE10",
        aliases=("Production Line 2", "Line 2", "Prod L2"),
    ),
    CostCenter(
        cost_center="CC-4200",
        name="Plant 2 Maintenance",
        company_code="DE10",
        aliases=("Plant 2 Maintenance", "Maintenance Plant 2", "Werk 2 Instandhaltung"),
    ),
    CostCenter(
        cost_center="CC-5300",
        name="Quality Laboratory",
        company_code="NL30",
        aliases=("Quality Laboratory", "QA Lab", "Lab NL"),
    ),
    CostCenter(
        cost_center="CC-6100",
        name="Warehouse Operations",
        company_code="NL30",
        aliases=("Warehouse Operations", "Warehouse Ops", "WH Ops"),
    ),
    CostCenter(
        cost_center="CC-7400",
        name="Site Facilities Toulouse",
        company_code="FR20",
        aliases=("Site Facilities Toulouse", "Facilities Toulouse", "Toulouse Facilities"),
    ),
    CostCenter(
        cost_center="CC-8200",
        name="Research and Development",
        company_code="FR20",
        aliases=("Research and Development", "R&D", "RnD"),
    ),
)


MATERIALS: tuple[Material, ...] = (
    Material(
        material_id="MAT-100045",
        description="Hex bolt M8x40, stainless",
        material_group="1000",
        base_uom="PCE",
        standard_price=38.50,
        price_unit=100,
    ),
    Material(
        material_id="MAT-100046",
        description="Sealing ring 22mm NBR",
        material_group="1000",
        base_uom="PCE",
        standard_price=12.80,
        price_unit=100,
    ),
    Material(
        material_id="MAT-200310",
        description="Servo controller board rev C",
        material_group="2000",
        base_uom="PCE",
        standard_price=412.00,
        price_unit=1,
    ),
    Material(
        material_id="MAT-200311",
        description="Linear actuator 300mm",
        material_group="2000",
        base_uom="PCE",
        standard_price=268.75,
        price_unit=1,
    ),
    Material(
        material_id="MAT-300120",
        description="Cutting fluid, semi-synthetic",
        material_group="3000",
        base_uom="L",
        standard_price=6.45,
        price_unit=1,
    ),
    Material(
        material_id="MAT-300121",
        description="Abrasive disc 125mm",
        material_group="3000",
        base_uom="PCE",
        standard_price=2.15,
        price_unit=1,
    ),
    Material(
        material_id="MAT-400505",
        description="Drive belt, toothed 1400mm",
        material_group="4000",
        base_uom="PCE",
        standard_price=94.20,
        price_unit=1,
    ),
    Material(
        material_id="MAT-500777",
        description="Buffer solution pH 7.00",
        material_group="5000",
        base_uom="L",
        standard_price=18.90,
        price_unit=1,
    ),
    Material(
        material_id="MAT-500778",
        description="Nitrile gloves, size L",
        material_group="5000",
        base_uom="BOX",
        standard_price=41.00,
        price_unit=1,
    ),
    Material(
        material_id="MAT-900001",
        description="Cleaning service, per hour",
        material_group="9000",
        base_uom="HR",
        standard_price=32.00,
        price_unit=1,
    ),
)


#: Alternative units of measure. The conversion factor is the reason a price
#: comparison has to normalise before it subtracts: a purchase order priced per
#: PCE and an invoice priced per BOX are not comparable numbers.
UOM_CONVERSIONS: tuple[UomConversion, ...] = (
    UomConversion(material_id="MAT-100045", alternative_uom="BOX", factor=250.0),
    UomConversion(material_id="MAT-100046", alternative_uom="BOX", factor=500.0),
    UomConversion(material_id="MAT-300121", alternative_uom="BOX", factor=25.0),
    UomConversion(material_id="MAT-300121", alternative_uom="CAR", factor=100.0),
    UomConversion(material_id="MAT-300120", alternative_uom="CAR", factor=20.0),
    UomConversion(material_id="MAT-500778", alternative_uom="CAR", factor=10.0),
    UomConversion(material_id="MAT-200310", alternative_uom="BOX", factor=4.0),
)


#: Our material id -> the vendor's own item number. Vendors invoice their code,
#: not ours, so resolving it is the first thing the engine has to do.
SUPPLIER_ITEM_NUMBERS: dict[str, str] = {
    "MAT-100045": "NW-8840-SS",
    "MAT-100046": "NW-RING-22N",
    "MAT-200310": "DC-SRVC-REVC",
    "MAT-200311": "DC-LA300",
    "MAT-300120": "AR-CF-SEMI",
    "MAT-300121": "AR-AD125",
    "MAT-400505": "HP-BELT-1400T",
    "MAT-500777": "OK-BUF-700",
    "MAT-500778": "OK-GLV-L",
    "MAT-900001": "MF-CLEAN-HR",
}

#: Tax codes and their rates. Mixed rates across a single invoice are ordinary,
#: so the engine may not assume one rate per document.
TAX_CODES: dict[str, float] = {"V1": 19.0, "V2": 7.0, "V0": 0.0}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

VENDORS_BY_ID: dict[str, Vendor] = {v.vendor_id: v for v in VENDORS}
MATERIALS_BY_ID: dict[str, Material] = {m.material_id: m for m in MATERIALS}
COST_CENTERS_BY_ID: dict[str, CostCenter] = {c.cost_center: c for c in COST_CENTERS}
GL_BY_MATERIAL_GROUP: dict[str, str] = {
    g.material_group: g.gl_account for g in GL_ACCOUNTS if g.material_group
}
MATERIAL_BY_SUPPLIER_ITEM: dict[str, str] = {v: k for k, v in SUPPLIER_ITEM_NUMBERS.items()}
UOM_FACTORS: dict[tuple[str, UnitOfMeasure], float] = {
    (c.material_id, c.alternative_uom): c.factor for c in UOM_CONVERSIONS
}


def cost_centers_for(company_code: str) -> tuple[CostCenter, ...]:
    return tuple(c for c in COST_CENTERS if c.company_code == company_code)


__all__ = [
    "COMPANY_CODES",
    "COST_CENTERS",
    "COST_CENTERS_BY_ID",
    "FX_TO_EUR",
    "GL_ACCOUNTS",
    "GL_BY_MATERIAL_GROUP",
    "MATERIALS",
    "MATERIALS_BY_ID",
    "MATERIAL_BY_SUPPLIER_ITEM",
    "REPORTING_CURRENCY",
    "SUPPLIER_ITEM_NUMBERS",
    "TAX_CODES",
    "UOM_CONVERSIONS",
    "UOM_FACTORS",
    "VENDORS",
    "VENDORS_BY_ID",
    "cost_centers_for",
]
