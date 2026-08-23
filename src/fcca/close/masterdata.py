"""Synthetic master data.

Entirely fictional. No real entity, account, cost centre, employee or amount from
any organisation appears in this repository.
"""

from __future__ import annotations

from typing import NamedTuple


class Entity(NamedTuple):
    company_code: str
    name: str
    country: str
    currency: str


class Account(NamedTuple):
    number: str
    name: str
    kind: str  # "bs" (balance sheet) or "pl" (profit and loss)


ENTITIES: tuple[Entity, ...] = (
    Entity("DE10", "Nordwind Manufacturing GmbH", "DE", "EUR"),
    Entity("CH20", "Nordwind Services AG", "CH", "CHF"),
    Entity("NL30", "Nordwind Logistics B.V.", "NL", "EUR"),
    Entity("US40", "Nordwind Inc.", "US", "USD"),
    Entity("PL50", "Nordwind Shared Services Sp. z o.o.", "PL", "PLN"),
)

ACCOUNTS: tuple[Account, ...] = (
    Account("100000", "Cash and cash equivalents", "bs"),
    Account("113000", "Trade receivables", "bs"),
    Account("141000", "Inventory", "bs"),
    Account("160000", "Prepaid expenses", "bs"),
    Account("199000", "Suspense and clearing", "bs"),
    Account("210000", "Trade payables", "bs"),
    Account("231000", "Accrued liabilities", "bs"),
    Account("289000", "Restructuring provisions", "bs"),
    Account("400000", "Revenue - products", "pl"),
    Account("410000", "Revenue - services", "pl"),
    Account("480000", "Other operating income", "pl"),
    Account("500000", "Cost of materials", "pl"),
    Account("510000", "Cost of sales adjustments", "pl"),
    Account("600000", "Personnel expenses", "pl"),
    Account("610000", "Consulting and professional fees", "pl"),
    Account("620000", "IT and software expenses", "pl"),
    Account("640000", "Travel and entertainment", "pl"),
    Account("700000", "Depreciation and amortisation", "pl"),
    Account("810000", "Interest expense", "pl"),
)

ACCOUNT_NAMES: dict[str, str] = {a.number: a.name for a in ACCOUNTS}

COST_CENTERS: tuple[tuple[str, str], ...] = (
    ("CC-1000", "Finance"),
    ("CC-2000", "Sales"),
    ("CC-3000", "Operations"),
    ("CC-4000", "Information Technology"),
    ("CC-5000", "Human Resources"),
    ("CC-6000", "Research and Development"),
    ("CC-7000", "Legal and Compliance"),
)

#: user id -> (role, home entity)
USERS: dict[str, tuple[str, str]] = {
    "u.becker": ("accountant", "DE10"),
    "u.hoffmann": ("accountant", "DE10"),
    "u.klein": ("financial_controller", "DE10"),
    "u.meier": ("accountant", "CH20"),
    "u.zimmer": ("financial_controller", "CH20"),
    "u.devries": ("accountant", "NL30"),
    "u.jansen": ("financial_controller", "NL30"),
    "u.carter": ("accountant", "US40"),
    "u.novak": ("accountant", "PL50"),
    "u.kowalski": ("accountant", "PL50"),
    "u.lewandow": ("financial_controller", "PL50"),
    "u.osei": ("shared_services", "PL50"),
    "u.tanaka": ("shared_services", "PL50"),
    "svc.interface": ("system", "DE10"),
}

CONTROLLERS: dict[str, str] = {
    "DE10": "u.klein",
    "CH20": "u.zimmer",
    "NL30": "u.jansen",
    "US40": "u.klein",
    "PL50": "u.lewandow",
}

#: document type -> (description, manual by default)
DOCUMENT_TYPES: dict[str, tuple[str, bool]] = {
    "SA": ("GL manual entry", True),
    "SB": ("GL adjustment", True),
    "KR": ("Vendor invoice", False),
    "DR": ("Customer invoice", False),
    "RV": ("Billing document", False),
    "AF": ("Depreciation run", False),
    "ZP": ("Payment run", False),
}

#: Fixed rates to the group reporting currency (EUR). Static by design: FX is not
#: what this prototype is about, and a fixed table keeps runs reproducible.
FX_TO_EUR: dict[str, float] = {
    "EUR": 1.0,
    "CHF": 1.05,
    "USD": 0.92,
    "PLN": 0.23,
    "GBP": 1.17,
}

BUSINESS_NARRATIVES: tuple[str, ...] = (
    "Accrual for services received not invoiced",
    "Reclassification of prepaid maintenance",
    "Monthly depreciation posting",
    "Vendor invoice for logistics services",
    "Intercompany recharge of shared services",
    "Provision for outstanding customer credits",
    "Correction of misposted cost center",
    "Accrual for external audit fees",
    "Release of unused restructuring provision",
    "Recognition of deferred revenue tranche",
    "Payroll accrual for variable compensation",
    "Cut-off adjustment for goods in transit",
    "Settlement of intercompany balance",
    "Capitalisation of software development cost",
    "Recharge of travel expenses to project",
)

WEAK_NARRATIVES: tuple[str, ...] = (
    "Adjustment",
    "Correction per instruction",
    "Manual posting July",
    "Ref 4471",
    "As discussed",
    "Reclass",
)
