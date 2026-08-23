"""Synthesising extraction payloads, including bad ones.

The gate in :func:`fcca.i2p.checks.extraction_confidence_gate` is only worth
having if something can be shown to get past the rest of the pipeline without
it. This module manufactures that something: extraction payloads over the
existing invoices, optionally degraded in the two ways a character recogniser
actually fails.

**Dropout.** The engine does not find a field at all. Realistic — a stamp across
the total, a fold through the IBAN — and the safe failure mode, because an
absent field is visibly absent. :class:`~fcca.i2p.extraction.DocumentSource`
leaves it at its base value marked ``synthetic``, so nothing silently becomes a
confident zero.

**Digit confusion.** The engine finds the field and reads it wrong. This is the
dangerous one, and it is not random noise: OCR errors cluster on glyph pairs
that look alike at low resolution — 0/8, 1/7, 5/6 — and on the decimal
separator, which is worse than all of them because a comma read as a thousands
separator moves an amount by two orders of magnitude. Those are the four
failures modelled here, and they are chosen because they change what gets paid
rather than merely what gets printed.

.. note::

   **The confidence-to-correctness relationship here is a modelling assumption,
   not a measurement.** A misread field is given a low confidence, because
   recognisers do generally score ambiguous glyphs lower — and that assumption
   is what makes the gate catch these. Real engines also produce confidently
   wrong readings, and this module deliberately produces a small share of those
   too (see ``confident_error_share``), because a demonstration in which the
   gate catches everything would be a demonstration of nothing. What the shipped
   metrics show is how the pipeline behaves under *this* model of extraction
   error. They are not evidence about any particular engine, and the README says
   so.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

from fcca.i2p.extraction import (
    ExtractedField,
    ExtractionPayload,
    line_path,
)
from fcca.i2p.models import BoundingBox, Invoice

#: Glyph pairs a character recogniser confuses at low resolution. Bidirectional.
DIGIT_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("0", "8"),
    ("1", "7"),
    ("5", "6"),
    ("3", "9"),
)

#: Fields worth extracting at all. Header fields plus the per-line values the
#: three-way match actually consumes.
EXTRACTABLE_HEADER: tuple[str, ...] = (
    "vendor_id",
    "company_code",
    "invoice_date",
    "vendor_reference",
    "currency",
    "stated_bank_iban",
    "stated_total_net",
    "stated_total_tax",
    "stated_total_gross",
)

EXTRACTABLE_LINE: tuple[str, ...] = (
    "quantity",
    "po_id",
    "price.list_price",
    "tax_rate",
)

#: Share of misreads the engine is nonetheless confident about. Small, but not
#: zero: a gate that catches every error by construction proves nothing.
CONFIDENT_ERROR_SHARE = 0.15


def _confuse_digits(text: str, rng: random.Random) -> str:
    """Swap one confusable glyph for its partner."""
    positions = [i for i, ch in enumerate(text) if any(ch in pair for pair in DIGIT_CONFUSIONS)]
    if not positions:
        return text
    index = rng.choice(positions)
    char = text[index]
    for a, b in DIGIT_CONFUSIONS:
        if char == a:
            return text[:index] + b + text[index + 1 :]
        if char == b:
            return text[:index] + a + text[index + 1 :]
    return text


def _shift_decimal(value: float, rng: random.Random) -> float:
    """Read the decimal separator as a thousands separator, or lose it entirely.

    The single most expensive OCR failure on an invoice: 1.234,56 read as
    123456. Modelled in both directions because both happen.
    """
    return value * rng.choice((100.0, 0.01, 1000.0))


#: Returned when a misread cannot be expressed as a valid value for its field.
#: The adapter contract requires schema-valid types, so an engine that reads
#: "2026-07-82" has not read a date — it has failed to read the field. Modelling
#: that as a dropout rather than quietly keeping the correct value is the honest
#: version: the pipeline sees an absence, which is exactly what it would see in
#: production when a parser rejects the string.
UNREADABLE = object()


def _corrupt(value: Any, rng: random.Random) -> Any:
    """Misread one value in an OCR-plausible way.

    Returns :data:`UNREADABLE` where the corruption cannot be a valid value for
    the field — a day of 82, a month of 19. Real adapters hit this constantly and
    the contract says what to do about it: emit nothing rather than a guess.
    """
    if isinstance(value, str) and _looks_like_iso_date(value):
        corrupted = _confuse_digits(value, rng)
        try:
            date.fromisoformat(corrupted)
        except ValueError:
            return UNREADABLE
        return corrupted
    if isinstance(value, float):
        if rng.random() < 0.35:
            return round(_shift_decimal(value, rng), 2)
        corrupted = _confuse_digits(f"{value:.2f}", rng)
        try:
            return float(corrupted)
        except ValueError:
            return value
    if isinstance(value, int):
        try:
            return int(_confuse_digits(str(value), rng))
        except ValueError:
            return value
    if isinstance(value, str):
        return _confuse_digits(value, rng)
    return value


def _looks_like_iso_date(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _bbox(rng: random.Random) -> BoundingBox:
    """A plausible box. Values are not meaningful — only that one exists."""
    x0 = round(rng.uniform(0.05, 0.6), 3)
    y0 = round(rng.uniform(0.05, 0.9), 3)
    return BoundingBox(
        page=1,
        x0=x0,
        y0=y0,
        x1=round(min(x0 + rng.uniform(0.08, 0.3), 1.0), 3),
        y1=round(min(y0 + rng.uniform(0.01, 0.03), 1.0), 3),
    )


def _read(value: Any, path: str) -> Any:
    """Value at a dotted path on an already-dumped invoice line or header."""
    if "." in path:
        outer, _, inner = path.partition(".")
        return value[outer][inner]
    return value[path]


def build_payload(
    invoice: Invoice,
    *,
    rng: random.Random,
    dropout_rate: float = 0.0,
    digit_confusion_rate: float = 0.0,
    engine: str = "tesseract",
    engine_version: str = "5.3.4",
) -> ExtractionPayload:
    """Manufacture one engine's reading of one invoice.

    With both rates at zero this is a perfect read: every field present, every
    confidence high, and the resulting invoice identical to the synthetic one.
    That is the default, and it is what keeps the shipped metrics measured on
    clean data.
    """
    data = invoice.model_dump()
    fields: list[ExtractedField] = []

    def emit(path: str, value: Any) -> None:
        if rng.random() < dropout_rate:
            return  # not read at all; stays synthetic and visibly absent
        misread = rng.random() < digit_confusion_rate
        if misread:
            value = _corrupt(value, rng)
            if value is UNREADABLE:
                return  # the engine could not produce a valid value; nothing is emitted
            # Usually the engine knows it struggled — but not always.
            confident = rng.random() < CONFIDENT_ERROR_SHARE
            confidence = round(rng.uniform(0.88, 0.97) if confident else rng.uniform(0.30, 0.74), 3)
        else:
            confidence = round(rng.uniform(0.90, 0.995), 3)
        fields.append(
            ExtractedField(
                field_path=path,
                value=value,
                confidence=confidence,
                bbox=_bbox(rng),
                raw_text=str(value),
            )
        )

    for path in EXTRACTABLE_HEADER:
        value = data[path]
        # Dates round-trip as strings; the schema parses them back.
        emit(path, value.isoformat() if hasattr(value, "isoformat") else value)

    for index, line in enumerate(data["lines"]):
        for field in EXTRACTABLE_LINE:
            value = _read(line, field)
            if value is None:
                continue  # an FI line has no purchase order to read
            emit(line_path(index, field), value)

    return ExtractionPayload(
        document_id=f"DOC-{invoice.invoice_id}",
        invoice_id=invoice.invoice_id,
        engine=engine,
        engine_version=engine_version,
        fields=tuple(fields),
    )


def build_payloads(
    invoices: list[Invoice],
    *,
    seed: int,
    dropout_rate: float = 0.0,
    digit_confusion_rate: float = 0.0,
) -> dict[str, ExtractionPayload]:
    """Payloads for a whole population, reproducibly.

    One generator seeded once and shared, so the same seed and rates always
    produce the same degradation — a demonstration nobody can reproduce is an
    anecdote.
    """
    rng = random.Random(seed)
    return {
        invoice.invoice_id: build_payload(
            invoice,
            rng=rng,
            dropout_rate=dropout_rate,
            digit_confusion_rate=digit_confusion_rate,
        )
        for invoice in invoices
    }


__all__ = [
    "CONFIDENT_ERROR_SHARE",
    "DIGIT_CONFUSIONS",
    "EXTRACTABLE_HEADER",
    "EXTRACTABLE_LINE",
    "UNREADABLE",
    "build_payload",
    "build_payloads",
]
