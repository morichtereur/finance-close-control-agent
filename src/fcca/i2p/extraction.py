"""Document extraction: the port, the contract, and the two adapters.

This module is the seam where a real OCR or document-understanding engine would
attach. **No engine is bundled**, and that is deliberate rather than unfinished.

Why a port and not an implementation
------------------------------------
Bundling pytesseract would add a system binary, a model download and a
platform-specific install to a repository whose whole claim is that it runs
anywhere with ``pip install -e .``. Bundling Azure Document Intelligence or
Klippa would add an account, a key and a per-page cost. Neither would make the
downstream arithmetic any more trustworthy, because the thing that matters to
every number in this system is not *how* a field was read — it is *how confident
the reader was, and what the pipeline does about it*. That is what this module
models, and it is testable without a single pixel.

The contract an adapter must satisfy
------------------------------------
An adapter converts whatever its engine returns into a tuple of
:class:`ExtractedField`. It must satisfy four things:

1. **``field_path`` matches the invoice schema.** ``stated_total_gross``,
   ``lines[0].quantity``. A path the schema does not have is a bug in the
   adapter, not a new field — :meth:`ExtractionPayload.unknown_paths` finds them.
2. **``confidence`` is the engine's own score**, in ``[0, 1]``, and is *never*
   produced by a language model. See the warning below.
3. **``bbox`` locates the value on the page** so a reviewer can check the
   original. An adapter that cannot supply one supplies ``None`` rather than a
   guess.
4. **Values are typed as the schema expects.** Parsing "1.234,56" into a float
   is the adapter's job; the pipeline must never receive a string where it
   expects a number.

A worked adapter skeleton is in ``docs/extraction-adapter.md``, and
``data/fixtures/extraction/`` holds payloads in exactly this shape, including
deliberately degraded ones.

.. warning::

   **Confidence must come from the extraction engine, never from a model.**

   The confidence attached to a field flows into
   :func:`fcca.i2p.checks.extraction_confidence_gate`, which can force an
   invoice to ``escalate``. That makes it load-bearing. An OCR engine's
   confidence is a measurement — a property of pixel evidence, calibrated
   against a character set. A language model's stated confidence is a token it
   generated, and asking one "how sure are you this says 1,234.56?" produces a
   number that looks identical and means nothing.

   If those were interchangeable, a model would be deciding a tolerance
   outcome through the side door: quietly, one layer removed, and in exactly the
   place the architecture claims it never happens. :class:`ExtractedField`
   therefore records which engine produced each score, and
   :meth:`ExtractionPayload.assert_not_model_scored` refuses a payload whose
   engine name looks like a language model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fcca.i2p.models import (
    BoundingBox,
    FieldProvenance,
    FieldSource,
    Invoice,
)
from fcca.shared.errors import DataNotFoundError, FCCAError


class ExtractionError(FCCAError):
    """An extraction payload could not be read, or is not a valid one."""


#: Engine names that must never appear on an extraction payload. Matched
#: case-insensitively as substrings, because the failure this guards against is
#: someone wiring an LLM in as a "reader" and the confidence scores silently
#: becoming generated tokens.
MODEL_ENGINE_MARKERS: tuple[str, ...] = (
    "gpt",
    "claude",
    "llm",
    "sonnet",
    "haiku",
    "opus",
    "gemini",
    "mistral",
    "llama",
    "bedrock",
    "vertex",
)


class ExtractedField(BaseModel):
    """One field an extraction engine read off a document."""

    model_config = ConfigDict(frozen=True)

    field_path: str = Field(
        description="Dotted path into the invoice schema, e.g. 'lines[0].quantity'."
    )
    value: Any = Field(description="Parsed to the type the schema expects, not a raw string.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The extraction engine's own score. Never a model-produced number.",
    )
    bbox: BoundingBox | None = Field(
        default=None, description="None where the engine cannot localise the value."
    )
    raw_text: str | None = Field(
        default=None, description="What was on the page before parsing, for a reviewer to check."
    )


class ExtractionPayload(BaseModel):
    """Everything one engine read off one document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    invoice_id: str = Field(description="The invoice this document is claimed to be.")
    engine: str = Field(description="Extraction engine name, e.g. 'tesseract' or 'azure-di'.")
    engine_version: str
    fields: tuple[ExtractedField, ...]

    @model_validator(mode="after")
    def _engine_is_not_a_model(self) -> Self:
        self.assert_not_model_scored()
        return self

    def assert_not_model_scored(self) -> None:
        """Refuse a payload whose confidences came from a language model.

        See the module warning. This is a cheap, blunt check on the engine name
        rather than anything clever, and it is here because the failure it
        guards against is silent: the numbers look the same either way.
        """
        lowered = self.engine.lower()
        for marker in MODEL_ENGINE_MARKERS:
            if marker in lowered:
                raise ValueError(
                    f"extraction engine {self.engine!r} looks like a language model. "
                    "Extraction confidence feeds a tolerance decision and must be a "
                    "measurement from an extraction engine, not a generated token. "
                    "See fcca.i2p.extraction for why."
                )

    @property
    def by_path(self) -> dict[str, ExtractedField]:
        return {field.field_path: field for field in self.fields}

    def unknown_paths(self) -> tuple[str, ...]:
        """Field paths this payload claims that the invoice schema does not have."""
        return tuple(sorted(p for p in self.by_path if not _path_exists(p)))

    def lowest(self, paths: Iterable[str]) -> tuple[str, float] | None:
        """The weakest of the named fields, as ``(path, confidence)``.

        Returns ``None`` when none of the named fields were extracted at all —
        which is a different condition from "extracted badly" and is handled
        separately by the gate.
        """
        scored = [(p, self.by_path[p].confidence) for p in paths if p in self.by_path]
        return min(scored, key=lambda pair: pair[1]) if scored else None


# ---------------------------------------------------------------------------
# The port
# ---------------------------------------------------------------------------


@runtime_checkable
class InvoiceSource(Protocol):
    """Where invoices come from.

    The pipeline depends on this and not on the generator, so a document-backed
    source is a substitution rather than a rewrite. Both adapters below return
    the *same* :class:`~fcca.i2p.models.Invoice` type — an extraction-backed
    invoice is not a different kind of object, it is the same object with
    different provenance attached.
    """

    def invoice_ids(self) -> tuple[str, ...]:
        """Every invoice this source can supply, in a stable order."""
        ...

    def invoice(self, invoice_id: str) -> Invoice:
        """One invoice, with provenance populated for every field it knows."""
        ...

    def provenance(self, invoice_id: str) -> dict[str, FieldProvenance]:
        """Field-level provenance for one invoice, keyed by field path."""
        ...


class SyntheticSource:
    """The generated dataset, as it has always been.

    Every field is marked ``synthetic``: there was no document, so there is no
    extraction confidence and nothing for the gate to act on. This is the source
    the shipped metrics are measured with, and it is what makes the "no OCR"
    claim in the README literally true rather than a caveat.
    """

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def invoice_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.repository.invoices))

    def invoice(self, invoice_id: str) -> Invoice:
        return self.repository.invoice(invoice_id)

    def provenance(self, invoice_id: str) -> dict[str, FieldProvenance]:
        invoice = self.repository.invoice(invoice_id)
        return synthetic_provenance(invoice)


class DocumentSource:
    """Invoices assembled from extraction payloads.

    The payloads are the engine's output; this adapter's only job is to overlay
    them onto a base invoice and record what came from where. It does **not**
    decide whether a confidence is good enough — that is
    :func:`fcca.i2p.checks.extraction_confidence_gate`, which runs as a named
    pipeline step and writes a trace record. Keeping the two apart is what makes
    the threshold reviewable: reading a field and judging the reading are
    different acts and they fail for different reasons.

    A field the engine did not read keeps its base value and is marked
    ``synthetic``, so a dropout is visible as an absence rather than silently
    becoming a confident zero.
    """

    def __init__(
        self,
        payloads: dict[str, ExtractionPayload],
        base: InvoiceSource,
    ) -> None:
        self.payloads = payloads
        self.base = base

    @classmethod
    def from_directory(cls, directory: Path, base: InvoiceSource) -> DocumentSource:
        """Load every ``*.json`` payload in a fixture directory."""
        if not directory.exists():
            raise DataNotFoundError(f"no extraction fixtures at {directory}")
        payloads: dict[str, ExtractionPayload] = {}
        for path in sorted(directory.glob("*.json")):
            payload = ExtractionPayload.model_validate_json(path.read_text(encoding="utf-8"))
            payloads[payload.invoice_id] = payload
        if not payloads:
            raise DataNotFoundError(f"no extraction payloads in {directory}")
        return cls(payloads, base)

    def invoice_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.payloads))

    def invoice(self, invoice_id: str) -> Invoice:
        base = self.base.invoice(invoice_id)
        payload = self.payloads.get(invoice_id)
        if payload is None:
            return base
        unknown = payload.unknown_paths()
        if unknown:
            raise ExtractionError(
                f"extraction payload for {invoice_id} names {len(unknown)} field(s) the "
                f"invoice schema does not have: {', '.join(unknown)}"
            )
        return _apply(base, payload)

    def provenance(self, invoice_id: str) -> dict[str, FieldProvenance]:
        provenance = synthetic_provenance(self.base.invoice(invoice_id))
        payload = self.payloads.get(invoice_id)
        if payload is None:
            return provenance
        for field in payload.fields:
            provenance[field.field_path] = FieldProvenance(
                source="extracted",
                confidence=field.confidence,
                bbox=field.bbox,
                engine=f"{payload.engine}@{payload.engine_version}",
                raw_text=field.raw_text,
            )
        return provenance


# ---------------------------------------------------------------------------
# Field paths
# ---------------------------------------------------------------------------

_INDEXED = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[(?P<index>\d+)\]$")

#: Header fields an extraction engine plausibly reads off a document. Used to
#: build synthetic provenance and to validate payload paths.
HEADER_PATHS: tuple[str, ...] = (
    "vendor_id",
    "company_code",
    "invoice_date",
    "received_date",
    "vendor_reference",
    "currency",
    "stated_bank_iban",
    "free_text",
    "stated_total_net",
    "stated_total_tax",
    "stated_total_gross",
)

#: Per-line fields, formatted with the line index.
LINE_PATHS: tuple[str, ...] = (
    "description",
    "supplier_item_no",
    "quantity",
    "uom",
    "tax_rate",
    "po_id",
    "po_line",
    "gl_account",
    "cost_center",
    "price.list_price",
)


def line_path(index: int, field: str) -> str:
    """``lines[2].quantity`` — the one place this format is constructed."""
    return f"lines[{index}].{field}"


def _path_exists(path: str) -> bool:
    """Whether a dotted path addresses a real field on the invoice schema."""
    if path in HEADER_PATHS:
        return True
    head, _, rest = path.partition(".")
    match = _INDEXED.match(head)
    if not match or match.group("name") != "lines":
        return False
    return rest in LINE_PATHS


def synthetic_provenance(invoice: Invoice) -> dict[str, FieldProvenance]:
    """Mark every field of an invoice as generated.

    The baseline every other source starts from: a field nobody extracted and no
    rule derived is one the generator made up, and saying so explicitly is more
    honest than leaving it unlabelled.
    """
    provenance = {path: FieldProvenance(source="synthetic") for path in HEADER_PATHS}
    for index in range(len(invoice.lines)):
        for field in LINE_PATHS:
            provenance[line_path(index, field)] = FieldProvenance(source="synthetic")
    return provenance


def _apply(base: Invoice, payload: ExtractionPayload) -> Invoice:
    """Overlay extracted values onto a base invoice."""
    data = base.model_dump()
    lines = [dict(line) for line in data["lines"]]

    for field in payload.fields:
        head, _, rest = field.field_path.partition(".")
        match = _INDEXED.match(head)
        if match is None:
            data[field.field_path] = field.value
            continue
        index = int(match.group("index"))
        if index >= len(lines):
            raise ExtractionError(
                f"payload for {payload.invoice_id} sets {field.field_path} but the "
                f"invoice has {len(lines)} line(s)"
            )
        if "." in rest:
            outer, _, inner = rest.partition(".")
            nested = dict(lines[index][outer])
            nested[inner] = field.value
            lines[index][outer] = nested
        else:
            lines[index][rest] = field.value

    data["lines"] = lines
    return Invoice.model_validate(data)


def read_payloads(directory: Path) -> Iterator[ExtractionPayload]:
    """Every payload in a directory, for tooling that does not want a source."""
    for path in sorted(directory.glob("*.json")):
        yield ExtractionPayload.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "HEADER_PATHS",
    "LINE_PATHS",
    "MODEL_ENGINE_MARKERS",
    "BoundingBox",
    "DocumentSource",
    "ExtractedField",
    "ExtractionError",
    "ExtractionPayload",
    "FieldProvenance",
    "FieldSource",
    "InvoiceSource",
    "SyntheticSource",
    "line_path",
    "read_payloads",
    "synthetic_provenance",
]
