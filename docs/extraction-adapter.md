# Writing an extraction adapter

This repository starts from structured JSON. It does not read PDFs, and it does
not ship an OCR engine. `src/fcca/i2p/extraction.py` is the seam where one would
attach, and this document is what someone writing that adapter needs.

## Why there is no engine here

Bundling `pytesseract` adds a system binary and a language-data download to a
repository whose claim is that it runs anywhere with `pip install -e .`.
Bundling Azure Document Intelligence or Klippa adds an account, a key and a
per-page cost. Neither makes any number downstream more trustworthy, because the
thing the pipeline actually depends on is not *how* a field was read — it is
**how confident the reader was, and what the pipeline does about it**. That part
is implemented, gated and tested here, and it is testable without a single pixel.

## The contract

An adapter turns whatever its engine returns into a tuple of `ExtractedField`
and wraps it in an `ExtractionPayload`. Four requirements:

**1. `field_path` addresses a real field.** Header fields by name
(`stated_total_gross`), line fields indexed (`lines[0].quantity`, and
`lines[2].price.list_price` for nested ones). `ExtractionPayload.unknown_paths()`
returns anything the schema does not have; `DocumentSource` raises on it rather
than silently ignoring it, because a path that does not resolve is an adapter
bug and the quiet version of that bug is a field that never gets gated.

The valid paths are `HEADER_PATHS` and `LINE_PATHS` in `extraction.py`.

**2. `confidence` is the engine's own score, in `[0, 1]`.** Not a model's.
See below — this is the one rule that is enforced in code.

**3. `bbox` locates the value**, in normalised page coordinates so a box means
the same thing at any scan resolution. An engine that cannot localise a value
supplies `None` rather than a guess.

**4. Values are parsed to the type the schema expects.** `"1.234,56"` becomes
`1234.56`; `"2026-07-15"` stays an ISO string the schema can parse. **A value
that cannot be expressed validly is not a value — emit nothing.** An engine that
reads `2026-07-82` has not read a date, and the honest representation is an
absent field, which the pipeline sees as a dropout. Keeping the correct value
because the misread would not parse makes the whole exercise circular.

## The rule that is enforced in code

> **Extraction confidence must never come from a language model.**

The confidence on a field flows into `extraction_confidence_gate`, which can
force an invoice to `escalate`. That makes it load-bearing. An OCR engine's
confidence is a measurement — a property of pixel evidence, calibrated against a
character set. A language model's stated confidence is a token it generated;
asking one "how sure are you this says 1,234.56?" produces a number that looks
identical and means nothing.

If those were interchangeable, a model would be deciding a tolerance outcome
through the side door: quietly, one layer removed, and in exactly the place the
architecture claims it never happens. `ExtractionPayload` therefore rejects a
payload whose engine name looks like a model, and there is a test for it.

A vision model *can* legitimately do the reading. What it cannot do is score its
own reading and have that score gate arithmetic. If you use one, take the
confidence from something else — a second engine agreeing, a checksum, a
field-level validation — and name the engine for what produced the score.

## Skeleton

```python
from fcca.i2p.extraction import BoundingBox, ExtractedField, ExtractionPayload


def to_payload(invoice_id: str, engine_result) -> ExtractionPayload:
    fields = []
    for item in engine_result.key_value_pairs:
        path = MAP.get(item.label)  # your engine's label -> schema path
        if path is None:
            continue  # a field we do not model; not an error
        value = parse(path, item.value)  # to the schema's type
        if value is UNPARSEABLE:
            continue  # emit nothing; see requirement 4
        fields.append(
            ExtractedField(
                field_path=path,
                value=value,
                confidence=item.confidence,  # the engine's, not a model's
                bbox=BoundingBox(
                    page=item.page,
                    x0=item.polygon.left,
                    y0=item.polygon.top,
                    x1=item.polygon.right,
                    y1=item.polygon.bottom,
                ),
                raw_text=item.content,
            )
        )
    return ExtractionPayload(
        document_id=engine_result.document_id,
        invoice_id=invoice_id,
        engine="azure-di",
        engine_version="2024-11-30",
        fields=tuple(fields),
    )
```

Then:

```python
source = DocumentSource({invoice_id: payload}, SyntheticSource(repository))
engine = InvoiceEngine(repository, settings, source=source)
```

Nothing else changes. The pipeline depends on the `InvoiceSource` port, so a
document-backed source is a substitution rather than a rewrite, and an
extraction-backed invoice is the same `Invoice` type with different provenance
attached.

## Fixtures

`data/fixtures/extraction/` holds three payloads in exactly this shape:

| file | what it demonstrates |
|---|---|
| `INV-00001-clean.json` | every field read, all confidences high — the gate passes and says so |
| `INV-00002-dropout.json` | fields the engine did not find; they stay at their base value marked `synthetic` |
| `INV-00003-confused.json` | digit confusion with low confidences — the gate fires before any model call |

`fcca.i2p.degrade.build_payload` generates these, seeded, so they are
reproducible rather than hand-tuned.

## What this does not solve

Line-item table extraction is the hard part of invoice OCR and none of it is
addressed here. The fixtures assume line structure is already known and only the
values are read. An adapter facing real documents has to solve table detection,
row segmentation and column assignment first, and a wrong row boundary produces
a confidently-read value on the wrong line — which this gate will not catch,
because the confidence is genuinely high. That is a real limitation of the
design and not a gap in the implementation.
