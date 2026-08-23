# Portfolio copy

Ready-to-use copy for morichtereur.github.io. Nothing below is a number the evaluation suite did not produce.

---

## Title

**Finance Control Agent**

## Subtitle (≤ 15 words)

Two finance exception processes on one auditable spine, with arithmetic kept out of the model.

## Problem (40–60 words)

Two finance processes have the same shape. Month-end close raises more exceptions than there are reviewer hours;
accounts payable raises more invoice mismatches than there are clerk hours. Most items are benign, a few are not,
and the failure mode is not a bad summary — it is money out of the door against a document nobody checked.

## What I built (60–90 words)

Two modules sharing configuration, a provider factory, routing tiers and an append-only trace. Close runs eighteen
deterministic control checks over a synthetic ledger, retrieves the governing policy sections with document and
clause preserved, and gates the result. Invoice-to-pay runs twelve deterministic steps ending in a three-way match,
normalising price and quantity on both sides before comparing. One rule holds across both: no language model touches
arithmetic, matching or a tolerance decision — structurally, not as guidance.

## Stack

Python 3.12 · LangChain · LlamaIndex · AWS Bedrock · Pydantic · DuckDB · SQLite · pytest · ruff · mypy

## Outcome

The invoice-to-pay case that justifies the whole design: a vendor drops a discount schedule and lifts the list
price, the printed prices differ by 1.09% — inside the 2% tolerance, so a naive three-way match pays the invoice.
Normalised to a net price per base unit it is 15.79%, or EUR 46,811.74 on one line. Across 334 invoices, nothing
was auto-cleared that should not have been; that count is enforced by an exit code and two tests rather than
reported as a metric. On the close side, two live Bedrock models over 60 labelled exceptions missed no escalation
between them, though the cheaper model rated risk correctly on 70% of cases against the other's 87%.

## So what

The interesting constraint in enterprise finance AI is not model capability — it is that a control has to produce
the same evidence twice, and that a wrong number is not a wrong sentence. Once you accept both, the architecture
stops being about the model: measurement and matching stay in code, the model is confined to classifying and citing,
and the routing that decides whether money moves is the one component it cannot influence. That is the difference
between a demo a CFO enjoys and a workflow an auditor accepts.

---

## Short card version (for a project grid)

> **Finance Control Agent**
> Month-end close and invoice-to-pay on one spine: deterministic checks and matching, retrieved policy evidence, a
> human-review gate, and an append-only trace. No model touches arithmetic.
> *Python · LangChain · LlamaIndex · Bedrock · RAG · evaluation · auditability*

## One-line version (for a CV or LinkedIn)

Built a two-module finance control prototype (LangChain, LlamaIndex, AWS Bedrock) covering month-end close and
invoice-to-pay, where deterministic code performs all matching and tolerance decisions and the model only classifies
and cites — benchmarked on live Bedrock models with a labelled set and an append-only trace.

## Talking points if someone asks

- **Why not just call an LLM API?** Because the interesting requirements are the ones around the call: what the model
  is allowed to see, what shape it may answer in, which of its claims survive a grounding check, and who decides when
  it is wrong.
- **What is the sharpest example?** A 1.09% price difference that pays, and is 15.79% once both sides are normalised
  — EUR 46,811.74 on a single line. Discounts applied in sequence rather than summed; summing them misprices the line
  by a quarter of the tolerance on its own.
- **Why keep the model away from arithmetic?** Because a wrong sentence is visible and a wrong number is not. The
  matching module imports no provider and cannot call one; a test asserts every trace record from a deterministic run
  is attributed to a rule.
- **What did swapping the model show?** That the safety property is not the model's. Cutting model cost by two thirds
  moved risk accuracy from 0.87 to 0.70 and moved missed escalations not at all — zero on both.
- **What's the weakest part?** Labels derived from the scenarios that generated the data, so every 1.000 measures
  pipeline integrity rather than difficulty. Invoice-to-pay has not been run on a live provider at all; only close
  has. Both are stated in the README rather than papered over.
