# Portfolio copy

Ready-to-use copy for morichtereur.github.io. Nothing below is a number the evaluation suite did not produce.

---

## Title

**Finance Close Control Agent**

## Subtitle (≤ 15 words)

Evidence-based triage of month-end close exceptions, auditable and portable across enterprise model providers.

## Problem (40–60 words)

Group finance raises more month-end close exceptions than it has reviewer hours — unsupported manual entries, late
postings, reconciliation differences, suspected duplicates. Triage is an obvious candidate for automation and an
obviously dangerous one: a control needs the same checks on every item, evidence a reviewer can open, and an audit
trail that survives a question asked six months later.

## What I built (60–90 words)

A control workflow that assesses one close exception end to end. Eighteen deterministic checks run over a synthetic
ledger in DuckDB; LlamaIndex retrieves the governing policy sections with document, section and chunk id preserved;
LangChain orchestrates a fixed pipeline over a provider abstraction covering AWS Bedrock, Google Vertex AI and a local
stub. The model returns a validated Pydantic decision, every citation is checked against what was actually retrieved,
a deterministic gate decides whether a person must look — and the whole basis is reconstructable from the audit log.

## Stack

Python 3.12 · LangChain · LlamaIndex · AWS Bedrock · Google Vertex AI · Pydantic · DuckDB · SQLite · pytest · ruff · mypy

## Outcome

Benchmarked on AWS Bedrock over 60 labelled exceptions, on two models differing only by an
environment variable. Neither missed an escalation — recall 1.00, zero false negatives on both —
because the gate that forces review is deterministic and never asks the model. What the cheaper
model costs is reviewer noise, not safety: Haiku rates risk correctly on 70% of cases against
Sonnet's 87% and clears 1 of 12 benign items where Sonnet clears 5, at a third of the price and
half the latency. Structured output validated on every case for both, with no ungrounded citations.
Vertex remains `not_run`.

## So what

The interesting constraint in enterprise finance AI is not model capability — it is that a control has to produce the
same evidence twice. Once you accept that, the architecture stops being about the model: deterministic measurement
stays in code, the model is confined to interpretation, citations are grounded against what was actually retrieved,
and the escalation gate is the one component the model cannot influence. That design costs very little to build and is
the difference between a demo a CFO enjoys and a workflow an auditor accepts.

---

## Short card version (for a project grid)

> **Finance Close Control Agent**
> Month-end close exceptions triaged against retrieved policy evidence, with a deterministic human-review gate and a
> reconstructable audit trail. Runs on AWS Bedrock, Vertex AI, or fully offline.
> *Python · LangChain · LlamaIndex · Bedrock · Vertex AI · RAG · evaluation · auditability*

## One-line version (for a CV or LinkedIn)

Built a multi-cloud finance-control prototype (LangChain, LlamaIndex, Bedrock, Vertex AI) that triages month-end close
exceptions against retrieved policy evidence, with structured outputs, a deterministic human-review gate, an audit
trail and a labelled evaluation harness.

## Talking points if someone asks

- **Why not just call an LLM API?** Because the interesting requirements are the ones around the call: what the model
  is allowed to see, what shape it may answer in, which of its claims survive a grounding check, and who decides when
  it is wrong.
- **Why multi-cloud?** Enterprise model access is a procurement and data-residency decision, not an engineering one,
  and it changes. Two models were benchmarked through the same pipeline with no code change between them, and a test
  runs the whole pipeline against a second chat model to keep that true.
- **What did swapping the model actually show?** That the safety property is not the model's. Cutting model cost by
  two thirds moved risk accuracy from 0.87 to 0.70 and moved missed escalations not at all — they were zero on both,
  because a deterministic gate decides that, not the model.
- **Why no autonomous agent?** In a close, the sequence of checks *is* the control design. An agent that plans its own
  path produces a different audit trail for every case, and a missed check becomes invisible.
- **What's the weakest part?** Labels that are ground truth by construction rather than by controller review — the
  benchmark validates the pipeline, not the finance judgement. Run-to-run drift is at least bounded: the same model
  run twice moved one case of sixty, and moved it toward more review rather than less.
