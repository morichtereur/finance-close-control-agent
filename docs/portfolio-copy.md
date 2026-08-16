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

A working prototype with a benchmark harness rather than a benchmark result. The 60-case labelled evaluation runs
end to end on the deterministic local provider with complete structured-output validity, correct escalation gating and
a full audit record for every case — which measures pipeline integrity, not model quality. The Bedrock and Vertex
adapters are implemented but have not been run against live endpoints, so those rows of the comparison table read
`not_run`. The point of the harness is that filling them in is one command with an account attached.

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
  and it changes. The portability is enforced by a test that runs the whole pipeline against a second chat model.
- **Why no autonomous agent?** In a close, the sequence of checks *is* the control design. An agent that plans its own
  path produces a different audit trail for every case, and a missed check becomes invisible.
- **What's the weakest part?** No live cloud benchmark yet, and labels that are ground truth by construction rather
  than by controller review. Both are stated in the README rather than papered over.
