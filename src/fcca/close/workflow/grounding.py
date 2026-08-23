"""Citation grounding.

A model may only cite policy that was actually put in front of it. Every citation
it returns is matched against the retrieved evidence set; anything unmatched is
stripped from the decision and recorded as ungrounded.

This is the difference between "the system cited a policy" and "the system's
recommendation is traceable to a passage a reviewer can open". It also gives the
evaluation suite an honest measure — the unsupported-recommendation rate — rather
than trusting the model's own claim to have used a source.
"""

from __future__ import annotations

import re

from fcca.close.models import ControlDecision, GroundingReport, PolicyEvidence

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalise_document(value: str) -> str:
    return _NON_ALNUM.sub(" ", value.lower()).strip()


def _normalise_section(value: str) -> str:
    """Reduce '§4.2', '4.2 Second-level approval' and '4.2.' to '4.2'."""
    cleaned = value.replace("§", " ").strip()
    head = cleaned.split(" ", 1)[0].strip().rstrip(".")
    return head or cleaned.lower()


def ground_citations(
    decision: ControlDecision, evidence: list[PolicyEvidence]
) -> tuple[ControlDecision, GroundingReport]:
    """Strip citations that were not retrieved and report on what was removed.

    Returns the cleaned decision and a :class:`GroundingReport`. The original
    decision object is not mutated, so the raw model output stays intact for the
    audit trail.
    """
    retrieved_sections = {
        (_normalise_document(item.document), _normalise_section(item.section)) for item in evidence
    }
    retrieved_documents = {_normalise_document(item.document) for item in evidence}

    grounded = []
    ungrounded: list[str] = []
    for citation in decision.policy_citations:
        document = _normalise_document(citation.document)
        section = _normalise_section(citation.section)
        if (document, section) in retrieved_sections or (
            document in retrieved_documents and not section
        ):
            grounded.append(citation)
        else:
            ungrounded.append(f"{citation.document} §{citation.section}")

    cleaned = decision.model_copy(update={"policy_citations": grounded})
    report = GroundingReport(
        total_citations=len(decision.policy_citations),
        grounded_citations=len(grounded),
        ungrounded_citations=ungrounded,
    )
    return cleaned, report


def cited_documents(decision: ControlDecision) -> set[str]:
    """Normalised set of policy documents the decision relies on."""
    return {_normalise_document(c.document) for c in decision.policy_citations}
