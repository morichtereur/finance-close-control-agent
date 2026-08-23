"""Deterministic local provider.

This is a real ``BaseChatModel``, not a patched-out function. The workflow, the
prompt, the parser, the grounding check, the gate and the audit trail all run
exactly as they do against a cloud model — only the token generation is replaced.
That is what makes the mock useful for CI, for demos and for screenshots, and it
is why the provider-portability test is meaningful.

**What the stub is, honestly.** It reads the structured case block from the
prompt and applies the risk-rating rubric of the Materiality and Escalation
Policy directly in Python. It is a rule engine wearing a chat-model interface. It
therefore says nothing about how a real model performs, and evaluation results
produced with it measure *pipeline integrity* — schema validity, citation
grounding, gate behaviour, audit completeness — not model quality. The benchmark
output labels mock rows accordingly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

#: Checks that mark a breach of a control requirement rather than a risk indicator.
_BREACH_CHECKS = {"CHK-01", "CHK-02", "CHK-04", "CHK-10", "CHK-14", "CHK-15"}

#: Checks that, on their own, make an item high risk under the policy rubric.
_HIGH_RISK_CHECKS = {"CHK-13", "CHK-06", "CHK-07", "CHK-03", "CHK-16"}

#: Indicators that do not by themselves warrant action when nothing else fired.
_SOFT_INDICATORS = {"CHK-05", "CHK-08", "CHK-09", "CHK-12", "CHK-17", "CHK-18"}


class MockChatModel(BaseChatModel):
    """Rule-driven chat model used for local development, tests and CI."""

    model_name: str = "deterministic-stub-v1"

    @property
    def _llm_type(self) -> str:
        return "fcca-mock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        case = _extract_case(messages)
        # One stub serves both modules. Dispatch on the case shape rather than on
        # a flag passed in, so the mock cannot be pointed at the wrong rubric by
        # a caller that forgot to set something.
        decision = decide_invoice(case) if "invoice_id" in case else decide(case)
        message = AIMessage(content=json.dumps(decision, indent=2))
        return ChatResult(generations=[ChatGeneration(message=message)])


def _extract_case(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Pull the structured case block out of the rendered prompt."""
    for message in reversed(messages):
        content = message.content if isinstance(message.content, str) else str(message.content)
        match = _JSON_BLOCK.search(content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return {}


def decide(case: dict[str, Any]) -> dict[str, Any]:
    """Apply the policy rubric to a structured case.

    Exposed separately from :meth:`MockChatModel._generate` so the rule set can be
    tested directly, and so the fixture used by the tests is the same code path
    the demo runs.
    """
    exception_id = str(case.get("exception_id", "UNKNOWN"))
    exception_type = str(case.get("exception_type", "unusual_journal_entry"))
    amount = float(case.get("amount_reporting_ccy", 0.0) or 0.0)
    trivial = float(case.get("thresholds", {}).get("trivial_threshold", 5_000.0))
    signals: list[dict[str, Any]] = list(case.get("control_signals", []))
    citations: list[dict[str, str]] = list(case.get("available_citations", []))

    fired = {s["check_id"] for s in signals if s.get("triggered")}
    critical = {
        s["check_id"] for s in signals if s.get("triggered") and s.get("severity") == "critical"
    }

    # CHK-15 is severity-dependent: a difference above the investigation threshold
    # is a warning, one above group materiality is critical and rates high.
    if fired & _HIGH_RISK_CHECKS or "CHK-15" in critical or len(critical) >= 2:
        risk = "high"
    elif fired & _BREACH_CHECKS and abs(amount) >= trivial:
        risk = "medium"
    else:
        risk = "low"

    action = _action_for(risk, fired, exception_type)
    classification = exception_type if fired else "no_finding"
    confidence = {"high": 0.87, "medium": 0.72, "low": 0.85}[risk]

    return {
        "exception_id": exception_id,
        "classification": classification,
        "risk_level": risk,
        "finding": _finding_text(signals, exception_type, risk),
        "recommended_action": _ACTION_TEXT[action],
        "action_category": action,
        "requires_human_review": risk != "low",
        "confidence": confidence,
        "policy_citations": citations[:2],
        "rationale": _rationale(signals, risk, action),
    }


def _action_for(risk: str, fired: set[str], exception_type: str) -> str:
    if risk == "high":
        if "CHK-07" in fired:
            return "propose_correcting_entry"
        if "CHK-03" in fired:
            return "refer_to_internal_audit"
        if exception_type == "material_variance":
            return "request_justification"
        return "escalate_to_financial_controller"
    if risk == "medium":
        if "CHK-01" in fired:
            return "request_supporting_documentation"
        if fired & {"CHK-04", "CHK-10"}:
            return "request_justification"
        if "CHK-02" in fired:
            return "route_to_reviewer"
        if fired & {"CHK-14", "CHK-15"}:
            return "route_to_preparer"
        return "route_to_preparer"
    if not fired or fired <= _SOFT_INDICATORS:
        return "no_action"
    return "route_to_preparer"


_ACTION_TEXT: dict[str, str] = {
    "no_action": "No further action; document the disposition and close the exception.",
    "request_supporting_documentation": "Request the supporting document from the preparer within two working days.",
    "request_justification": "Obtain a documented justification from the preparer and retain it with the entry.",
    "route_to_preparer": "Route to the preparer for confirmation and documentation.",
    "route_to_reviewer": "Route to an independent reviewer for second-level approval before sign-off.",
    "escalate_to_financial_controller": "Escalate to the entity Financial Controller before entity sign-off.",
    "propose_correcting_entry": "Prepare a correcting entry referencing both source documents, for approval.",
    "refer_to_internal_audit": "Refer directly to Internal Audit as a potential control failure.",
}


def _finding_text(signals: list[dict[str, Any]], exception_type: str, risk: str) -> str:
    triggered = [s for s in signals if s.get("triggered")]
    if not triggered:
        return (
            "No deterministic control check was breached; the monitoring rule fired on a "
            "routine posting."
        )
    lead = triggered[0]
    label = str(lead.get("name", "control")).replace("_", " ")
    extra = f" and {len(triggered) - 1} further indicator(s)" if len(triggered) > 1 else ""
    return (
        f"{exception_type.replace('_', ' ').capitalize()}: {label} breached{extra}. Rated {risk}."
    )


def _rationale(signals: list[dict[str, Any]], risk: str, action: str) -> str:
    triggered = [s for s in signals if s.get("triggered")]
    if not triggered:
        return (
            "No control check was breached and the amount is below the clearly trivial "
            "threshold, so the item is rated low and no remediation is proposed."
        )
    detail = "; ".join(f"{s['check_id']} {s.get('name')}" for s in triggered[:4])
    return (
        f"Triggered checks: {detail}. Under the risk rubric of the Materiality and Escalation "
        f"Policy this combination is rated {risk}, which implies '{action}'."
    )


# ---------------------------------------------------------------------------
# Invoice-to-pay rubric
# ---------------------------------------------------------------------------

#: What the stub proposes for each exception class. This is a lookup table, and
#: saying so matters: it is not a model's judgement about the right remedy, it is
#: the remedy the process already has for that class of problem. Evaluation rows
#: produced with the mock therefore measure whether the pipeline carries a
#: proposal through validation, grounding and routing intact — not whether a
#: model chooses well.
_INVOICE_ACTIONS: dict[str, str] = {
    "bank_details_mismatch": "refer_to_vendor_master_team",
    "duplicate_invoice": "reject_as_duplicate",
    "missing_or_delayed_goods_receipt": "request_goods_receipt",
    "price_variance": "block_for_price_review",
    "quantity_variance": "propose_credit_memo",
    "gl_account_missing": "derive_and_post",
    "cost_center_missing": "route_to_requisitioner",
    "no_exception": "post_invoice",
}

#: How sure the stub claims to be, by class. Lower where the resolution depends
#: on reading prose than where it is a table lookup, because that is where a real
#: model would be less sure too, and a stub that claimed 0.95 everywhere would
#: make the routing thresholds untestable.
_INVOICE_CONFIDENCE: dict[str, float] = {
    "bank_details_mismatch": 0.93,
    "duplicate_invoice": 0.91,
    "missing_or_delayed_goods_receipt": 0.90,
    "price_variance": 0.88,
    "quantity_variance": 0.88,
    "gl_account_missing": 0.94,
    "cost_center_missing": 0.72,
    "no_exception": 0.90,
}

#: Precedence when several rules fired, mirroring the engine's own ordering.
_INVOICE_PRECEDENCE = (
    "bank_details_mismatch",
    "duplicate_invoice",
    "quantity_variance",
    "price_variance",
    "missing_or_delayed_goods_receipt",
    "cost_center_missing",
    "gl_account_missing",
)


def decide_invoice(case: dict[str, Any]) -> dict[str, Any]:
    """Apply the invoice-to-pay rubric to a structured case.

    Exposed separately so the rule set can be tested directly and so the fixture
    the tests use is the same code path the demo runs.
    """
    invoice_id = str(case.get("invoice_id", "UNKNOWN"))
    findings: list[dict[str, Any]] = list(case.get("findings", []))
    available: list[str] = list(case.get("available_evidence", []))
    valid_cost_centers: list[str] = list(case.get("valid_cost_centers", []))

    types = {str(f.get("exception_type")) for f in findings}
    classification = next((t for t in _INVOICE_PRECEDENCE if t in types), "no_exception")
    action = _INVOICE_ACTIONS.get(classification, "escalate_to_ap_manager")
    confidence = _INVOICE_CONFIDENCE.get(classification, 0.5)

    # Cite the evidence the winning finding actually rests on, restricted to
    # paths the case offered. The stub cannot invent a field path, which is why
    # the grounding test uses a hand-built assessment rather than this output.
    winning = next((f for f in findings if str(f.get("exception_type")) == classification), None)
    cited: list[str] = []
    if winning is not None:
        rule_path = f"finding[{winning.get('rule_id')}].detail"
        if rule_path in available:
            cited.append(rule_path)
        line_no = winning.get("line_no")
        if line_no is not None:
            for suffix in (
                "price.residual_pct",
                "price.line_residual_abs",
                "quantity.residual_base_qty",
            ):
                path = f"line[{line_no}].{suffix}"
                if path in available:
                    cited.append(path)
    if classification == "bank_details_mismatch":
        cited += [p for p in ("invoice.stated_bank_iban", "vendor.bank_iban") if p in available]
    if classification == "duplicate_invoice" and "duplicate_candidates" in available:
        cited.append("duplicate_candidates")

    proposed_cost_center = None
    if classification == "cost_center_missing":
        proposed_cost_center = _cost_center_from_free_text(
            str(case.get("invoice", {}).get("free_text", "")), valid_cost_centers
        )
        if proposed_cost_center is None:
            confidence = 0.35

    return {
        "invoice_id": invoice_id,
        "classification": classification,
        "proposed_action": action,
        "rationale": _invoice_rationale(classification, findings),
        "evidence": [{"field_path": path} for path in cited[:4]],
        "confidence": confidence,
        "proposed_cost_center": proposed_cost_center,
    }


def _cost_center_from_free_text(free_text: str, valid: list[str]) -> str | None:
    """Match the free-text note against cost-centre names and aliases.

    Substring matching over a closed list, not extraction. The stub can only ever
    return a code that exists, which is the same constraint the real prompt
    imposes and the agent layer re-checks. It is a poor imitation of what a model
    does here and it is meant to be: this is the one exception class where the
    stub's number should not be read as an estimate of model performance.
    """
    from fcca.i2p.masterdata import COST_CENTERS_BY_ID

    lowered = free_text.lower()
    for code in valid:
        centre = COST_CENTERS_BY_ID.get(code)
        if centre is None:
            continue
        for alias in (*centre.aliases, centre.name):
            if alias.lower() in lowered:
                return code
    return None


def _invoice_rationale(classification: str, findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No rule fired; the invoice matches on price, quantity and coding."
    detail = next(
        (str(f.get("detail")) for f in findings if str(f.get("exception_type")) == classification),
        str(findings[0].get("detail", "")),
    )
    others = len(findings) - 1
    suffix = f" {others} further finding(s) on the same invoice." if others > 0 else ""
    return f"{detail}{suffix}"
