"""The control workflow.

A fixed six-stage LangChain pipeline:

.. code-block:: text

    load case -> deterministic controls -> policy retrieval -> LLM classification
              -> citation grounding -> review gate -> audit

LangChain is used for what it is good at here: composing typed steps into a named,
traceable sequence with one uniform chat-model interface underneath, so the same
pipeline runs against a local stub, Bedrock or Vertex AI without a branch.

It is *not* used to let a model choose its own path. The order of controls in a
close is a control design, not a plan the system may improvise — see
:mod:`fcca.close.workflow.tools` for the reasoning.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from fcca.close.analytics import CloseAnalytics
from fcca.close.controls.engine import run_controls, to_close_exception, to_journal_entry
from fcca.close.models import (
    CaseFailure,
    CaseResult,
    ControlDecision,
    RunMetadata,
)
from fcca.close.retrieval.retriever import PolicyRetrievalService
from fcca.close.workflow.gate import apply_gate, failed_case_gate
from fcca.close.workflow.grounding import ground_citations
from fcca.close.workflow.prompts import PROMPT_VERSION, build_messages
from fcca.close.workflow.structured import invoke_structured
from fcca.shared.audit.logger import AuditLog
from fcca.shared.config import ProviderName, Settings, get_settings
from fcca.shared.errors import FCCAError, StructuredOutputError
from fcca.shared.providers.base import ProviderSpec, describe_provider, get_llm
from fcca.shared.trace import TraceWriter

logger = logging.getLogger(__name__)


class ControlAgent:
    """Assesses one close exception end to end."""

    def __init__(
        self,
        settings: Settings,
        analytics: CloseAnalytics,
        retrieval: PolicyRetrievalService,
        llm: BaseChatModel,
        spec: ProviderSpec,
        audit: AuditLog | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self.settings = settings
        self.analytics = analytics
        self.retrieval = retrieval
        self.llm = llm
        self.spec = spec
        self.audit = audit
        self.trace = trace or TraceWriter(settings.close_trace_path, module="close")
        self._chain = self._build_chain()

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def build(
        cls,
        provider: ProviderName | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
        with_audit: bool = True,
    ) -> ControlAgent:
        """Construct an agent for a provider.

        This is the only place a caller names a provider. Everything below this
        line works against ``BaseChatModel``.
        """
        settings = settings or get_settings()
        spec = describe_provider(provider, model_name, settings)
        llm = get_llm(provider, model_name, settings)
        return cls(
            settings=settings,
            analytics=CloseAnalytics(settings),
            retrieval=PolicyRetrievalService(settings),
            llm=llm,
            spec=spec,
            audit=AuditLog(settings) if with_audit else None,
        )

    def close(self) -> None:
        self.analytics.close()
        if self.audit is not None:
            self.audit.close()

    def __enter__(self) -> ControlAgent:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ chain
    def _build_chain(self) -> Runnable:
        return (
            RunnableLambda(self._load_case, name="load_case")
            | RunnableLambda(self._deterministic_controls, name="deterministic_controls")
            | RunnableLambda(self._retrieve_policy, name="policy_retrieval")
            | RunnableLambda(self._classify, name="llm_classification")
            | RunnableLambda(self._ground_and_gate, name="grounding_and_review_gate")
            | RunnableLambda(self._assemble, name="assemble_result")
        )

    # ------------------------------------------------------------------ steps
    def _load_case(self, state: dict[str, Any]) -> dict[str, Any]:
        exception = to_close_exception(self.analytics.exception(state["exception_id"]))
        entry = to_journal_entry(self.analytics.journal_entry(exception.journal_id))
        self.trace.step(
            case_id=exception.exception_id,
            step_name="intake",
            actor="rule",
            rule_id="CLOSE-STEP-01",
            inputs={"exception_id": state["exception_id"]},
            outcome="loaded",
            summary=(
                f"Loaded {exception.exception_id} ({exception.exception_type}) "
                f"for entry {entry.journal_id}, {entry.currency} {entry.amount:,.2f}."
            ),
            detail={
                "journal_id": entry.journal_id,
                "company_code": entry.company_code,
                "close_period": exception.close_period,
                "amount_reporting_ccy": entry.amount_reporting_ccy,
            },
        )
        return {**state, "stage": "controls", "exception": exception, "entry": entry}

    def _deterministic_controls(self, state: dict[str, Any]) -> dict[str, Any]:
        signals = run_controls(state["entry"], state["exception"], self.analytics, self.settings)
        triggered = [s for s in signals if s.triggered]
        self.trace.step(
            case_id=state["exception"].exception_id,
            step_name="deterministic_controls",
            actor="rule",
            rule_id="CLOSE-STEP-02",
            inputs=state["entry"].model_dump(mode="json"),
            outcome="exception" if triggered else "pass",
            summary=(
                f"{len(triggered)} of {len(signals)} control checks triggered"
                + (f": {', '.join(s.check_id for s in triggered)}." if triggered else ".")
            ),
            detail={
                "triggered": [
                    {
                        "check_id": s.check_id,
                        "name": s.name,
                        "severity": s.severity,
                        "observed_value": s.observed_value,
                        "threshold": s.threshold,
                    }
                    for s in triggered
                ],
                "checks_run": len(signals),
            },
        )
        return {**state, "stage": "retrieval", "signals": signals}

    def _retrieve_policy(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence = self.retrieval.retrieve_for_case(
            state["exception"], state["entry"], state["signals"]
        )
        self.trace.step(
            case_id=state["exception"].exception_id,
            step_name="policy_retrieval",
            actor="rule",
            rule_id="CLOSE-STEP-03",
            inputs=[s.check_id for s in state["signals"] if s.triggered],
            outcome="retrieved" if evidence else "no_evidence",
            summary=(
                f"Retrieved {len(evidence)} policy section(s): "
                + "; ".join(e.citation for e in evidence)
                if evidence
                else "No policy section met the relevance threshold."
            ),
            detail={
                "sections": [
                    {"citation": e.citation, "node_id": e.node_id, "score": round(e.score, 4)}
                    for e in evidence
                ]
            },
        )
        return {**state, "stage": "inference", "evidence": evidence}

    def _classify(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(
            state["exception"],
            state["entry"],
            state["signals"],
            state["evidence"],
            self.settings,
        )
        prompt_hash = hashlib.sha256(
            "\n".join(str(m.content) for m in messages).encode("utf-8")
        ).hexdigest()[:32]

        state = {**state, "stage": "validation", "prompt_sha256": prompt_hash}
        invocation = invoke_structured(self.llm, messages, ControlDecision, self.settings)
        decision = invocation.value
        assert isinstance(decision, ControlDecision)

        # The model echoes the exception id; the pipeline owns it.
        if decision.exception_id != state["exception"].exception_id:
            decision = decision.model_copy(update={"exception_id": state["exception"].exception_id})

        self.trace.step(
            case_id=state["exception"].exception_id,
            step_name="llm_classification",
            actor="model",
            model=self.spec.model,
            prompt_version=PROMPT_VERSION,
            inputs={"prompt_sha256": prompt_hash},
            outcome=decision.classification,
            summary=(
                f"Classified as {decision.classification}, risk {decision.risk_level}, "
                f"confidence {decision.confidence:.2f}; proposed "
                f"{decision.action_category}."
            ),
            detail={
                "risk_level": decision.risk_level,
                "confidence": decision.confidence,
                "action_category": decision.action_category,
                "citations": [f"{c.document} §{c.section}" for c in decision.policy_citations],
                "parse_attempts": invocation.attempts,
                "latency_ms": invocation.latency_ms,
            },
        )

        return {
            **state,
            "decision": decision,
            "raw_output": invocation.raw_text,
            "run": RunMetadata(
                provider=self.spec.provider,
                model=self.spec.model,
                latency_ms=invocation.latency_ms,
                prompt_sha256=prompt_hash,
                input_tokens=invocation.input_tokens,
                output_tokens=invocation.output_tokens,
                estimated_cost_usd=self._estimate_cost(
                    invocation.input_tokens, invocation.output_tokens
                ),
                parse_attempts=invocation.attempts,
                structured_output_mode=self.settings.structured_output_mode,
            ),
        }

    def _ground_and_gate(self, state: dict[str, Any]) -> dict[str, Any]:
        decision, grounding = ground_citations(state["decision"], state["evidence"])
        self.trace.step(
            case_id=state["exception"].exception_id,
            step_name="citation_grounding",
            actor="rule",
            rule_id="CLOSE-STEP-05",
            inputs={"citations": grounding.total_citations},
            outcome="grounded" if grounding.is_fully_grounded else "ungrounded_stripped",
            summary=(
                f"{grounding.grounded_citations} of {grounding.total_citations} "
                "citation(s) matched retrieved policy"
                + (
                    f"; stripped {', '.join(grounding.ungrounded_citations)}."
                    if grounding.ungrounded_citations
                    else "."
                )
            ),
            detail=grounding.model_dump(mode="json"),
        )

        gate = apply_gate(decision, grounding, state["signals"], self.settings)
        self.trace.step(
            case_id=state["exception"].exception_id,
            step_name="review_gate",
            actor="rule",
            rule_id="CLOSE-STEP-06",
            inputs={
                "risk_level": decision.risk_level,
                "confidence": decision.confidence,
                "action_category": decision.action_category,
                "triggered": [s.check_id for s in state["signals"] if s.triggered],
            },
            outcome=gate.disposition,
            summary=(
                "Routed to human review: " + gate.reasons[0]
                if gate.requires_human_review and gate.reasons
                else "Cleared as an automatic recommendation; no gate condition fired."
            ),
            detail={"reasons": gate.reasons},
        )
        return {**state, "decision": decision, "grounding": grounding, "gate": gate}

    def _assemble(self, state: dict[str, Any]) -> dict[str, Any]:
        result = CaseResult(
            exception=state["exception"],
            entry=state["entry"],
            signals=state["signals"],
            evidence=state["evidence"],
            decision=state["decision"],
            grounding=state["grounding"],
            gate=state["gate"],
            run=state["run"],
            decided_at=datetime.now(UTC),
        )
        return {"result": result, "raw_output": state.get("raw_output", "")}

    # -------------------------------------------------------------- execution
    def run(self, exception_id: str) -> CaseResult:
        """Assess one exception. Raises on failure; see :meth:`run_safe`."""
        output = self._chain.invoke({"exception_id": exception_id, "stage": "load"})
        result = output["result"]
        assert isinstance(result, CaseResult)
        if self.audit is not None:
            # The unvalidated response is kept so a reviewer can see what the model
            # actually said, not only what survived validation.
            self.audit.record_case_with_raw(result, raw_output=output["raw_output"])
        return result

    def run_safe(self, exception_id: str) -> CaseResult | CaseFailure:
        """Assess one exception, converting failures into an explicit failed state.

        A failure is never a silent pass: it is written to the audit log and
        returned as a :class:`CaseFailure` whose gate requires human review.
        """
        try:
            return self.run(exception_id)
        except StructuredOutputError as exc:
            return self._record_failure(exception_id, "validation", str(exc))
        except FCCAError as exc:
            return self._record_failure(exception_id, "load", str(exc))
        except Exception as exc:
            logger.exception("unexpected failure assessing %s", exception_id)
            return self._record_failure(exception_id, "inference", f"{type(exc).__name__}: {exc}")

    def _record_failure(self, exception_id: str, stage: str, error: str) -> CaseFailure:
        if self.audit is not None:
            self.audit.record_failure(
                exception_id=exception_id,
                provider=self.spec.provider,
                model=self.spec.model,
                error=error,
            )
        return CaseFailure(
            exception_id=exception_id,
            stage=stage,  # type: ignore[arg-type]
            error=error,
            provider=self.spec.provider,
            model=self.spec.model,
            gate=failed_case_gate(error),
            failed_at=datetime.now(UTC),
        )

    # ---------------------------------------------------------------- helpers
    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Estimate USD cost, but only when real prices are configured and real
        token counts were reported. Never guessed."""
        if input_tokens is None or output_tokens is None:
            return None
        if self.settings.input_cost_per_mtok is None or self.settings.output_cost_per_mtok is None:
            return None
        return round(
            input_tokens / 1_000_000 * self.settings.input_cost_per_mtok
            + output_tokens / 1_000_000 * self.settings.output_cost_per_mtok,
            6,
        )
