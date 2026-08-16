"""LangChain tools exposing the deterministic capabilities.

These are real ``StructuredTool`` objects with typed arguments, and they are the
same functions the workflow calls. What they are *not* is a toolbox handed to an
autonomous agent to use as it sees fit.

That is a deliberate architectural decision. In a finance control process the
sequence of checks is a control design, not a planning problem: every exception
must receive the same checks in the same order, or the population is no longer
comparable and the close is not auditable. An agent that decides for itself which
controls to run produces a different audit trail for every case.

They are defined as tools anyway because it costs nothing, it makes the
capability surface explicit and typed, and it is the natural seam if a future
version does want a model-driven triage step for a bounded sub-problem.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from fcca.analytics import CloseAnalytics
from fcca.config import Settings, get_settings
from fcca.controls.engine import to_journal_entry
from fcca.controls.journal_checks import check_supporting_document
from fcca.controls.materiality import materiality_band
from fcca.retrieval.retriever import PolicyRetrievalService


def build_control_tools(
    analytics: CloseAnalytics,
    retrieval: PolicyRetrievalService,
    settings: Settings | None = None,
) -> list[StructuredTool]:
    """Build the tool set bound to a specific dataset and policy index."""
    settings = settings or get_settings()

    def calculate_materiality(amount_reporting_ccy: float) -> dict[str, Any]:
        """Classify an absolute amount in reporting currency against policy thresholds."""
        band = materiality_band(abs(amount_reporting_ccy), settings)
        return {
            "band": band,
            "group_materiality": settings.materiality_group,
            "approval_threshold": settings.journal_approval_threshold,
            "trivial_threshold": settings.trivial_threshold,
            "is_material": band == "material",
        }

    def get_account_risk(account: str) -> dict[str, Any]:
        """Return the inherent risk rating of a general ledger account."""
        high = account in settings.high_risk_accounts
        return {
            "account": account,
            "risk_rating": "high" if high else "standard",
            "source": "control catalogue (Journal Entry Policy §6)",
        }

    def retrieve_policy(query: str) -> list[dict[str, Any]]:
        """Retrieve the most relevant policy passages for a query, with citations."""
        return [
            {
                "document": item.document,
                "section": item.section,
                "score": item.score,
                "passage": item.passage,
                "node_id": item.node_id,
            }
            for item in retrieval.retrieve(query)
        ]

    def check_document_support(journal_id: str) -> dict[str, Any]:
        """Check whether a journal entry carries a supporting document reference."""
        entry = to_journal_entry(analytics.journal_entry(journal_id))
        signal = check_supporting_document(entry, settings)
        return signal.model_dump()

    def check_reconciliation_status(company_code: str, account: str, period: str) -> dict[str, Any]:
        """Return the reconciliation record for an account and close period."""
        record = analytics.reconciliation(company_code, account, period)
        if record is None:
            return {"found": False, "reason": "no reconciliation exists for this account/period"}
        return {"found": True, **{k: str(v) for k, v in record.items()}}

    def calculate_variance(company_code: str, account: str, period: str) -> dict[str, Any]:
        """Compare an account's period movement against the prior-period average."""
        variance = analytics.account_variance(company_code, account, period)
        return {
            "current_period_total": round(variance.current_period_total, 2),
            "prior_average": round(variance.prior_average, 2),
            "absolute_change": round(variance.absolute_change, 2),
            "percent_change": (
                round(variance.percent_change, 1) if variance.percent_change is not None else None
            ),
        }

    return [
        StructuredTool.from_function(calculate_materiality),
        StructuredTool.from_function(get_account_risk),
        StructuredTool.from_function(retrieve_policy),
        StructuredTool.from_function(check_document_support),
        StructuredTool.from_function(check_reconciliation_status),
        StructuredTool.from_function(calculate_variance),
    ]


def tool_catalogue(tools: list[StructuredTool]) -> list[dict[str, str]]:
    """Human-readable listing of the available tools, used by ``fcca info``."""
    return [{"name": t.name, "description": (t.description or "").strip()} for t in tools]
