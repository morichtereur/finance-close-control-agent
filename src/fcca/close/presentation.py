"""Terminal rendering.

Deliberately plain: a control review is read, not admired. Tables, one accent per
severity, no boxes around boxes.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from fcca.close.controls.materiality import amount_in_reporting_currency
from fcca.close.evaluation.benchmark import NOT_RUN, BenchmarkRun
from fcca.close.models import CaseFailure, CaseResult

RISK_STYLE = {"high": "bold red", "medium": "bold yellow", "low": "bold green"}
SEVERITY_STYLE = {"critical": "red", "warning": "yellow", "info": "dim"}


def console() -> Console:
    """A console with a stable width, so piped output is not silently truncated."""
    return Console(highlight=False, width=118, soft_wrap=False)


def _money(value: float) -> str:
    return f"{value:,.2f}"


def render_case(result: CaseResult, con: Console, show_all_signals: bool = False) -> None:
    """Print a full control review for one case."""
    exception = result.exception
    entry = result.entry
    decision = result.decision

    con.print()
    con.print(
        Text.assemble(
            ("Exception ", "dim"),
            (exception.exception_id, "bold"),
            ("  ·  ", "dim"),
            (exception.exception_type.replace("_", " "), ""),
            ("  ·  ", "dim"),
            (f"{entry.company_code}", ""),
            ("  ·  ", "dim"),
            (f"period {exception.close_period}", "dim"),
        )
    )
    con.print(Text(exception.description, style="dim italic"))
    con.print()

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim", justify="right")
    facts.add_column()
    facts.add_row("journal entry", f"{entry.journal_id}   {entry.document_type}")
    facts.add_row("account", f"{entry.account}  {entry.account_name}")
    facts.add_row("cost center", entry.cost_center)
    facts.add_row(
        "amount",
        f"{entry.currency} {_money(entry.amount)}"
        + (
            f"   (EUR {_money(amount_in_reporting_currency(entry))} reporting)"
            if entry.currency != "EUR"
            else ""
        ),
    )
    facts.add_row(
        "posted",
        f"{entry.posting_timestamp:%Y-%m-%d %H:%M} by {entry.user_id}"
        + ("   [manual]" if entry.manual_posting else "   [system]"),
    )
    facts.add_row("document date", f"{entry.document_date}  ({entry.days_to_post} days to post)")
    facts.add_row("support", entry.supporting_document or "none")
    facts.add_row("approver", entry.approved_by or "none")
    facts.add_row("reconciliation", entry.reconciliation_status)
    con.print(facts)
    con.print()

    signals = result.signals if show_all_signals else result.triggered_signals
    heading = f"Deterministic controls — {len(result.triggered_signals)} of {len(result.signals)} triggered"
    con.print(Text(heading, style="bold"))
    if not signals:
        con.print(Text("  no control check was triggered", style="dim"))
    else:
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        table.add_column("check", no_wrap=True)
        table.add_column("severity", no_wrap=True)
        table.add_column("observed", no_wrap=True)
        table.add_column("detail", overflow="fold")
        for signal in signals:
            table.add_row(
                f"{signal.check_id} {signal.name}",
                Text(signal.severity.upper(), style=SEVERITY_STYLE[signal.severity]),
                str(signal.observed_value if signal.observed_value is not None else ""),
                signal.detail,
            )
        con.print(table)
    con.print()

    con.print(
        Text.assemble(
            ("Risk        ", "dim"),
            (decision.risk_level.upper(), RISK_STYLE[decision.risk_level]),
            ("    confidence ", "dim"),
            (f"{decision.confidence:.2f}", ""),
            ("    classification ", "dim"),
            (decision.classification, ""),
        )
    )
    con.print(Text.assemble(("Finding     ", "dim"), (decision.finding, "")))
    con.print(Text.assemble(("Action      ", "dim"), (decision.recommended_action, "")))
    con.print(
        Text.assemble(
            ("Disposition ", "dim"),
            (
                "HUMAN REVIEW REQUIRED"
                if result.final_requires_human_review
                else "AUTO RECOMMENDATION",
                "bold red" if result.final_requires_human_review else "bold green",
            ),
        )
    )
    for reason in result.gate.reasons:
        con.print(Text(f"  · {reason}", style="dim"))
    con.print()

    con.print(Text("Policy evidence", style="bold"))
    if not result.evidence:
        con.print(Text("  no policy passage passed the relevance threshold", style="dim"))
    cited = {(c.document, c.section) for c in decision.policy_citations}
    for item in result.evidence:
        marker = ">" if (item.document, item.section) in cited else " "
        con.print(
            Text.assemble(
                (f"  {marker} ", "bold" if marker == ">" else "dim"),
                (f"{item.document} §{item.section}", "bold" if marker == ">" else ""),
                (f"   relevance {item.score:.2f}", "dim"),
                (f"   {item.source_path}", "dim"),
            )
        )
    if result.grounding.ungrounded_citations:
        con.print(
            Text(
                "  ! ungrounded citations stripped: "
                + ", ".join(result.grounding.ungrounded_citations),
                style="red",
            )
        )
    con.print()
    con.print(Text(f"Rationale   {decision.rationale}", style="dim"))
    con.print()
    con.print(
        Text(
            f"{result.run.provider}:{result.run.model}  ·  {result.run.latency_ms} ms  ·  "
            f"{result.run.parse_attempts} parse attempt(s)  ·  "
            f"mode {result.run.structured_output_mode}  ·  prompt {result.run.prompt_sha256[:12]}",
            style="dim",
        )
    )
    con.print()


def render_failure(failure: CaseFailure, con: Console) -> None:
    con.print()
    con.print(Text(f"Exception {failure.exception_id}", style="bold"))
    con.print(Text(f"  automated assessment FAILED at stage '{failure.stage}'", style="bold red"))
    con.print(Text(f"  {failure.error}", style="red"))
    con.print(Text("  Disposition: HUMAN REVIEW REQUIRED", style="bold red"))
    con.print()


def render_benchmark(runs: list[BenchmarkRun], con: Console) -> None:
    """Print the provider comparison table."""
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("provider", no_wrap=True)
    table.add_column("model", no_wrap=True, max_width=30, overflow="ellipsis")
    for column in (
        "status",
        "cases",
        "risk acc",
        "esc prec",
        "esc rec",
        "cite acc",
        "valid out",
        "p50 ms",
    ):
        table.add_column(column, no_wrap=True)

    for run in runs:
        if run.metrics is None:
            table.add_row(
                run.provider,
                run.model,
                Text(NOT_RUN, style="yellow"),
                *[Text(NOT_RUN, style="dim") for _ in range(7)],
            )
            continue
        m = run.metrics
        table.add_row(
            run.provider,
            run.model,
            Text("ok", style="green"),
            str(m.n_cases),
            f"{m.risk_accuracy:.2f}",
            f"{m.escalation_precision:.2f}",
            f"{m.escalation_recall:.2f}",
            f"{m.citation_accuracy:.2f}",
            f"{m.structured_output_success:.2f}",
            f"{m.median_latency_ms:.0f}",
        )
    con.print(table)
    for run in runs:
        if run.note:
            con.print(Text(f"  {run.provider}: {run.note}", style="dim"))
