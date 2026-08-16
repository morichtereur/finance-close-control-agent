"""Run the labelled evaluation for one or more providers.

    fcca evaluate --provider mock
    fcca evaluate --provider bedrock
    fcca evaluate --compare              # all three, uncredentialed ones as not_run

Results are written to ``results/benchmark.csv`` plus a per-provider detail file.
"""

from __future__ import annotations

import argparse
import logging
import sys

from fcca.config import get_settings
from fcca.errors import FCCAError
from fcca.evaluation.benchmark import (
    BenchmarkRun,
    collect_runs,
    run_benchmark,
    write_benchmark_csv,
    write_run_detail,
)
from fcca.presentation import console, render_benchmark

logger = logging.getLogger(__name__)

ALL_PROVIDERS = ("mock", "bedrock", "vertex")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca evaluate",
        description="Evaluate the control workflow against the labelled exception set.",
    )
    parser.add_argument(
        "--provider", choices=list(ALL_PROVIDERS), default=None, help="Provider to evaluate."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate every provider; unavailable ones are recorded as not_run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--no-audit", action="store_true", help="Do not write audit records for benchmark runs."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = get_settings()
    settings.ensure_directories()
    con = console()

    providers = list(ALL_PROVIDERS) if args.compare else [args.provider or settings.llm_provider]

    runs: list[BenchmarkRun] = []
    for provider in providers:

        def progress(exception_id: str, index: int, total: int, _p: str = provider) -> None:
            if not args.quiet:
                con.print(f"  [{_p}] {index}/{total} {exception_id}", style="dim", end="\r")

        try:
            run = run_benchmark(
                provider,  # type: ignore[arg-type]
                limit=args.limit,
                settings=settings,
                with_audit=not args.no_audit,
                on_case=progress,
            )
        except FCCAError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            con.print(" " * 60, end="\r")
        runs.append(run)
        if run.metrics is not None:
            write_run_detail(run, settings.results_dir / f"eval_{provider}.json")

    runs = collect_runs(runs, settings)
    csv_path = write_benchmark_csv(runs, settings.results_dir / "benchmark.csv")

    con.print()
    render_benchmark(runs, con)
    con.print()
    con.print(f"wrote {csv_path}", style="dim")

    detailed = next((r for r in runs if r.metrics is not None), None)
    if detailed is not None and detailed.metrics is not None:
        _print_detail(detailed, con)
    return 0


def _print_detail(run: BenchmarkRun, con: object) -> None:
    metrics = run.metrics
    assert metrics is not None
    from rich.console import Console

    assert isinstance(con, Console)
    con.print()
    con.print(f"Detail — {run.provider}:{run.model}", style="bold")
    matrix = metrics.confusion
    con.print(
        f"  escalation confusion: tp={matrix.get('tp')} fp={matrix.get('fp')} "
        f"tn={matrix.get('tn')} fn={matrix.get('fn')}",
        style="dim",
    )
    con.print(
        f"  structured output valid {metrics.structured_output_success:.2f} · "
        f"ungrounded citations {metrics.ungrounded_citation_rate:.2f} · "
        f"unsupported recommendations {metrics.unsupported_recommendation_rate:.2f}",
        style="dim",
    )
    weakest = sorted(metrics.per_scenario_risk_accuracy.items(), key=lambda kv: kv[1])[:5]
    if weakest and weakest[0][1] < 1.0:
        con.print("  weakest scenarios (risk accuracy):", style="dim")
        for scenario, accuracy in weakest:
            if accuracy < 1.0:
                con.print(f"    {scenario:34s} {accuracy:.2f}", style="dim")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
