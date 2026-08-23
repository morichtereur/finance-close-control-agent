"""Render the escalation outcome of a benchmark run as a single stacked bar.

One chart, one claim: of 60 labelled exceptions, how many reached a reviewer
who needed to see them, how many reached one who did not, and how many that
needed review were cleared instead. The last number is the one that matters and
it is zero, which is difficult to show — so it is annotated rather than drawn.

Reads `results/eval_<provider>.json`, which the benchmark writes, so the figure
cannot drift from the numbers it illustrates. Regenerate with:

    python -m fcca.close.figure --provider bedrock

Requires the optional `figures` extra; nothing else in the package imports
matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcca.shared.config import Settings, get_settings

#: Taken from the portfolio site's design tokens so the figure sits in the page
#: rather than on it.
BACKGROUND = "#e9eae4"
INK = "#121a17"
MUTED = "#59635e"
ACCENT = "#146b54"
ACCENT_SOFT = "#8fb3a6"
WARN = "#b8763a"
RULE = "#7e857a"


def render(
    provider: str = "bedrock", model: str | None = None, settings: Settings | None = None
) -> Path:
    """Write the escalation-outcome bar for one recorded run.

    Runs are keyed by provider and model, so the model has to be named — or
    left to the configured default, which is what a caller with one run per
    provider expects.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fcca.close.evaluation.benchmark import run_slug

    settings = settings or get_settings()
    model = model or settings.model_name_for(provider)  # type: ignore[arg-type]
    source = settings.results_dir / f"{run_slug(provider, model)}.json"
    if not source.exists():
        available = sorted(p.name for p in settings.results_dir.glob("eval_*.json"))
        raise FileNotFoundError(
            f"{source.name} not found — run `fcca evaluate --provider {provider}`. "
            f"Recorded runs: {available or 'none'}"
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    matrix = payload["metrics"]["confusion"]
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    total = tp + fp + tn + fn

    segments = [
        (tp, ACCENT, "Escalated, correctly"),
        (fp, WARN, "Escalated, unnecessarily"),
        (tn, ACCENT_SOFT, "Cleared, correctly"),
    ]

    fig, ax = plt.subplots(figsize=(12.0, 3.9), dpi=140)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    left = 0.0
    narrow_seen = 0
    for count, colour, label in segments:
        if count == 0:
            continue
        ax.barh(
            [0], [count], left=left, height=0.42, color=colour, edgecolor=BACKGROUND, linewidth=1.5
        )
        share = count / total * 100
        ax.text(
            left + count / 2,
            0,
            f"{count}",
            ha="center",
            va="center",
            color=BACKGROUND if colour in (ACCENT, WARN) else INK,
            fontsize=15,
            fontweight="bold",
        )
        # A narrow segment's caption is wider than the segment itself, so
        # consecutive narrow ones collide when both sit on the same line.
        # Stagger them and run a leader down to the bar.
        is_narrow = share < 20
        depth = -0.34 if not is_narrow else (-0.62 if narrow_seen % 2 == 0 else -1.02)
        centre = left + count / 2
        if is_narrow:
            ax.plot([centre, centre], [-0.24, depth + 0.06], color=RULE, linewidth=0.8, zorder=0)
            narrow_seen += 1
        ax.text(
            centre,
            depth,
            f"{label}\n{share:.0f}%",
            ha="center",
            va="top",
            color=MUTED,
            fontsize=10.5,
            linespacing=1.5,
        )
        left += count

    # The absent segment is the finding. Drawn as a gap marker at the right
    # edge, because a bar of width zero communicates nothing.
    ax.annotate(
        f"{fn} missed",
        xy=(total, 0),
        xytext=(total + total * 0.035, 0.30),
        color=WARN,
        fontsize=13,
        fontweight="bold",
        va="center",
        arrowprops={"arrowstyle": "-", "color": RULE, "linewidth": 1},
    )
    ax.text(
        total + total * 0.035,
        0.02,
        "No exception that required review\nwas cleared instead.",
        color=MUTED,
        fontsize=10.5,
        va="top",
        linespacing=1.5,
    )

    ax.set_xlim(0, total * 1.30)
    ax.set_ylim(-1.34, 0.72)
    ax.axis("off")
    ax.text(
        0,
        0.52,
        f"{total} labelled close exceptions · {payload['model']}",
        color=MUTED,
        fontsize=10.5,
        va="bottom",
    )

    destination = settings.results_dir / f"escalation_outcomes_{run_slug(provider, model)[5:]}.png"
    fig.savefig(destination, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return destination


#: Colour per line, mirroring what the CLI prints in a terminal. Keyed on
#: content rather than on ANSI codes: the fixture is captured through a pipe,
#: where rich correctly emits no colour at all.
#: Character columns of the control table, as `presentation.py` lays it out.
#: Colouring by column rather than by line is what keeps a severity from
#: bleeding into the sentence beside it — the earlier version painted a whole
#: CRITICAL row orange, so the first line of a detail was coloured and its
#: continuation was not, which reads as a rendering fault rather than emphasis.
_COL_SEVERITY = 33
_COL_OBSERVED = 43

AMBER = "#8a6a3a"


def _segments(line: str) -> list[tuple[int, str, str, bool]]:
    """Split one line into (start column, text, colour, bold) pieces."""
    stripped = line.strip()

    if line.startswith(("CHK-", "  CHK-")):
        return [
            (0, line[:_COL_SEVERITY].rstrip(), INK, False),
            (
                _COL_SEVERITY,
                line[_COL_SEVERITY:_COL_OBSERVED].rstrip(),
                WARN if "CRITICAL" in line else AMBER,
                True,
            ),
            (_COL_OBSERVED, line[_COL_OBSERVED:].rstrip(), MUTED, False),
        ]

    if stripped.startswith("Disposition"):
        return [
            (0, "Disposition", INK, True),
            (12, line[12:].rstrip(), WARN, True),
        ]

    if stripped.startswith(">"):
        return [(0, line.rstrip(), ACCENT, False)]

    if stripped.startswith(("Exception ", "Risk ", "Deterministic controls", "Policy evidence")):
        return [(0, line.rstrip(), INK, True)]

    if line.startswith(("bedrock:", "mock:", "vertex:")):
        return [(0, line.rstrip(), MUTED, False)]

    if line.startswith(("Finding", "Action", "Rationale")):
        return [(0, line[:12], INK, True), (12, line[12:].rstrip(), INK, False)]

    return [(0, line.rstrip(), MUTED if line.startswith(" ") else INK, False)]


def render_review(source: Path | None = None, settings: Settings | None = None) -> Path:
    """Typeset a captured control review as an image.

    The text is the tool's own output, committed at ``docs/example-review.txt``
    and reproduced verbatim apart from two marked abridgements. It is typeset
    rather than photographed because a terminal screenshot carries whatever
    font, theme and window chrome the machine happened to have, none of which
    is part of the work.

    Capture it with a wide console (``FCCA_CONSOLE_WIDTH=150``). At terminal
    width the wrap points fall mid-entry — a policy citation's file path lands
    alone on the next line at column zero — which is ordinary wrapping in a
    terminal and looks like a broken layout once it is set as an image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = settings or get_settings()
    source = source or (settings.base_dir / "docs" / "example-review.txt")
    lines = source.read_text(encoding="utf-8").rstrip("\n").split("\n")

    char_w, line_h, size = 0.0088, 0.019, 10.5
    margin = 0.03
    width = max(len(line) for line in lines) * char_w + 2 * margin
    height = len(lines) * line_h + 2 * margin

    fig = plt.figure(figsize=(width * 10, height * 10), dpi=140)
    fig.patch.set_facecolor(BACKGROUND)

    for index, line in enumerate(lines):
        y = 1 - margin - (index + 0.8) * (line_h / height)
        for column, text, colour, bold in _segments(line):
            if not text:
                continue
            fig.text(
                margin + column * char_w / width,
                y,
                text,
                family="monospace",
                fontsize=size,
                color=colour,
                fontweight="bold" if bold else "normal",
                va="top",
                ha="left",
            )

    destination = settings.results_dir / "example_review.png"
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor=BACKGROUND)
    plt.close(fig)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca figure", description="Render a figure from a recorded run."
    )
    parser.add_argument("--provider", default="bedrock", help="Which recorded run to render.")
    parser.add_argument("--model", default=None, help="Model id; defaults to the configured one.")
    parser.add_argument(
        "--review", action="store_true", help="Render the captured control review instead."
    )
    args = parser.parse_args(argv)
    print(render_review() if args.review else render(args.provider, args.model))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
