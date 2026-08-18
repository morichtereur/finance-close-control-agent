"""Render the escalation outcome of a benchmark run as a single stacked bar.

One chart, one claim: of 60 labelled exceptions, how many reached a reviewer
who needed to see them, how many reached one who did not, and how many that
needed review were cleared instead. The last number is the one that matters and
it is zero, which is difficult to show — so it is annotated rather than drawn.

Reads `results/eval_<provider>.json`, which the benchmark writes, so the figure
cannot drift from the numbers it illustrates. Regenerate with:

    python -m fcca.figure --provider bedrock

Requires the optional `figures` extra; nothing else in the package imports
matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcca.config import Settings, get_settings

#: Taken from the portfolio site's design tokens so the figure sits in the page
#: rather than on it.
BACKGROUND = "#e9eae4"
INK = "#121a17"
MUTED = "#59635e"
ACCENT = "#146b54"
ACCENT_SOFT = "#8fb3a6"
WARN = "#b8763a"
RULE = "#7e857a"


def render(provider: str = "bedrock", settings: Settings | None = None) -> Path:
    """Write the escalation-outcome bar for one provider's recorded run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = settings or get_settings()
    source = settings.results_dir / f"eval_{provider}.json"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found — run `fcca evaluate --provider {provider}`")

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

    destination = settings.results_dir / f"escalation_outcomes_{provider}.png"
    fig.savefig(destination, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fcca figure", description="Render the escalation outcome of a benchmark run."
    )
    parser.add_argument("--provider", default="bedrock", help="Which recorded run to render.")
    args = parser.parse_args(argv)
    print(render(args.provider))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
