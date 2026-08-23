"""Evaluation metrics for the invoice-to-pay module.

The data is synthetic and the labels are ground truth by construction, so the
numbers here are real measurements of a real pipeline over a known population —
and they are *not* evidence about how the system would behave on a real accounts
payable ledger. Both halves of that sentence matter, and the README states them
together rather than reporting the first and omitting the second.

What is measured:

**Touchless rate.** The share of invoices cleared without a person. Defined as
``tier == auto_clear`` and nothing else, so it cannot drift into meaning
"required little effort".

**Exception rate by type.** How often each class occurs, predicted and actual.
A touchless rate is uninterpretable without it: a system that flags nothing has
a wonderful touchless rate.

**Confusion matrix.** Predicted class against actual class over the closed
vocabulary. Full matrix rather than a headline accuracy, because the interesting
question is never "how often is it right" but "what does it confuse with what".

**Precision and recall per class.** Reported per class rather than averaged,
because the classes have different consequences and an average hides the one
that matters.

**False auto-post count.** Invoices routed to ``auto_clear`` that ground truth
says were exceptions. This is not a metric with a target — it is a test, and
:mod:`tests.test_i2p_evaluation` fails the build if it is anything but zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from fcca.i2p.models import ExceptionType

if TYPE_CHECKING:
    from fcca.i2p.resolver import ResolvedInvoice

#: The full class vocabulary, fixed so that the matrix has stable axes even when
#: a class does not occur in a particular run.
CLASSES: tuple[ExceptionType, ...] = (
    "no_exception",
    "bank_details_mismatch",
    "duplicate_invoice",
    "price_variance",
    "quantity_variance",
    "missing_or_delayed_goods_receipt",
    "cost_center_missing",
    "gl_account_missing",
)


class ClassMetrics(BaseModel):
    """Precision, recall and support for one exception class."""

    model_config = ConfigDict(frozen=True)

    exception_type: str
    support: int = Field(description="How many invoices actually are of this class.")
    predicted: int = Field(description="How many the system assigned to this class.")
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Of the ones we called this, how many were. Undefined as 0.0 if none."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Of the ones that were, how many we caught."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def is_absent(self) -> bool:
        """No instances and no predictions.

        Precision and recall are undefined here, not zero. Reporting 0.000
        would read as "the system failed at this", when what happened is that
        the class never came up. quantity_variance is the live example: the
        engine can raise it, and the specified scenario set does not seed it,
        so the honest report is an absence.
        """
        return self.support == 0 and self.predicted == 0


class EvaluationReport(BaseModel):
    """Everything one evaluation run measured."""

    provider: str
    model: str
    invoices: int

    # ---- routing
    tier_counts: dict[str, int]
    touchless_rate: float

    # ---- the safety property
    false_auto_post_count: int = Field(
        description="Exceptions routed to auto_clear. Required to be zero."
    )
    false_auto_post_ids: list[str]

    # ---- classification
    confusion: dict[str, dict[str, int]] = Field(
        description="confusion[actual][predicted] = count."
    )
    per_class: list[ClassMetrics]
    actual_counts: dict[str, int]
    predicted_counts: dict[str, int]

    # ---- extraction
    extraction_gated: int = Field(
        default=0,
        description=(
            "Invoices escalated because a load-bearing field was read too weakly to "
            "compute on. Zero on synthetic data, where there is no document and so no "
            "confidence to gate."
        ),
    )

    # ---- model usage
    model_calls: int
    invoices_with_findings: int
    ungrounded_citation_invoices: int
    mean_confidence: float | None

    # ---- configuration in force, so an old report can be reread correctly
    settings_snapshot: dict[str, float | int]

    @property
    def exact_agreement(self) -> float:
        correct = sum(self.confusion[c][c] for c in self.confusion)
        return correct / self.invoices if self.invoices else 0.0

    @property
    def is_safe(self) -> bool:
        return self.false_auto_post_count == 0

    @property
    def ungrounded_citation_rate(self) -> float:
        """Ungrounded citations as a share of the invoices a model actually saw."""
        if not self.invoices_with_findings:
            return 0.0
        return self.ungrounded_citation_invoices / self.invoices_with_findings

    @property
    def is_grounded(self) -> bool:
        """Whether citation grounding stayed inside the configured limit.

        Separate from ``is_safe`` on purpose. A false auto-post is money out of the
        door; an ungrounded citation was removed before anything used it. Both fail
        the run, but conflating them would let a reader assume the wrong severity.
        """
        limit = self.settings_snapshot.get("max_ungrounded_citation_rate")
        if limit is None:  # report written before the limit existed
            return True
        return self.ungrounded_citation_rate <= float(limit)


def evaluate(
    resolved: list[ResolvedInvoice],
    labels: dict[str, str],
    provider: str,
    model: str,
    settings_snapshot: dict[str, float | int],
) -> EvaluationReport:
    """Score a completed run against ground truth."""
    confusion: dict[str, dict[str, int]] = {actual: dict.fromkeys(CLASSES, 0) for actual in CLASSES}
    tier_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    false_auto_post: list[str] = []
    confidences: list[float] = []
    model_calls = 0
    with_findings = 0
    gated = 0
    ungrounded = 0

    for item in resolved:
        actual = labels[item.invoice_id]
        predicted = item.result.primary_exception

        confusion[actual][predicted] += 1
        actual_counts[actual] += 1
        predicted_counts[predicted] += 1
        tier_counts[item.routing.tier] += 1

        if item.result.is_exception:
            with_findings += 1
        if item.result.extraction_gated:
            gated += 1
        if item.model_called:
            model_calls += 1
        if item.assessment is not None:
            confidences.append(item.assessment.assessment.confidence)
            if item.assessment.grounding.ungrounded_citations:
                ungrounded += 1

        # The safety property: cleared without a person, when it should not have been.
        if item.touchless and actual != "no_exception":
            false_auto_post.append(item.invoice_id)

    per_class = []
    for exception_type in CLASSES:
        true_positives = confusion[exception_type][exception_type]
        false_positives = sum(
            confusion[other][exception_type] for other in CLASSES if other != exception_type
        )
        false_negatives = sum(
            confusion[exception_type][other] for other in CLASSES if other != exception_type
        )
        per_class.append(
            ClassMetrics(
                exception_type=exception_type,
                support=actual_counts[exception_type],
                predicted=predicted_counts[exception_type],
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )
        )

    total = len(resolved)
    return EvaluationReport(
        provider=provider,
        model=model,
        invoices=total,
        tier_counts=dict(tier_counts),
        touchless_rate=(tier_counts["auto_clear"] / total) if total else 0.0,
        false_auto_post_count=len(false_auto_post),
        false_auto_post_ids=sorted(false_auto_post),
        confusion=confusion,
        per_class=per_class,
        actual_counts=dict(actual_counts),
        predicted_counts=dict(predicted_counts),
        extraction_gated=gated,
        model_calls=model_calls,
        invoices_with_findings=with_findings,
        ungrounded_citation_invoices=ungrounded,
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        settings_snapshot=settings_snapshot,
    )


def render_report(report: EvaluationReport) -> str:
    """Fixed-width plain text. Used by the CLI and pasted into the README."""
    lines: list[str] = []
    lines.append(f"provider           {report.provider}:{report.model}")
    lines.append(f"invoices           {report.invoices}")
    lines.append("")

    lines.append("Routing")
    for tier in ("auto_clear", "propose_and_approve", "escalate"):
        count = report.tier_counts.get(tier, 0)
        share = count / report.invoices if report.invoices else 0.0
        lines.append(f"  {tier:<22} {count:>5}   {share:>6.1%}")
    lines.append(f"  {'touchless rate':<22} {report.touchless_rate:>13.3f}")
    lines.append("")

    lines.append("Safety")
    lines.append(f"  false auto-post count  {report.false_auto_post_count}")
    if report.false_auto_post_ids:
        lines.append(f"  offending invoices     {', '.join(report.false_auto_post_ids)}")
    lines.append("")

    lines.append("Exception rate by type (actual vs predicted)")
    width = max(len(c) for c in CLASSES)
    for exception_type in CLASSES:
        actual_count = report.actual_counts.get(exception_type, 0)
        predicted_count = report.predicted_counts.get(exception_type, 0)
        rate = actual_count / report.invoices if report.invoices else 0.0
        lines.append(
            f"  {exception_type.ljust(width)}  actual {actual_count:>4} ({rate:>5.1%})   "
            f"predicted {predicted_count:>4}"
        )
    lines.append("")

    lines.append("Per-class precision / recall")
    lines.append(
        f"  {'class'.ljust(width)}  {'supp':>5} {'tp':>4} {'fp':>4} {'fn':>4} "
        f"{'prec':>6} {'rec':>6} {'f1':>6}"
    )
    for metrics in report.per_class:
        if metrics.is_absent:
            # Printing 0.000 for a class with no instances and no predictions
            # would read as a failure. It is an absence, and the two are not the
            # same thing — see ClassMetrics.is_absent.
            scores = f"{'n/a':>6} {'n/a':>6} {'n/a':>6}"
        else:
            scores = f"{metrics.precision:>6.3f} {metrics.recall:>6.3f} {metrics.f1:>6.3f}"
        lines.append(
            f"  {metrics.exception_type.ljust(width)}  {metrics.support:>5} "
            f"{metrics.true_positives:>4} {metrics.false_positives:>4} "
            f"{metrics.false_negatives:>4} {scores}"
        )
    lines.append(
        "  n/a marks a class with no instances and no predictions: an absence, not a failure."
    )
    lines.append("")

    lines.append("Confusion matrix (rows actual, columns predicted)")
    header = "  " + "".ljust(width) + "".join(f" {c[:9]:>10}" for c in CLASSES)
    lines.append(header)
    for actual in CLASSES:
        row = "".join(f" {report.confusion[actual][p]:>10}" for p in CLASSES)
        lines.append(f"  {actual.ljust(width)}{row}")
    lines.append("")

    lines.append("Model usage")
    lines.append(f"  invoices with a finding      {report.invoices_with_findings}")
    lines.append(f"  model calls                  {report.model_calls}")
    limit = report.settings_snapshot.get("max_ungrounded_citation_rate")
    grounding = f"  invoices citing missing data {report.ungrounded_citation_invoices}"
    if report.invoices_with_findings:
        grounding += f" ({report.ungrounded_citation_rate:.1%} of assessed"
        grounding += f", limit {float(limit):.0%})" if limit is not None else ")"
    lines.append(grounding)
    if report.mean_confidence is not None:
        lines.append(f"  mean confidence              {report.mean_confidence:.3f}")
    return "\n".join(lines)


def confusion_as_rows(report: EvaluationReport) -> list[dict[str, object]]:
    """Long-form confusion matrix, for the UI and for CSV export."""
    rows: list[dict[str, object]] = []
    for actual, predictions in report.confusion.items():
        for predicted, count in predictions.items():
            if count:
                rows.append({"actual": actual, "predicted": predicted, "count": count})
    return rows


def exception_rate_by_type(report: EvaluationReport) -> dict[str, float]:
    counts: defaultdict[str, float] = defaultdict(float)
    for exception_type, count in report.actual_counts.items():
        counts[exception_type] = count / report.invoices if report.invoices else 0.0
    return dict(counts)


__all__ = [
    "CLASSES",
    "ClassMetrics",
    "EvaluationReport",
    "confusion_as_rows",
    "evaluate",
    "exception_rate_by_type",
    "render_report",
]
