import { evaluation } from "@/lib/data";
import { label } from "@/lib/format";

export default function EvaluationPage() {
  const report = evaluation();

  return (
    <>
      <a className="back" href="/">
        &larr; queue
      </a>

      <h2>Measured over the labelled population</h2>
      <p className="muted" style={{ maxWidth: "68ch" }}>
        The data is synthetic and the labels are ground truth by construction — each invoice was
        generated from a named scenario whose expected outcome was known before the engine saw it.
        These are therefore real measurements of this pipeline over a known population, and not
        evidence about how it would behave on a real ledger. Every class scoring 1.000 measures
        pipeline integrity, not difficulty.
      </p>

      <h3>Routing</h3>
      <table style={{ maxWidth: "520px" }}>
        <tbody>
          {["auto_clear", "propose_and_approve", "escalate"].map((tier) => (
            <tr key={tier}>
              <td>{label(tier)}</td>
              <td className="num">{report.tier_counts[tier] ?? 0}</td>
              <td className="num dim">
                {(((report.tier_counts[tier] ?? 0) / report.invoices) * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
          <tr>
            <td>touchless rate</td>
            <td className="num">{report.touchless_rate.toFixed(3)}</td>
            <td />
          </tr>
          <tr>
            <td>false auto-post count</td>
            <td className="num">{report.false_auto_post_count}</td>
            <td className="dim">required to be zero</td>
          </tr>
        </tbody>
      </table>

      <h3>Per class</h3>
      <table style={{ maxWidth: "820px" }}>
        <thead>
          <tr>
            <th>Class</th>
            <th style={{ textAlign: "right" }}>Support</th>
            <th style={{ textAlign: "right" }}>TP</th>
            <th style={{ textAlign: "right" }}>FP</th>
            <th style={{ textAlign: "right" }}>FN</th>
            <th style={{ textAlign: "right" }}>Precision</th>
            <th style={{ textAlign: "right" }}>Recall</th>
          </tr>
        </thead>
        <tbody>
          {report.per_class.map((metrics) => (
            <tr key={metrics.exception_type}>
              <td>{label(metrics.exception_type)}</td>
              <td className="num">{metrics.support}</td>
              <td className="num">{metrics.true_positives}</td>
              <td className="num">{metrics.false_positives}</td>
              <td className="num">{metrics.false_negatives}</td>
              <td className="num">{metrics.is_absent ? "n/a" : metrics.precision.toFixed(3)}</td>
              <td className="num">{metrics.is_absent ? "n/a" : metrics.recall.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="dim" style={{ maxWidth: "68ch" }}>
        n/a marks a class with no instances and no predictions — an absence, not a failure. The
        engine can raise quantity_variance; the specified scenario set does not seed it.
      </p>

      <h3>Thresholds in force for this run</h3>
      <table style={{ maxWidth: "520px" }}>
        <tbody>
          {Object.entries(report.settings_snapshot).map(([key, value]) => (
            <tr key={key}>
              <td>{label(key)}</td>
              <td className="num">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
