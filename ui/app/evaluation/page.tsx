import Link from "next/link";

import { evaluation } from "@/lib/data";
import { label } from "@/lib/format";

export default function EvaluationPage() {
  const report = evaluation();

  return (
    <>
      <Link className="back" href="/">
        ← Queue
      </Link>

      <header className="case">
        <div className="case-top">
          <div>
            <div className="case-id">Evaluation</div>
            <div className="case-vendor">
              {report.invoices} invoices · {report.provider}:{report.model}
            </div>
          </div>
        </div>
      </header>

      <p className="note" style={{ marginBottom: "var(--s5)" }}>
        The data is synthetic and the labels are ground truth by construction — each invoice was
        generated from a named scenario whose expected outcome was known before the engine saw it.
        These are real measurements of this pipeline over a known population, and not evidence
        about how it would behave on a real ledger. Every class scoring 1.000 measures pipeline
        integrity, not difficulty.
      </p>

      <div className="split">
        <section className="panel">
          <h2 className="panel-head">Routing</h2>
          <table className="compare">
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
              <tr className="is-total">
                <td>Touchless rate</td>
                <td className="num">{report.touchless_rate.toFixed(3)}</td>
                <td />
              </tr>
              <tr className="is-total">
                <td>False auto-post count</td>
                <td className="num">{report.false_auto_post_count}</td>
                <td className="dim">must be zero</td>
              </tr>
            </tbody>
          </table>

          <h2 className="panel-head" style={{ marginTop: "var(--s5)" }}>
            Thresholds in force
          </h2>
          <table className="compare">
            <tbody>
              {Object.entries(report.settings_snapshot).map(([key, value]) => (
                <tr key={key}>
                  <td className="muted">{label(key)}</td>
                  <td className="num">{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="panel">
          <h2 className="panel-head">
            Per class
            <span className="of">precision · recall</span>
          </h2>
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th style={{ textAlign: "right" }}>Supp</th>
                <th style={{ textAlign: "right" }}>TP</th>
                <th style={{ textAlign: "right" }}>FP</th>
                <th style={{ textAlign: "right" }}>FN</th>
                <th style={{ textAlign: "right" }}>Prec</th>
                <th style={{ textAlign: "right" }}>Rec</th>
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
                  <td className="num">
                    {metrics.is_absent ? <span className="dim">n/a</span> : metrics.precision.toFixed(3)}
                  </td>
                  <td className="num">
                    {metrics.is_absent ? <span className="dim">n/a</span> : metrics.recall.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="fine" style={{ marginTop: "var(--s2)" }}>
            n/a marks a class with no instances and no predictions — an absence, not a failure. The
            engine can raise quantity variance; the specified scenario set does not seed it.
          </p>
        </section>
      </div>
    </>
  );
}
