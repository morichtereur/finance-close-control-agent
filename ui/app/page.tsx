import { evaluation, queue } from "@/lib/data";
import { QueueTable } from "./queue-table";

export default function QueuePage() {
  const rows = queue();
  const report = evaluation();

  return (
    <>
      <dl className="stats">
        <div className="stat">
          <dt>Invoices</dt>
          <dd>{report.invoices}</dd>
        </div>
        <div className="stat">
          <dt>Touchless</dt>
          <dd>{(report.touchless_rate * 100).toFixed(1)}%</dd>
        </div>
        <div className="stat">
          <dt>Exceptions</dt>
          <dd>{report.invoices_with_findings}</dd>
        </div>
        <div className="stat">
          <dt>Model calls</dt>
          <dd>{report.model_calls}</dd>
        </div>
        <div className="stat">
          <dt>False auto-post</dt>
          <dd className="is-zero">{report.false_auto_post_count}</dd>
        </div>
      </dl>
      <QueueTable rows={rows} />
    </>
  );
}
