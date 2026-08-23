import { evaluation, queue } from "@/lib/data";
import { QueueTable } from "./queue-table";

export default function QueuePage() {
  const rows = queue();
  const report = evaluation();

  return (
    <>
      <div className="summary">
        <div>
          <span className="k">invoices</span>
          <span className="v">{report.invoices}</span>
        </div>
        <div>
          <span className="k">touchless</span>
          <span className="v">{(report.touchless_rate * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="k">exceptions</span>
          <span className="v">{report.invoices_with_findings}</span>
        </div>
        <div>
          <span className="k">model calls</span>
          <span className="v">{report.model_calls}</span>
        </div>
        <div>
          <span className="k">false auto-post</span>
          <span className="v">{report.false_auto_post_count}</span>
        </div>
        <div>
          <span className="k">provider</span>
          <span className="v">
            {report.provider}:{report.model}
          </span>
        </div>
      </div>
      <QueueTable rows={rows} />
    </>
  );
}
