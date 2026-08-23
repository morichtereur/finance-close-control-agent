import { label } from "@/lib/format";
import type { TraceRecord } from "@/lib/types";

/**
 * The trace, verbatim and in order.
 *
 * The actor column is the reason it exists, so it is the thing the styling
 * encodes: a step a model took is set in full-strength ink on a tinted row, and
 * the deterministic steps recede. A reviewer scanning fourteen rows should be
 * able to see the one that was inferred without reading any of them.
 */
export function Trace({ records }: { records: TraceRecord[] }) {
  if (records.length === 0) {
    return <p className="note">No trace records for this case.</p>;
  }

  return (
    <div className="table-scroll">
      <table className="trace">
        <tbody>
          {records.map((record, index) => (
            <tr key={`${record.step_name}-${index}`} className={`by-${record.actor}`}>
              <td className="t-step">{record.step_name}</td>
              <td className="t-actor">{record.actor}</td>
              <td className="t-hash">
                {record.rule_id ?? `${record.model}@${record.prompt_version}`}
              </td>
              <td className="t-outcome">{label(record.outcome)}</td>
              <td className="t-hash">{record.input_hash}</td>
              <td className="t-summary">{record.summary}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
