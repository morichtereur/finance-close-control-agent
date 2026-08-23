import { label } from "@/lib/format";
import type { TraceRecord } from "@/lib/types";

// The trace is rendered verbatim and in order. The actor column is the reason
// it exists: a reviewer must be able to tell at a glance which lines were
// computed and which were inferred.
export function Trace({ records }: { records: TraceRecord[] }) {
  if (records.length === 0) {
    return <p className="muted" style={{ padding: 10 }}>No trace records.</p>;
  }
  return (
    <table className="trace">
      <tbody>
        {records.map((record, index) => (
          <tr key={`${record.step_name}-${index}`}>
            <td className="step">{record.step_name}</td>
            <td className={`actor-${record.actor}`}>{record.actor}</td>
            <td className="dim">{record.rule_id ?? `${record.model}@${record.prompt_version}`}</td>
            <td className="outcome">{label(record.outcome)}</td>
            <td className="dim">{record.input_hash}</td>
            <td>{record.summary}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
