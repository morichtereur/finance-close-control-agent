import type { FieldMapping, PostingPayload } from "@/lib/types";

/**
 * The payload that would post.
 *
 * Shown only for auto_clear invoices, which is the point: this is the one place
 * in the system where something happens to an invoice that no person is going to
 * look at, so it is the one place where the mapping needs to be visible without
 * anyone asking for it.
 *
 * The diff column is what makes it reviewable. A value that reaches the ledger
 * unchanged needs no attention; one that was transformed on the way — a date
 * reserialised, an amount stringified, a GL account derived rather than stated —
 * is a decision this code made, and a reviewer who knows the target system is
 * the only person who can say it made it correctly.
 */
export function Posting({ posting }: { posting: PostingPayload }) {
  const transformed = posting.mapping.filter(
    (entry) => entry.source_value !== null && entry.value !== entry.source_value,
  );

  return (
    <section className="panel">
      <h2 className="panel-head">
        Payload that would post
        <span className="of">
          {posting.service} · dry run
        </span>
      </h2>

      <p className="note" style={{ marginTop: 0 }}>
        Built because this invoice routed <strong>auto_clear</strong>. It is not sent: the
        adapter has no transport, and <code className="mono">dry_run</code> is a read-only
        property rather than a setting. Key <span className="mono">{posting.posting_key}</span>,
        claimed once — a second sighting of the same vendor, document number and fiscal year is
        a duplicate.
      </p>

      <h3 className="eyebrow">Mapping</h3>
      <div className="table-scroll">
        <table className="compare">
          <thead>
            <tr>
              <th>Target field</th>
              <th>From</th>
              <th style={{ textAlign: "right" }}>On the document</th>
              <th style={{ textAlign: "right" }}>Posted as</th>
            </tr>
          </thead>
          <tbody>
            {posting.mapping.map((entry) => (
              <MappingRow key={entry.target_field} entry={entry} />
            ))}
          </tbody>
        </table>
      </div>

      <p className="fine" style={{ marginTop: "var(--s2)" }}>
        {transformed.length === 0
          ? "No value changed on the way to the payload."
          : `${transformed.length} value(s) differ from the document — reserialised for the target schema, or derived because the document did not state them.`}
      </p>
    </section>
  );
}

function MappingRow({ entry }: { entry: FieldMapping }) {
  const changed = entry.source_value !== null && entry.value !== entry.source_value;
  return (
    <tr>
      <td className="mono">{entry.target_field}</td>
      <td className="mono dim">
        {entry.source_path ?? <span title={entry.note}>derived</span>}
      </td>
      <td className="num dim">
        {entry.source_value === null ? "—" : String(entry.source_value)}
      </td>
      <td className={changed ? "num breach" : "num"}>{String(entry.value)}</td>
    </tr>
  );
}
