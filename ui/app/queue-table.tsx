"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { label, money } from "@/lib/format";
import type { QueueRow, Tier } from "@/lib/types";

const TIERS: Tier[] = ["escalate", "propose_and_approve", "auto_clear"];

// Escalations first. A queue sorted by date is a queue where the thing that
// needed attention today is on page four.
const TIER_ORDER: Record<Tier, number> = {
  escalate: 0,
  propose_and_approve: 1,
  auto_clear: 2,
};

export function QueueTable({ rows }: { rows: QueueRow[] }) {
  const [tiers, setTiers] = useState<Set<Tier>>(new Set(TIERS));
  const [types, setTypes] = useState<Set<string>>(new Set());

  const allTypes = useMemo(
    () => Array.from(new Set(rows.map((row) => row.exception_type))).sort(),
    [rows],
  );

  const visible = useMemo(() => {
    return rows
      .filter((row) => tiers.has(row.tier))
      .filter((row) => types.size === 0 || types.has(row.exception_type))
      .sort(
        (a, b) =>
          TIER_ORDER[a.tier] - TIER_ORDER[b.tier] ||
          b.document_value - a.document_value ||
          a.invoice_id.localeCompare(b.invoice_id),
      );
  }, [rows, tiers, types]);

  function toggle<T>(set: Set<T>, value: T, apply: (next: Set<T>) => void) {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    apply(next);
  }

  return (
    <>
      <div className="filters">
        <label>Tier</label>
        {TIERS.map((tier) => (
          <button
            key={tier}
            className="chip"
            aria-pressed={tiers.has(tier)}
            onClick={() => toggle(tiers, tier, setTiers)}
          >
            {label(tier)}
          </button>
        ))}
      </div>
      <div className="filters">
        <label>Exception</label>
        {allTypes.map((type) => (
          <button
            key={type}
            className="chip"
            aria-pressed={types.has(type)}
            onClick={() => toggle(types, type, setTypes)}
          >
            {label(type)}
          </button>
        ))}
        {types.size > 0 && (
          <button className="chip" onClick={() => setTypes(new Set())}>
            clear
          </button>
        )}
      </div>

      <p className="muted" style={{ margin: "0 0 8px" }}>
        {visible.length} of {rows.length} invoices
      </p>

      <table>
        <thead>
          <tr>
            <th>Invoice</th>
            <th>Vendor</th>
            <th>Ref</th>
            <th>Type</th>
            <th>Date</th>
            <th style={{ textAlign: "right" }}>Value</th>
            <th style={{ textAlign: "right" }}>Lines</th>
            <th>Exception</th>
            <th style={{ textAlign: "right" }}>Conf</th>
            <th>Tier</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => (
            <tr key={row.invoice_id}>
              <td className="mono">
                <Link href={`/invoice/${row.invoice_id}/`}>{row.invoice_id}</Link>
              </td>
              <td>{row.vendor_name}</td>
              <td className="mono dim">{row.vendor_reference}</td>
              <td className="mono dim">{row.category}</td>
              <td className="mono dim">{row.invoice_date}</td>
              <td className="num">{money(row.document_value, row.currency)}</td>
              <td className="num dim">{row.lines}</td>
              <td className={row.severity ? `sev sev-${row.severity}` : "sev"}>
                {row.exception_type === "no_exception" ? (
                  <span className="dim">—</span>
                ) : (
                  label(row.exception_type)
                )}
              </td>
              <td className="num dim">
                {row.confidence === null ? "—" : row.confidence.toFixed(2)}
              </td>
              <td className={`tier tier-${row.tier}`}>{label(row.tier)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
