"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { label, money } from "@/lib/format";
import type { QueueRow, Tier } from "@/lib/types";

const TIERS: Tier[] = ["escalate", "propose_and_approve", "auto_clear"];

// Escalations first. A queue ordered by date puts the thing that needed
// attention today on page four.
const TIER_ORDER: Record<Tier, number> = {
  escalate: 0,
  propose_and_approve: 1,
  auto_clear: 2,
};

export function QueueTable({ rows }: { rows: QueueRow[] }) {
  const router = useRouter();
  const [tiers, setTiers] = useState<Set<Tier>>(new Set(TIERS));
  const [types, setTypes] = useState<Set<string>>(new Set());
  const [cursor, setCursor] = useState(0);
  const bodyRef = useRef<HTMLTableSectionElement>(null);

  const allTypes = useMemo(
    () => Array.from(new Set(rows.map((row) => row.exception_type))).sort(),
    [rows],
  );

  const visible = useMemo(
    () =>
      rows
        .filter((row) => tiers.has(row.tier))
        .filter((row) => types.size === 0 || types.has(row.exception_type))
        .sort(
          (a, b) =>
            TIER_ORDER[a.tier] - TIER_ORDER[b.tier] ||
            b.document_value - a.document_value ||
            a.invoice_id.localeCompare(b.invoice_id),
        ),
    [rows, tiers, types],
  );

  useEffect(() => {
    setCursor(0);
  }, [tiers, types]);

  // A queue is worked one item at a time, so it is worth being able to move
  // through it without leaving the keyboard.
  const onKey = useCallback(
    (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      if (event.key === "j" || event.key === "ArrowDown") {
        event.preventDefault();
        setCursor((c) => Math.min(c + 1, visible.length - 1));
      } else if (event.key === "k" || event.key === "ArrowUp") {
        event.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
      } else if (event.key === "Enter" && visible[cursor]) {
        event.preventDefault();
        router.push(`/invoice/${visible[cursor].invoice_id}/`);
      }
    },
    [cursor, router, visible],
  );

  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);

  useEffect(() => {
    bodyRef.current
      ?.querySelector('tr[aria-current="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  function toggle<T>(set: Set<T>, value: T, apply: (next: Set<T>) => void) {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    apply(next);
  }

  const shares = TIERS.map((tier) => ({
    tier,
    n: visible.filter((row) => row.tier === tier).length,
  }));

  return (
    <>
      {/* The shape of the queue, drawn to scale. Width is share of the
          population, so the balance between the three tiers is legible before
          any number is read. */}
      <div className="ribbon" role="img" aria-label={
        shares.map((s) => `${label(s.tier)} ${s.n}`).join(", ")
      }>
        {shares.map(({ tier, n }) => (
          <span
            key={tier}
            className={`r-${tier}`}
            style={{ width: `${visible.length ? (n / visible.length) * 100 : 0}%` }}
          />
        ))}
      </div>
      <div className="ribbon-key">
        {shares.map(({ tier, n }) => (
          <span key={tier}>
            {label(tier)} {n}
          </span>
        ))}
      </div>

      <div className="filters">
        <span className="filter-label">Tier</span>
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
        <span className="filter-label">Exception</span>
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
          <button className="chip chip-clear" onClick={() => setTypes(new Set())}>
            clear
          </button>
        )}
      </div>

      <p className="count">
        {visible.length} of {rows.length} invoices
      </p>
      <p className="hint">
        <kbd>j</kbd> <kbd>k</kbd> move · <kbd>enter</kbd> open
      </p>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Invoice</th>
              <th>Vendor</th>
              <th>Ref</th>
              <th>Date</th>
              <th style={{ textAlign: "right" }}>Value</th>
              <th style={{ textAlign: "right" }}>Lines</th>
              <th>Exception</th>
              <th style={{ textAlign: "right" }}>Conf</th>
              <th>Tier</th>
            </tr>
          </thead>
          <tbody ref={bodyRef}>
            {visible.map((row, index) => (
              <tr
                key={row.invoice_id}
                aria-current={index === cursor ? "true" : undefined}
                onMouseEnter={() => setCursor(index)}
              >
                <td className="mono">
                  <Link href={`/invoice/${row.invoice_id}/`}>{row.invoice_id}</Link>
                </td>
                <td>{row.vendor_name}</td>
                <td className="mono dim">{row.vendor_reference}</td>
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
      </div>

      {visible.length === 0 && (
        <p className="note">
          No invoices match these filters. Turn a tier back on, or clear the exception filter.
        </p>
      )}
    </>
  );
}
