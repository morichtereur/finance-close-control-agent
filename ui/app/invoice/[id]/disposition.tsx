"use client";

import { useEffect, useState } from "react";

import { label } from "@/lib/format";
import type { Tier } from "@/lib/types";

type Action = "approved" | "rejected" | "escalated";

// The interface is a static site with no backend, because authentication is a
// stated non-goal and the honest version of "no auth" is "no endpoint". So a
// disposition recorded here is held in this browser and is explicitly *not* in
// the trace file.
//
// Rather than pretend otherwise, the panel says so and prints the command that
// does append it. That is the truthful version of this control: the reviewer
// makes the decision, and the append-only record is written by something that
// can actually write to it.
const STORAGE_KEY = "i2p-dispositions";

function load(): Record<string, { action: Action; at: string }> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

export function Disposition({
  invoiceId,
  tier,
  proposedAction,
}: {
  invoiceId: string;
  tier: Tier;
  proposedAction: string | null;
}) {
  const [recorded, setRecorded] = useState<{ action: Action; at: string } | null>(null);

  useEffect(() => {
    setRecorded(load()[invoiceId] ?? null);
  }, [invoiceId]);

  function record(action: Action) {
    const entry = { action, at: new Date().toISOString() };
    const all = load();
    all[invoiceId] = entry;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    setRecorded(entry);
  }

  const canApprove = tier !== "escalate";

  return (
    <div className="actions">
      <div className="pane-head">Disposition</div>
      <div className="pane-body">
        <button onClick={() => record("approved")} disabled={!canApprove}>
          Accept{proposedAction ? `: ${label(proposedAction)}` : ""}
        </button>
        <button onClick={() => record("rejected")} disabled={!canApprove}>
          Reject
        </button>
        <button onClick={() => record("escalated")}>Escalate</button>
        {!canApprove && (
          <span className="muted">
            Accept and reject are unavailable: this invoice is routed to escalate, which requires
            investigation rather than approval.
          </span>
        )}
      </div>
      {recorded && (
        <div className="recorded">
          Recorded <code>{recorded.action}</code> at {recorded.at} — in this browser only. It is
          <strong> not </strong> in the append-only trace. To write it there:
          <div className="command">
            fcca review --exception {invoiceId} --action {recorded.action} --reviewer &lt;your
            user id&gt;
          </div>
        </div>
      )}
    </div>
  );
}
