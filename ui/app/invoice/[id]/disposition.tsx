"use client";

import { useEffect, useState } from "react";

import { label } from "@/lib/format";
import type { Tier } from "@/lib/types";

type Action = "approved" | "rejected" | "escalated";

/**
 * The decision bar.
 *
 * Sticky, because deciding is the job: a reviewer should never have to scroll
 * back up past the evidence to act on it.
 *
 * There is no backend — authentication is a stated non-goal, and the honest
 * version of "no auth" is "no endpoint". So a disposition recorded here lives in
 * this browser and is explicitly not in the append-only trace. Rather than imply
 * otherwise, the panel says so and prints the command that does append it.
 */
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
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    } catch {
      // Storage can be unavailable; the decision still shows on screen.
    }
    setRecorded(entry);
  }

  const canApprove = tier !== "escalate";

  return (
    <div className="decide">
      <span className="decide-label">Decision</span>
      <button className="primary" onClick={() => record("approved")} disabled={!canApprove}>
        {proposedAction ? `Accept — ${label(proposedAction)}` : "Accept"}
      </button>
      <button onClick={() => record("rejected")} disabled={!canApprove}>
        Reject
      </button>
      <button onClick={() => record("escalated")}>Escalate</button>

      {!canApprove && (
        <span className="decide-note">
          This invoice is routed to escalate. It needs investigation, not approval.
        </span>
      )}

      {recorded && (
        <div className="recorded" style={{ flexBasis: "100%" }}>
          Recorded <strong>{recorded.action}</strong> at {recorded.at}, in this browser only. It is
          not in the append-only trace. To write it there:
          <div className="command">
            fcca review --exception {invoiceId} --action {recorded.action} --reviewer &lt;user&gt;
          </div>
        </div>
      )}
    </div>
  );
}
