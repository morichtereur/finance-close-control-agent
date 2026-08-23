// Data access. Everything is read from disk at build time — there is no fetch,
// no API route and no server. `fcca i2p-export` writes ui/data; this reads it.

import fs from "node:fs";
import path from "node:path";

import type { Evaluation, InvoiceDetail, QueueRow } from "./types";

const DATA_DIR = path.join(process.cwd(), "data");

function readJson<T>(...segments: string[]): T {
  return JSON.parse(fs.readFileSync(path.join(DATA_DIR, ...segments), "utf-8")) as T;
}

export function queue(): QueueRow[] {
  return readJson<QueueRow[]>("queue.json");
}

export function evaluation(): Evaluation {
  return readJson<Evaluation>("evaluation.json");
}

export function invoiceIds(): string[] {
  return fs
    .readdirSync(path.join(DATA_DIR, "invoices"))
    .filter((name) => name.endsWith(".json"))
    .map((name) => name.replace(/\.json$/, ""));
}

export function invoiceDetail(invoiceId: string): InvoiceDetail {
  return readJson<InvoiceDetail>("invoices", `${invoiceId}.json`);
}
