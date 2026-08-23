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

/**
 * Every invoice the export produced.
 *
 * The `INV-00001` shape is asserted rather than assumed. macOS/iCloud sync
 * leaves duplicates beside the originals — `INV-00311 2.json` — and without this
 * they become routes, which fail the static export with a file-not-found on a
 * URL-encoded name. A directory listing is untrusted input like any other.
 */
const INVOICE_ID = /^INV-\d{5}$/;

export function invoiceIds(): string[] {
  return fs
    .readdirSync(path.join(DATA_DIR, "invoices"))
    .filter((name) => name.endsWith(".json"))
    .map((name) => name.replace(/\.json$/, ""))
    .filter((id) => INVOICE_ID.test(id))
    .sort();
}

export function invoiceDetail(invoiceId: string): InvoiceDetail {
  return readJson<InvoiceDetail>("invoices", `${invoiceId}.json`);
}
