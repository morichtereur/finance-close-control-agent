import Link from "next/link";

import { invoiceDetail, invoiceIds } from "@/lib/data";
import {
  discountSchedule,
  label,
  money,
  percent,
  quantity as fmtQuantity,
  signed,
  unitPrice,
} from "@/lib/format";
import type {
  FieldProvenance,
  InvoiceDetail,
  LineResolution,
  PurchaseOrderLine,
} from "@/lib/types";

import { ToleranceBridge } from "./bridge";
import { Disposition } from "./disposition";
import { Posting } from "./posting";
import { Trace } from "./trace";

export function generateStaticParams() {
  return invoiceIds().map((id) => ({ id }));
}

export default async function InvoicePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const detail = invoiceDetail(id);

  // The page is ordered by what the reviewer needs, not by what the pipeline
  // produced: the verdict and the reason for it first, then the one figure that
  // drives it, then the evidence, then the record of how it was reached.
  const priced = detail.result.resolutions.filter((r) => r.price !== null);
  const worst = priced.reduce<LineResolution | null>((acc, r) => {
    if (!acc) return r;
    return Math.abs(r.price!.residual_pct) > Math.abs(acc.price!.residual_pct) ? r : acc;
  }, null);

  return (
    <>
      <Link className="back" href="/">
        ← Queue
      </Link>

      <CaseHeader detail={detail} />

      {worst?.price && (
        <ToleranceBridge price={worst.price} currency={detail.result.currency} />
      )}

      <div className="split">
        <SourceDocument detail={detail} />
        <div>
          <Comparison detail={detail} />
          <Findings detail={detail} />
          <Reasoning detail={detail} />
          {detail.posting && <Posting posting={detail.posting} />}
        </div>
      </div>

      <section className="panel" style={{ marginTop: "var(--s5)" }}>
        <h2 className="panel-head">
          Trace
          <span className="of">
            {detail.trace.length} steps ·{" "}
            {detail.trace.filter((r) => r.actor === "model").length} by model
          </span>
        </h2>
        <Trace records={detail.trace} />
      </section>

      <Disposition
        invoiceId={detail.invoice.invoice_id}
        tier={detail.routing.tier}
        proposedAction={detail.assessment?.assessment.proposed_action ?? null}
      />
    </>
  );
}

/* ------------------------------------------------------------ case header */

/**
 * Provenance, rendered beside the value rather than in a separate panel.
 *
 * A reviewer asking "was that on the document or did we work it out?" is asking
 * about one field, at the moment they are looking at it. Answering in a panel
 * somewhere else means the question mostly goes unasked. Synthetic is not
 * labelled: on this dataset it is every field, and a label on everything labels
 * nothing.
 */
function Prov({ p }: { p: FieldProvenance | undefined }) {
  if (!p || p.source === "synthetic") return null;
  const weak = p.confidence !== null && p.confidence < 0.8;
  return (
    <span className={weak ? "prov prov-weak" : "prov"} title={p.engine ?? undefined}>
      {p.source === "extracted"
        ? `read ${p.confidence?.toFixed(2)}`
        : p.source.replace("_", " ")}
    </span>
  );
}

function CaseHeader({ detail }: { detail: InvoiceDetail }) {
  const { invoice, vendor, routing, result } = detail;
  return (
    <header className="case">
      <div className="case-top">
        <div>
          <div className="case-id">{invoice.invoice_id}</div>
          <div className="case-vendor">
            {vendor?.name ?? invoice.vendor_id} · ref {invoice.vendor_reference}
          </div>
        </div>
        <dl className="case-facts">
          <div className="case-fact">
            <dt>Value</dt>
            <dd>{money(result.document_value, result.currency)}</dd>
          </div>
          <div className="case-fact">
            <dt>Type</dt>
            <dd>{result.category}</dd>
          </div>
          <div className="case-fact">
            <dt>Received</dt>
            <dd>{invoice.received_date}</dd>
          </div>
        </dl>
      </div>

      <div className="verdict">
        <span className={`verdict-tier tier-${routing.tier}`}>{label(routing.tier)}</span>
        <span className="verdict-why">{routing.deciding_reason}</span>
      </div>

      {result.extraction_gated && (
        <p className="gate-note">
          <strong>Extraction gate fired.</strong> {result.extraction_gate_reasons.join(", ")} — the
          field was read too weakly to compute on, so this invoice escalated before any model was
          called. Nothing downstream ran on the value.
        </p>
      )}
    </header>
  );
}

/* ------------------------------------------------------------- left panel */

function SourceDocument({ detail }: { detail: InvoiceDetail }) {
  const { invoice, vendor } = detail;
  const prov = detail.result.provenance ?? {};
  const bankDiffers = vendor != null && invoice.stated_bank_iban !== vendor.bank_iban;

  return (
    <div>
      <section className="panel">
        <h2 className="panel-head">
          Document as received
          <span className="of">{invoice.lines.length} lines</span>
        </h2>
        <dl className="fields">
          <dt>Invoice date</dt>
          <dd>{invoice.invoice_date}</dd>
          <dt>Company code</dt>
          <dd>{invoice.company_code}</dd>
          <dt>Net</dt>
          <dd>{money(invoice.stated_total_net, invoice.currency)}</dd>
          <dt>Tax</dt>
          <dd>{money(invoice.stated_total_tax, invoice.currency)}</dd>
          <dt>Gross</dt>
          <dd>
            {money(invoice.stated_total_gross, invoice.currency)}
            <Prov p={prov["stated_total_gross"]} />
          </dd>
          <dt>Bank stated</dt>
          <dd className={bankDiffers ? "breach" : undefined}>
            {invoice.stated_bank_iban}
            <Prov p={prov["stated_bank_iban"]} />
          </dd>
          {vendor && (
            <>
              <dt>Bank on file</dt>
              <dd>{vendor.bank_iban}</dd>
            </>
          )}
        </dl>

        {invoice.free_text && (
          <p className="freetext">
            <span className="src">Free text — written by the sender, treated as untrusted</span>
            {invoice.free_text}
          </p>
        )}
      </section>

      <section className="panel">
        <h2 className="panel-head">Lines as invoiced</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Their item</th>
                <th>Description</th>
                <th style={{ textAlign: "right" }}>Qty</th>
                <th>UoM</th>
                <th style={{ textAlign: "right" }}>List</th>
                <th>Discounts</th>
                <th style={{ textAlign: "right" }}>Tax</th>
              </tr>
            </thead>
            <tbody>
              {invoice.lines.map((line) => (
                <tr key={line.line_no}>
                  <td className="mono dim">{line.line_no}</td>
                  <td className="mono">{line.supplier_item_no}</td>
                  <td>{line.description}</td>
                  <td className="num">
                    {fmtQuantity(line.quantity)}
                    <Prov p={prov[`lines[${line.line_no - 1}].quantity`]} />
                  </td>
                  <td className="mono dim">{line.uom}</td>
                  <td className="num">
                    {line.price.list_price}
                    <Prov p={prov[`lines[${line.line_no - 1}].price.list_price`]} />
                  </td>
                  <td className="mono dim">
                    {discountSchedule(line.price.discount_pct, line.price.surcharge_per_unit)}
                  </td>
                  <td className="num dim">{line.tax_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {Object.keys(detail.purchase_orders).length > 0 && (
        <section className="panel">
          <h2 className="panel-head">Purchase order</h2>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>PO</th>
                  <th>#</th>
                  <th>Material</th>
                  <th style={{ textAlign: "right" }}>Qty</th>
                  <th>UoM</th>
                  <th style={{ textAlign: "right" }}>List</th>
                  <th>Discounts</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(detail.purchase_orders).flatMap((order) =>
                  order.lines.map((line: PurchaseOrderLine) => (
                    <tr key={`${order.po_id}-${line.po_line}`}>
                      <td className="mono dim">{order.po_id}</td>
                      <td className="mono dim">{line.po_line}</td>
                      <td className="mono">{line.material_id}</td>
                      <td className="num">{fmtQuantity(line.quantity)}</td>
                      <td className="mono dim">{line.uom}</td>
                      <td className="num">{line.price.list_price}</td>
                      <td className="mono dim">
                        {discountSchedule(line.price.discount_pct, line.price.surcharge_per_unit)}
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {detail.goods_receipts.length > 0 && (
        <section className="panel">
          <h2 className="panel-head">Goods receipts</h2>
          <table>
            <thead>
              <tr>
                <th>Receipt</th>
                <th>PO line</th>
                <th>Date</th>
                <th style={{ textAlign: "right" }}>Qty</th>
                <th>UoM</th>
              </tr>
            </thead>
            <tbody>
              {detail.goods_receipts.map((receipt) => (
                <tr key={receipt.gr_id}>
                  <td className="mono">{receipt.gr_id}</td>
                  <td className="mono dim">
                    {receipt.po_id} / {receipt.po_line}
                  </td>
                  <td className="mono dim">{receipt.receipt_date}</td>
                  <td className="num">{fmtQuantity(receipt.quantity)}</td>
                  <td className="mono dim">{receipt.uom}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

/* ------------------------------------------------------------ right panel */

function Comparison({ detail }: { detail: InvoiceDetail }) {
  const priced = detail.result.resolutions.filter((r) => r.price !== null);

  return (
    <section className="panel">
      <h2 className="panel-head">
        Comparison
        <span className="of">normalised to one base unit</span>
      </h2>

      {priced.length === 0 ? (
        <p className="note">
          No purchase order to compare against. This is an FI invoice, governed by coding
          completeness and approval rather than by three-way match.
        </p>
      ) : (
        priced.map((resolution) => (
          <LineCompare
            key={resolution.line_no}
            resolution={resolution}
            currency={detail.result.currency}
            multiple={priced.length > 1}
          />
        ))
      )}

      <h3 className="eyebrow" style={{ marginTop: "var(--s4)" }}>
        Coding
      </h3>
      <table className="compare">
        <tbody>
          {detail.result.resolutions.map((resolution) => (
            <tr key={resolution.line_no} className={resolution.gl_source === "derived" ? "is-derived" : undefined}>
              <td className="mono dim">line {resolution.line_no}</td>
              <td className="mono">{resolution.material_id ?? "—"}</td>
              <td className="mono">{resolution.tax_code ?? "—"}</td>
              <td className="mono">
                {resolution.gl_account ?? "—"}{" "}
                <span className="dim">{label(resolution.gl_source)}</span>
              </td>
              <td className="mono">
                {resolution.cost_center ?? "—"}{" "}
                <span className="dim">{label(resolution.cost_center_source)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="fine" style={{ marginTop: "var(--s2)" }}>
        Stated means the vendor supplied it. Derived means a rule computed it from master data.
      </p>
    </section>
  );
}

function LineCompare({
  resolution,
  currency,
  multiple,
}: {
  resolution: LineResolution;
  currency: string;
  multiple: boolean;
}) {
  const price = resolution.price!;
  const breach = !price.within_tolerance;
  const cls = breach ? "num breach" : "num";

  return (
    <div style={{ marginBottom: "var(--s4)" }}>
      {multiple && <h3 className="eyebrow eyebrow-plain">Line {resolution.line_no}</h3>}
      <table className="compare">
        <tbody>
          <tr>
            <td className="muted">Purchase order, per base unit</td>
            <td className="num">{unitPrice(price.po_unit_price_normalised)}</td>
          </tr>
          <tr>
            <td className="muted">Invoice, per base unit</td>
            <td className="num">{unitPrice(price.invoice_unit_price_normalised)}</td>
          </tr>
          <tr>
            <td className="muted">Residual per unit</td>
            <td className={cls}>{signed(price.residual_abs)}</td>
          </tr>
          <tr>
            <td className="muted">Residual</td>
            <td className={cls}>{percent(price.residual_pct)}</td>
          </tr>
          <tr>
            <td className="muted">Tolerance</td>
            <td className="num dim">
              ±{price.tolerance_pct}% or {price.tolerance_abs.toFixed(2)}
            </td>
          </tr>
          <tr className="is-total">
            <td>On the line</td>
            <td className={cls}>
              {currency} {signed(price.line_residual_abs)}
            </td>
          </tr>
          {resolution.quantity && (
            <>
              <tr>
                <td className="muted">Invoiced / available, base units</td>
                <td className="num">
                  {fmtQuantity(resolution.quantity.invoiced_base_qty)} /{" "}
                  {fmtQuantity(resolution.quantity.open_base_qty)}
                </td>
              </tr>
              <tr>
                <td className="muted">Over-billed by</td>
                <td className={resolution.quantity.within_tolerance ? "num" : "num breach"}>
                  {fmtQuantity(resolution.quantity.residual_base_qty)}
                </td>
              </tr>
            </>
          )}
          <tr className="is-aside">
            <td>Comparing the printed prices instead would have shown</td>
            <td className="num">{signed(price.naive_residual_abs)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function Findings({ detail }: { detail: InvoiceDetail }) {
  if (detail.result.findings.length === 0) {
    return (
      <section className="panel">
        <h2 className="panel-head">Findings</h2>
        <p className="note">
          No rule fired. The invoice matches on price, quantity and coding, so no model was
          consulted.
        </p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2 className="panel-head">
        Findings
        <span className="of">produced by rules, not by the model</span>
      </h2>
      <table>
        <tbody>
          {detail.result.findings.map((finding, index) => (
            <tr key={`${finding.rule_id}-${index}`}>
              <td className="mono dim" style={{ whiteSpace: "nowrap" }}>
                {finding.rule_id}
              </td>
              <td className={`sev sev-${finding.severity}`} style={{ whiteSpace: "nowrap" }}>
                {label(finding.exception_type)}
              </td>
              <td className="mono dim">{finding.line_no ? `line ${finding.line_no}` : "header"}</td>
              <td>{finding.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Reasoning({ detail }: { detail: InvoiceDetail }) {
  const { routing, assessment } = detail;

  return (
    <section className="panel">
      <h2 className="panel-head">
        Routing
        {assessment && (
          <span className="of">
            {assessment.run.provider}:{assessment.run.model}
          </span>
        )}
      </h2>

      <ol style={{ margin: "0 0 var(--s3)", paddingLeft: "1.1rem" }}>
        {routing.reasons.map((reason, index) => (
          <li key={index} style={{ marginBottom: 4, color: "var(--ink-2)" }}>
            {reason}
          </li>
        ))}
      </ol>

      {assessment ? (
        <>
          <dl className="fields">
            <dt>Proposes</dt>
            <dd>{label(assessment.assessment.proposed_action)}</dd>
            <dt>Confidence</dt>
            <dd>{assessment.assessment.confidence.toFixed(2)}</dd>
            <dt>Tier before model</dt>
            <dd>{label(detail.result.routing.tier)}</dd>
            {assessment.assessment.proposed_cost_center && (
              <>
                <dt>Proposed cost centre</dt>
                <dd>{assessment.assessment.proposed_cost_center}</dd>
              </>
            )}
          </dl>

          {assessment.assessment.evidence.length > 0 && (
            <>
              <h3 className="eyebrow">Evidence cited</h3>
              <ul className="mono fine" style={{ margin: 0, paddingLeft: "1.1rem" }}>
                {assessment.assessment.evidence.map((citation) => (
                  <li key={citation.field_path}>{citation.field_path}</li>
                ))}
              </ul>
            </>
          )}

          {assessment.grounding.ungrounded_citations.length > 0 && (
            <p className="breach" style={{ marginBottom: 0 }}>
              Stripped {assessment.grounding.ungrounded_citations.length} citation(s) naming fields
              the model was not given.
            </p>
          )}
        </>
      ) : (
        <p className="note" style={{ marginTop: 0 }}>
          No model was consulted. A clean invoice never reaches one.
        </p>
      )}
    </section>
  );
}
