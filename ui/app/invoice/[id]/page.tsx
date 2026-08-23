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
import type { InvoiceDetail, LineResolution, PurchaseOrderLine } from "@/lib/types";

import { Disposition } from "./disposition";
import { Trace } from "./trace";

export function generateStaticParams() {
  return invoiceIds().map((id) => ({ id }));
}

export default async function InvoicePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const detail = invoiceDetail(id);
  const { invoice, vendor, result, routing, assessment } = detail;

  return (
    <>
      <Link className="back" href="/">
        &larr; queue
      </Link>

      <div className="split">
        <SourceDocument detail={detail} />
        <div>
          <Discrepancies detail={detail} />
          <Findings detail={detail} />
          <Routing detail={detail} />
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="pane">
          <div className="pane-head">Trace — every step, in order, with its actor</div>
          <div className="pane-body" style={{ padding: 0 }}>
            <Trace records={detail.trace} />
          </div>
        </div>
      </div>

      <Disposition
        invoiceId={invoice.invoice_id}
        tier={routing.tier}
        proposedAction={assessment?.assessment.proposed_action ?? null}
      />

      <p className="dim" style={{ marginTop: 16, maxWidth: "80ch" }}>
        {result.findings.length === 0
          ? "No rule fired on this invoice, so no model was consulted."
          : `${result.findings.length} rule finding(s). The model was consulted only after the rules flagged this invoice, and only to classify it, propose a resolution and cite evidence.`}{" "}
        {vendor ? `Checked against vendor master record ${vendor.vendor_id}.` : ""}
      </p>
    </>
  );
}

/* ------------------------------------------------------------ left pane */

function SourceDocument({ detail }: { detail: InvoiceDetail }) {
  const { invoice, vendor } = detail;
  return (
    <div className="pane">
      <div className="pane-head">Source document — as received</div>
      <div className="pane-body">
        <dl className="fields">
          <dt>Invoice</dt>
          <dd>{invoice.invoice_id}</dd>
          <dt>Vendor</dt>
          <dd>
            {vendor?.name ?? invoice.vendor_id} ({invoice.vendor_id})
          </dd>
          <dt>Their reference</dt>
          <dd>{invoice.vendor_reference}</dd>
          <dt>Category</dt>
          <dd>{invoice.category}</dd>
          <dt>Invoice date</dt>
          <dd>{invoice.invoice_date}</dd>
          <dt>Received</dt>
          <dd>{invoice.received_date}</dd>
          <dt>Company code</dt>
          <dd>{invoice.company_code}</dd>
          <dt>Net</dt>
          <dd>{money(invoice.stated_total_net, invoice.currency)}</dd>
          <dt>Tax</dt>
          <dd>{money(invoice.stated_total_tax, invoice.currency)}</dd>
          <dt>Gross</dt>
          <dd>{money(invoice.stated_total_gross, invoice.currency)}</dd>
          <dt>Bank stated</dt>
          <dd className={vendor && invoice.stated_bank_iban !== vendor.bank_iban ? "residual-out" : ""}>
            {invoice.stated_bank_iban}
          </dd>
          {vendor && (
            <>
              <dt>Bank on file</dt>
              <dd>{vendor.bank_iban}</dd>
            </>
          )}
        </dl>

        {invoice.free_text && (
          <div className="freetext">
            <span className="dim">Free text as supplied by the sender — untrusted input:</span>
            <br />
            {invoice.free_text}
          </div>
        )}

        <h3>Lines as invoiced</h3>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Their item</th>
              <th>Description</th>
              <th style={{ textAlign: "right" }}>Qty</th>
              <th>UoM</th>
              <th style={{ textAlign: "right" }}>List</th>
              <th style={{ textAlign: "right" }}>/unit</th>
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
                <td className="num">{fmtQuantity(line.quantity)}</td>
                <td className="mono dim">{line.uom}</td>
                <td className="num">{line.price.list_price}</td>
                <td className="num dim">{line.price.price_unit}</td>
                <td className="mono dim">
                  {discountSchedule(line.price.discount_pct, line.price.surcharge_per_unit)}
                </td>
                <td className="num dim">{line.tax_rate}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        {Object.keys(detail.purchase_orders).length > 0 && (
          <>
            <h3>Purchase order lines</h3>
            <table>
              <thead>
                <tr>
                  <th>PO</th>
                  <th>#</th>
                  <th>Material</th>
                  <th style={{ textAlign: "right" }}>Qty</th>
                  <th>UoM</th>
                  <th style={{ textAlign: "right" }}>List</th>
                  <th style={{ textAlign: "right" }}>/unit</th>
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
                      <td className="num dim">{line.price.price_unit}</td>
                      <td className="mono dim">
                        {discountSchedule(line.price.discount_pct, line.price.surcharge_per_unit)}
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </>
        )}

        {detail.goods_receipts.length > 0 && (
          <>
            <h3>Goods receipts</h3>
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
          </>
        )}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- right pane */

function Discrepancies({ detail }: { detail: InvoiceDetail }) {
  const priced = detail.result.resolutions.filter((r) => r.price !== null);

  return (
    <div className="pane">
      <div className="pane-head">
        Comparison — system value, document value, normalised, residual, tolerance
      </div>
      <div className="pane-body">
        {priced.length === 0 ? (
          <p className="muted" style={{ margin: 0 }}>
            No purchase order to compare against — this is an FI invoice, governed by coding
            completeness and approval rather than by three-way match.
          </p>
        ) : (
          priced.map((resolution) => (
            <LineComparison
              key={resolution.line_no}
              resolution={resolution}
              currency={detail.result.currency}
            />
          ))
        )}
        <CodingTable detail={detail} />
      </div>
    </div>
  );
}

function LineComparison({
  resolution,
  currency,
}: {
  resolution: LineResolution;
  currency: string;
}) {
  const price = resolution.price!;
  const outside = !price.within_tolerance;

  return (
    <div style={{ marginBottom: 14 }}>
      <h3 style={{ marginTop: 0 }}>Line {resolution.line_no} — price</h3>
      <table>
        <tbody>
          <tr>
            <td className="muted">Purchase order, normalised per base unit</td>
            <td className="num">{unitPrice(price.po_unit_price_normalised)}</td>
          </tr>
          <tr>
            <td className="muted">Invoice, normalised per base unit</td>
            <td className="num">{unitPrice(price.invoice_unit_price_normalised)}</td>
          </tr>
          <tr>
            <td className="muted">Residual per unit</td>
            <td className={`num ${outside ? "residual-out" : "residual-in"}`}>
              {signed(price.residual_abs)}
            </td>
          </tr>
          <tr>
            <td className="muted">Residual %</td>
            <td className={`num ${outside ? "residual-out" : "residual-in"}`}>
              {percent(price.residual_pct)}
            </td>
          </tr>
          <tr>
            <td className="muted">Residual on the line</td>
            <td className={`num ${outside ? "residual-out" : "residual-in"}`}>
              {currency} {signed(price.line_residual_abs)}
            </td>
          </tr>
          <tr>
            <td className="muted">Tolerance</td>
            <td className="num dim">
              {price.tolerance_pct}% or {price.tolerance_abs.toFixed(2)}
            </td>
          </tr>
          <tr>
            <td className="muted">Within tolerance</td>
            <td className={`num ${outside ? "residual-out" : ""}`}>
              {price.within_tolerance ? "yes" : "no"}
            </td>
          </tr>
          <tr>
            <td className="dim">
              Naive comparison of the printed prices would have shown
            </td>
            <td className="num dim">{signed(price.naive_residual_abs)}</td>
          </tr>
        </tbody>
      </table>

      {resolution.quantity && (
        <table style={{ marginTop: 6 }}>
          <tbody>
            <tr>
              <td className="muted">Invoiced, base units</td>
              <td className="num">{fmtQuantity(resolution.quantity.invoiced_base_qty)}</td>
            </tr>
            <tr>
              <td className="muted">Received, base units</td>
              <td className="num">{fmtQuantity(resolution.quantity.received_base_qty)}</td>
            </tr>
            <tr>
              <td className="muted">Available to invoice</td>
              <td className="num">{fmtQuantity(resolution.quantity.open_base_qty)}</td>
            </tr>
            <tr>
              <td className="muted">Over-billed by</td>
              <td
                className={`num ${
                  resolution.quantity.within_tolerance ? "residual-in" : "residual-out"
                }`}
              >
                {fmtQuantity(resolution.quantity.residual_base_qty)}
              </td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}

function CodingTable({ detail }: { detail: InvoiceDetail }) {
  return (
    <>
      <h3>Coding</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Material</th>
            <th>Tax code</th>
            <th>GL account</th>
            <th>Cost centre</th>
          </tr>
        </thead>
        <tbody>
          {detail.result.resolutions.map((resolution) => (
            <tr key={resolution.line_no}>
              <td className="mono dim">{resolution.line_no}</td>
              <td className="mono">{resolution.material_id ?? "—"}</td>
              <td className="mono">{resolution.tax_code ?? "—"}</td>
              <td className="mono">
                {resolution.gl_account ?? "—"}{" "}
                <span className="dim">({label(resolution.gl_source)})</span>
              </td>
              <td className="mono">
                {resolution.cost_center ?? "—"}{" "}
                <span className="dim">({label(resolution.cost_center_source)})</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="dim" style={{ marginTop: 4 }}>
        &ldquo;stated&rdquo; means the vendor supplied the value; &ldquo;derived&rdquo; means a rule
        computed it from master data.
      </p>
    </>
  );
}

function Findings({ detail }: { detail: InvoiceDetail }) {
  if (detail.result.findings.length === 0) return null;
  return (
    <div className="pane" style={{ marginTop: 16 }}>
      <div className="pane-head">Findings — produced by rules, not by the model</div>
      <div className="pane-body" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>Rule</th>
              <th>Type</th>
              <th>Line</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {detail.result.findings.map((finding, index) => (
              <tr key={`${finding.rule_id}-${index}`}>
                <td className="mono">{finding.rule_id}</td>
                <td className={`sev sev-${finding.severity}`}>{label(finding.exception_type)}</td>
                <td className="mono dim">{finding.line_no ?? "header"}</td>
                <td>{finding.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Routing({ detail }: { detail: InvoiceDetail }) {
  const { routing, assessment } = detail;
  return (
    <div className="pane" style={{ marginTop: 16 }}>
      <div className="pane-head">Routing</div>
      <div className="pane-body">
        <dl className="fields">
          <dt>Tier</dt>
          <dd className={`tier tier-${routing.tier}`}>{label(routing.tier)}</dd>
          <dt>Rules alone said</dt>
          <dd>{label(detail.result.routing.tier)}</dd>
          {assessment && (
            <>
              <dt>Model</dt>
              <dd>
                {assessment.run.provider}:{assessment.run.model}
              </dd>
              <dt>Proposes</dt>
              <dd>{label(assessment.assessment.proposed_action)}</dd>
              <dt>Confidence</dt>
              <dd>{assessment.assessment.confidence.toFixed(2)}</dd>
              {assessment.assessment.proposed_cost_center && (
                <>
                  <dt>Proposed cost centre</dt>
                  <dd>{assessment.assessment.proposed_cost_center}</dd>
                </>
              )}
            </>
          )}
        </dl>

        <h3>Why</h3>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          {routing.reasons.map((reason, index) => (
            <li key={index} style={{ marginBottom: 4 }}>
              {reason}
            </li>
          ))}
        </ul>

        {assessment && assessment.assessment.evidence.length > 0 && (
          <>
            <h3>Evidence the model cited</h3>
            <ul className="mono" style={{ margin: 0, paddingLeft: 18, fontSize: 11.5 }}>
              {assessment.assessment.evidence.map((citation) => (
                <li key={citation.field_path}>{citation.field_path}</li>
              ))}
            </ul>
            {assessment.grounding.ungrounded_citations.length > 0 && (
              <p className="residual-out" style={{ marginBottom: 0 }}>
                Stripped {assessment.grounding.ungrounded_citations.length} citation(s) naming
                fields the model was not given.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
