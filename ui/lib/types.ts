// Shapes written by `fcca i2p-export`. Kept narrow: the interface renders what
// the pipeline recorded and computes nothing of its own, so anything the
// reviewer sees can be traced to a Python module that produced it.

export type Tier = "auto_clear" | "propose_and_approve" | "escalate";
export type Severity = "low" | "medium" | "high";
export type Actor = "rule" | "model" | "human";

export interface QueueRow {
  invoice_id: string;
  vendor_id: string;
  vendor_name: string;
  category: "MM" | "FI";
  currency: string;
  document_value: number;
  invoice_date: string;
  received_date: string;
  vendor_reference: string;
  exception_type: string;
  severity: Severity | null;
  findings: number;
  tier: Tier;
  tier_before_model: Tier;
  model_called: boolean;
  confidence: number | null;
  lines: number;
}

export interface PriceElements {
  list_price: number;
  price_unit: number;
  discount_pct: number[];
  surcharge_per_unit: number;
}

export interface InvoiceLine {
  line_no: number;
  description: string;
  supplier_item_no: string;
  quantity: number;
  uom: string;
  price: PriceElements;
  tax_rate: number;
  po_id: string | null;
  po_line: number | null;
  gl_account: string | null;
  cost_center: string | null;
}

export interface Invoice {
  invoice_id: string;
  vendor_id: string;
  company_code: string;
  category: "MM" | "FI";
  invoice_date: string;
  received_date: string;
  vendor_reference: string;
  currency: string;
  lines: InvoiceLine[];
  stated_bank_iban: string;
  free_text: string;
  stated_total_net: number;
  stated_total_tax: number;
  stated_total_gross: number;
}

export interface Vendor {
  vendor_id: string;
  name: string;
  country: string;
  currency: string;
  payment_terms: string;
  tax_id: string;
  bank_iban: string;
  bank_name: string;
}

export interface PriceComparison {
  po_unit_price_normalised: number;
  invoice_unit_price_normalised: number;
  residual_abs: number;
  residual_pct: number;
  line_residual_abs: number;
  naive_residual_abs: number;
  within_tolerance: boolean;
  tolerance_pct: number;
  tolerance_abs: number;
}

export interface QuantityComparison {
  invoiced_base_qty: number;
  received_base_qty: number;
  open_base_qty: number;
  residual_base_qty: number;
  within_tolerance: boolean;
  tolerance_pct: number;
}

export interface LineResolution {
  line_no: number;
  material_id: string | null;
  material_source: string;
  tax_code: string | null;
  gl_account: string | null;
  gl_source: "stated" | "derived" | "unresolved";
  cost_center: string | null;
  cost_center_source: "stated" | "derived_from_po" | "unresolved";
  price: PriceComparison | null;
  quantity: QuantityComparison | null;
}

export interface Finding {
  rule_id: string;
  exception_type: string;
  line_no: number | null;
  severity: Severity;
  detail: string;
  evidence: Record<string, unknown>;
}

export interface RoutingDecision {
  tier: Tier;
  reasons: string[];
  deciding_reason: string;
  model_confidence: number | null;
  document_value: number;
  exception_type: string;
}

export interface InvoiceResult {
  invoice_id: string;
  category: "MM" | "FI";
  document_value: number;
  currency: string;
  resolutions: LineResolution[];
  findings: Finding[];
  duplicate_candidates: string[];
  routing: RoutingDecision;
  evaluated_at: string;
}

export interface Assessment {
  assessment: {
    invoice_id: string;
    classification: string;
    proposed_action: string;
    rationale: string;
    evidence: { field_path: string }[];
    confidence: number;
    proposed_cost_center: string | null;
  };
  grounding: {
    total_citations: number;
    grounded_citations: number;
    ungrounded_citations: string[];
  };
  run: {
    provider: string;
    model: string;
    latency_ms: number;
    prompt_sha256: string;
    parse_attempts: number;
  };
  raw_output: string;
}

export interface TraceRecord {
  timestamp: string;
  case_id: string;
  module: string;
  step_name: string;
  actor: Actor;
  input_hash: string;
  outcome: string;
  summary: string;
  rule_id: string | null;
  model: string | null;
  prompt_version: string | null;
  detail: Record<string, unknown>;
}

export interface PurchaseOrderLine {
  po_line: number;
  material_id: string;
  supplier_item_no: string;
  quantity: number;
  uom: string;
  price: PriceElements;
  tax_code: string;
  gl_account: string;
  cost_center: string;
}

export interface PurchaseOrder {
  po_id: string;
  vendor_id: string;
  company_code: string;
  po_date: string;
  currency: string;
  lines: PurchaseOrderLine[];
}

export interface GoodsReceipt {
  gr_id: string;
  po_id: string;
  po_line: number;
  receipt_date: string;
  quantity: number;
  uom: string;
}

export interface InvoiceDetail {
  invoice_id: string;
  invoice: Invoice;
  vendor: Vendor | null;
  purchase_orders: Record<string, PurchaseOrder>;
  goods_receipts: GoodsReceipt[];
  result: InvoiceResult;
  routing: RoutingDecision;
  assessment: Assessment | null;
  model_called: boolean;
  trace: TraceRecord[];
}

export interface ClassMetrics {
  exception_type: string;
  support: number;
  predicted: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  is_absent: boolean;
}

export interface Evaluation {
  provider: string;
  model: string;
  invoices: number;
  tier_counts: Record<string, number>;
  touchless_rate: number;
  false_auto_post_count: number;
  false_auto_post_ids: string[];
  per_class: ClassMetrics[];
  actual_counts: Record<string, number>;
  predicted_counts: Record<string, number>;
  model_calls: number;
  invoices_with_findings: number;
  exact_agreement: number;
  settings_snapshot: Record<string, number>;
}
