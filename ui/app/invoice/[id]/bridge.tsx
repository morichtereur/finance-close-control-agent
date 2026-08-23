import { percent, signed, unitPrice } from "@/lib/format";
import type { PriceComparison } from "@/lib/types";

/**
 * The tolerance bridge.
 *
 * A price variance is four numbers — agreed price, billed price, residual,
 * permitted band — and read as four numbers it takes a reviewer a moment to
 * work out whether this is a rounding argument or a real dispute. Drawn to
 * scale it takes no time at all: the band is what was agreed to be acceptable,
 * the marker is where the invoice landed, and the distance between them is the
 * disagreement.
 *
 * The scale is anchored on the purchase-order price and expressed in percent,
 * because that is the axis the tolerance is configured in. The domain widens to
 * fit whatever the residual turns out to be, so a 200% variance is still drawn
 * honestly rather than clipped at the edge and made to look marginal.
 *
 * Nothing here is decorative. Position is magnitude, band width is the
 * configured tolerance, and the only colour is the one that means "act".
 */
export function ToleranceBridge({
  price,
  currency,
}: {
  price: PriceComparison;
  currency: string;
}) {
  const breach = !price.within_tolerance;

  // Percent domain, symmetric, wide enough to hold both the band and the mark
  // with air around them. Minimum of three tolerance widths so a tiny residual
  // is not magnified into something that looks alarming.
  const tol = price.tolerance_pct;
  const residual = price.residual_pct;
  const domain = Math.max(tol * 3, Math.abs(residual) * 1.35, 1);

  const W = 1000;
  const H = 92;
  const padX = 8;
  const axisY = 58;
  const bandTop = 20;
  const bandH = 30;

  const x = (pct: number) => padX + ((pct + domain) / (2 * domain)) * (W - padX * 2);

  const bandLeft = x(-tol);
  const bandRight = x(tol);
  const markX = x(Math.max(-domain, Math.min(domain, residual)));
  const centreX = x(0);

  // Ticks at the band edges and the domain edges. Deliberately few: this is an
  // instrument for one judgement, not a chart to be read off.
  const ticks = [-domain, -tol, 0, tol, domain];

  return (
    <section className="bridge" aria-label="Price against tolerance">
      <div className="bridge-head">
        <span className={`bridge-residual${breach ? " is-breach" : ""}`}>
          {percent(price.residual_pct)}
        </span>
        <span className="bridge-sub">
          {signed(price.residual_abs)} per base unit &nbsp;·&nbsp; {currency}{" "}
          {signed(price.line_residual_abs)} on the line
        </span>
        <span className="bridge-sub" style={{ marginLeft: "auto" }}>
          {breach ? "outside tolerance" : "within tolerance"}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" preserveAspectRatio="none"
           aria-label={
             `Purchase order price ${unitPrice(price.po_unit_price_normalised)}. ` +
             `Invoice price ${unitPrice(price.invoice_unit_price_normalised)}. ` +
             `Residual ${percent(price.residual_pct)}, ` +
             `tolerance plus or minus ${tol} percent. ` +
             (breach ? "Outside tolerance." : "Within tolerance.")
           }>
        {/* Permitted band. A flat fill, no stroke on the long edges, so it
            reads as an area of acceptance rather than as a box. */}
        <rect
          x={bandLeft}
          y={bandTop}
          width={bandRight - bandLeft}
          height={bandH}
          fill="var(--paper-3)"
        />
        <line x1={bandLeft} y1={bandTop} x2={bandLeft} y2={bandTop + bandH}
              stroke="var(--rule-strong)" strokeWidth="1" />
        <line x1={bandRight} y1={bandTop} x2={bandRight} y2={bandTop + bandH}
              stroke="var(--rule-strong)" strokeWidth="1" />

        {/* Axis. */}
        <line x1={padX} y1={axisY} x2={W - padX} y2={axisY}
              stroke="var(--rule)" strokeWidth="1" />

        {/* The agreed price: a heavier rule through the whole figure, because
            it is the thing everything else is measured against. */}
        <line x1={centreX} y1={bandTop - 6} x2={centreX} y2={axisY + 6}
              stroke="var(--ink)" strokeWidth="2" />

        {/* Where the invoice landed. */}
        <line
          x1={markX}
          y1={bandTop - 10}
          x2={markX}
          y2={axisY + 10}
          stroke={breach ? "var(--signal)" : "var(--ink-2)"}
          strokeWidth="3"
        />

        {/* The gap between agreed and billed, drawn on the axis so the distance
            itself is visible rather than only implied by two marks. */}
        <line
          x1={centreX}
          y1={axisY}
          x2={markX}
          y2={axisY}
          stroke={breach ? "var(--signal)" : "var(--ink-2)"}
          strokeWidth="3"
        />

        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} y1={axisY} x2={x(t)} y2={axisY + 5}
                  stroke="var(--rule)" strokeWidth="1" />
            <text
              x={x(t)}
              y={axisY + 20}
              textAnchor={t === -domain ? "start" : t === domain ? "end" : "middle"}
              fontFamily="var(--font-mono-stack)"
              fontSize="13"
              fill="var(--ink-4)"
            >
              {t === 0 ? "0" : `${t > 0 ? "+" : ""}${t.toFixed(t === tol || t === -tol ? 1 : 0)}%`}
            </text>
          </g>
        ))}

        <text x={centreX} y={bandTop - 10} textAnchor="middle"
              fontFamily="var(--font-mono-stack)" fontSize="13" fill="var(--ink-3)">
          purchase order
        </text>
      </svg>

      <div className="bridge-legend">
        <span>
          Agreed {unitPrice(price.po_unit_price_normalised)} / base unit
        </span>
        <span>Billed {unitPrice(price.invoice_unit_price_normalised)}</span>
        <span>
          Band ±{price.tolerance_pct}% or {price.tolerance_abs.toFixed(2)}
        </span>
        <span style={{ marginLeft: "auto" }}>Both sides normalised before comparison</span>
      </div>
    </section>
  );
}
