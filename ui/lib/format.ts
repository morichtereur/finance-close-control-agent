// Formatting only. No business logic lives in the interface: every number shown
// was computed in Python and is rendered here exactly as it was recorded.

export function money(value: number, currency: string): string {
  return `${currency} ${value.toLocaleString("en-GB", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function unitPrice(value: number): string {
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: 6,
    maximumFractionDigits: 6,
  });
}

export function quantity(value: number): string {
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });
}

export function percent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

export function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString("en-GB", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function label(value: string): string {
  return value.replace(/_/g, " ");
}

export function discountSchedule(discounts: number[], surcharge: number): string {
  const parts: string[] = [];
  if (discounts.length) parts.push(discounts.map((d) => `${d}%`).join(" then "));
  if (surcharge) parts.push(`${surcharge >= 0 ? "+" : ""}${surcharge}/unit`);
  return parts.length ? parts.join(", ") : "none";
}
