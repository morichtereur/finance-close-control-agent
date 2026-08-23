import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "Invoice exception queue",
  description:
    "Review interface for the invoice-to-pay exception queue. Synthetic data; posting is simulated.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="masthead">
          <h1>
            <Link href="/">Invoice exception queue</Link>
          </h1>
          <nav>
            <Link href="/">Queue</Link>
            <Link href="/evaluation/">Evaluation</Link>
          </nav>
        </header>
        {/*
          Stated on every page rather than once on a landing page. Somebody
          arriving at a single invoice from a link should not have to go looking
          for the caveat.
        */}
        <div className="notice">
          Synthetic data. No real vendor, invoice, bank account or amount appears here. Posting is
          simulated: nothing in this system writes to an ERP, and no path in the codebase can.
        </div>
        <main>{children}</main>
      </body>
    </html>
  );
}
