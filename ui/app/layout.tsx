import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Link from "next/link";

import "./globals.css";

// IBM Plex, self-hosted at build time. Chosen rather than a neutral UI stack
// because it was drawn for an enterprise engineering context and carries true
// tabular figures — this interface is mostly numbers in columns, and figures
// that do not align are figures a reviewer cannot scan. Sans and Mono are one
// superfamily, so the page speaks in a single voice rather than two borrowed
// ones.
const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
  fallback: ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
  fallback: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
});

export const metadata: Metadata = {
  title: "Invoice exception queue",
  description:
    "Review interface for the invoice-to-pay exception queue. Synthetic data; posting is simulated.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <header className="masthead">
          <Link className="mark" href="/">
            <span className="mark-rule" aria-hidden="true" />
            Invoice exception queue
          </Link>
          <nav>
            <Link href="/">Queue</Link>
            <Link href="/evaluation/">Evaluation</Link>
          </nav>
          {/*
            Stated in the masthead rather than once on a landing page. Someone
            arriving at a single invoice from a link should not have to go
            looking for the caveat.
          */}
          <p className="provenance">
            Synthetic data · posting simulated · nothing here writes to an ERP
          </p>
        </header>
        <main id="main">{children}</main>
      </body>
    </html>
  );
}
