/** @type {import('next').NextConfig} */

// GitHub Pages serves project sites from a subpath — /finance-close-control-agent/ —
// so every asset and internal link has to carry that prefix. It is read from the
// environment rather than hard-coded so `npm run dev` still serves from the root
// and a different host (Vercel, a custom domain) needs no code change.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig = {
  // Static export. There is no server, because with no authentication a live
  // API would be an unauthenticated endpoint serving finance documents — see
  // src/fcca/i2p/export_ui.py. Every page is rendered at build time from the
  // JSON in ui/data, which `fcca i2p-export` writes.
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath || undefined,
  images: { unoptimized: true },
};

export default nextConfig;
