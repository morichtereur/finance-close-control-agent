/** @type {import('next').NextConfig} */

// Served from a project page at /<repo>/ rather than a domain root, so the
// asset paths need that prefix baked in at build time. Empty locally, which
// keeps `next dev` and a root deployment working unchanged.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig = {
  // Static export. There is no server, because with no authentication a live
  // API would be an unauthenticated endpoint serving finance documents — see
  // src/fcca/i2p/export_ui.py. Every page is rendered at build time from the
  // JSON in ui/data, which `fcca i2p-export` writes.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath,
  assetPrefix: basePath || undefined,
};

export default nextConfig;
