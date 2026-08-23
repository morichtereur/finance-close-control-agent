/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export. There is no server, because with no authentication a live
  // API would be an unauthenticated endpoint serving finance documents — see
  // src/fcca/i2p/export_ui.py. Every page is rendered at build time from the
  // JSON in ui/data, which `fcca i2p-export` writes.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
