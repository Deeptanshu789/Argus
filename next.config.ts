import type { NextConfig } from "next";

// Served by server.ts, which also owns the WebSocket upgrade. `next dev`/`next
// start` on their own will boot the UI but not /ws — use `npm run dev`.
const config: NextConfig = {
  eslint: { ignoreDuringBuilds: true },
};

export default config;
