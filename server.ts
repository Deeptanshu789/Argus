/**
 * Custom Next.js server: UI, /api route handlers, and the WebSocket upgrade in
 * ONE process.
 *
 * Next's App Router has no native WebSocket support, so /ws is attached to the
 * underlying HTTP server here. Running `next dev` directly boots the UI but not
 * /ws — always use `npm run dev`.
 *
 * Video inference does NOT run here. It runs in worker/ingest.ts, out of
 * process, because a CPU-pinned decode loop in this process would stall the
 * dashboard — exactly what a live demo cannot afford. The worker reaches this
 * process through Redis pub/sub, and every message it publishes is relayed to
 * the connected browsers unchanged.
 */
import { createServer } from "node:http";
import next from "next";
import { WebSocketServer, type WebSocket } from "ws";
import { mockEvents } from "@/server/mock";
import { subscribe } from "@/server/bus";
import type { ServerMessage } from "@/contract";

const dev = process.env.NODE_ENV !== "production";
const port = Number(process.env.PORT ?? 3000);
const app = next({ dev });
const handle = app.getRequestHandler();

const clients = new Set<WebSocket>();

/** The hub. Fed by the ingest worker over Redis, and by the canned loop below
 *  while MOCK is on. */
export function broadcast(msg: ServerMessage) {
  const payload = JSON.stringify(msg);
  for (const c of clients) if (c.readyState === c.OPEN) c.send(payload);
}

await app.prepare();

const server = createServer((req, res) => void handle(req, res));
const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  if (new URL(req.url ?? "/", "http://x").pathname !== "/ws") return socket.destroy();
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
});

wss.on("connection", (ws) => {
  clients.add(ws);
  ws.on("close", () => clients.delete(ws));
  ws.on("error", () => clients.delete(ws));
});

// The live feed: whatever the worker publishes reaches every browser.
subscribe(broadcast);

// Canned event loop, for building the UI with no pipeline running.
// MOCK=0 turns it off; leaving it on alongside a live worker means the map
// shows both real and invented traffic, which is not what you want in a demo.
const mock = process.env.MOCK !== "0";
if (mock) {
  void (async () => {
    for await (const msg of mockEvents()) if (clients.size) broadcast(msg);
  })();
}

server.listen(port, () => {
  console.log(`argus  http://localhost:${port}   ws://localhost:${port}/ws`);
  console.log(`  live events: redis  |  canned events: ${mock ? "ON (MOCK=0 to stop)" : "off"}`);
});
