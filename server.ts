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
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { readFileSync, existsSync } from "node:fs";
import { networkInterfaces } from "node:os";
import next from "next";
import { WebSocketServer, type WebSocket } from "ws";
import { mockEvents } from "@/server/mock";
import { subscribe } from "@/server/bus";
import * as db from "@/server/db";
import { dropFeed, nextFrame, putFrame, STALE_MS } from "@/server/frames";
import type { ServerMessage } from "@/contract";

const dev = process.env.NODE_ENV !== "production";
const port = Number(process.env.PORT ?? 3000);
const app = next({ dev });
const handle = app.getRequestHandler();

const clients = new Set<WebSocket>();

/** Addresses a phone on the same network can actually reach. Printed at boot
 *  because "open localhost on your phone" is the first thing everyone tries. */
function lanAddresses(): string[] {
  return Object.values(networkInterfaces())
    .flat()
    .filter((n): n is NonNullable<typeof n> =>
      !!n && n.family === "IPv4" && !n.internal)
    // Docker's bridges are IPv4 and non-internal but reach nothing useful.
    .map((n) => n.address)
    .filter((a) => !a.startsWith("172.1") && !a.startsWith("172.2"));
}

/** The hub. Fed by the ingest worker over Redis, and by the canned loop below
 *  while MOCK is on. */
export function broadcast(msg: ServerMessage) {
  const payload = JSON.stringify(msg);
  for (const c of clients) if (c.readyState === c.OPEN) c.send(payload);
}

await app.prepare();

/**
 * MJPEG out. This is what a paired phone's camera looks like to the sidecar,
 * and it is served HERE rather than from an /api route handler because Next
 * buffers and transforms responses in ways an endless multipart stream does not
 * survive. The raw Node response gives exact control over the boundary framing
 * that OpenCV's FFmpeg backend expects.
 */
async function mjpeg(cameraId: string, res: ServerResponse) {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=argusframe",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    Connection: "close",
    Pragma: "no-cache",
  });

  let open = true;
  res.on("close", () => { open = false; });

  while (open) {
    const frame = await nextFrame(cameraId, STALE_MS);
    // Nothing for STALE_MS: the phone is gone. Ending the response ends the
    // sidecar's stream, and the supervisor restarts it — which is the right
    // behaviour for a camera that dropped off, and exactly what it does for a
    // real one.
    if (!frame || frame.length === 0) break;
    if (!open) break;
    res.write(`--argusframe\r\nContent-Type: image/jpeg\r\n` +
              `Content-Length: ${frame.length}\r\n\r\n`);
    res.write(frame);
    res.write("\r\n");
  }
  res.end();
}

const server = createServer((req: IncomingMessage, res: ServerResponse) => {
  const path = new URL(req.url ?? "/", "http://x").pathname;
  // Intercepted before Next sees it. See mjpeg() above.
  const stream = /^\/cam-stream\/([A-Za-z0-9_-]+)$/.exec(path);
  if (stream) return void mjpeg(stream[1]!, res);
  void handle(req, res);
});

const wss = new WebSocketServer({ noServer: true });
/** Phones pushing JPEG frames. Separate server: different protocol entirely —
 *  binary in, nothing out — and mixing it with the dashboard hub would mean
 *  every camera frame fanning out to every browser. */
const camWss = new WebSocketServer({ noServer: true });

wss.on("connection", (ws) => {
  clients.add(ws);
  ws.on("close", () => clients.delete(ws));
  ws.on("error", () => clients.delete(ws));
});

camWss.on("connection", (ws: WebSocket, req: IncomingMessage) => {
  const code = new URL(req.url ?? "/", "http://x").searchParams.get("code") ?? "";
  void (async () => {
    // The code is the only credential, so it is checked here and not trusted
    // from anything the client says afterwards. An unknown or revoked code gets
    // no camera and no frames are ever accepted.
    const device = await db.pairDevice(code, "browser", null).catch(() => null);
    if (!device) {
      ws.send(JSON.stringify({ error: "unknown or revoked code" }));
      return ws.close();
    }
    const cameraId = device.camera_id;
    ws.send(JSON.stringify({ ok: true, camera_id: cameraId }));
    console.log(`[cam] ${cameraId} paired from a browser`);

    let frames = 0;
    ws.on("message", (data: Buffer, isBinary: boolean) => {
      if (!isBinary || !data.length) return;
      putFrame(cameraId, data);
      // One write per second at 5 fps, not one per frame.
      if (frames++ % 25 === 0) void db.touchDevice(cameraId).catch(() => {});
    });
    ws.on("close", () => {
      console.log(`[cam] ${cameraId} disconnected after ${frames} frames`);
      dropFeed(cameraId);
    });
    ws.on("error", () => dropFeed(cameraId));
  })();
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

const upgrade = (
  req: IncomingMessage, socket: import("node:stream").Duplex, head: Buffer,
) => {
  const url = new URL(req.url ?? "/", "http://x");
  if (url.pathname === "/ws") {
    return wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
  }
  if (url.pathname === "/ws/cam") {
    return camWss.handleUpgrade(req, socket, head, (ws) => camWss.emit("connection", ws, req));
  }
  socket.destroy();
};
server.on("upgrade", upgrade);

server.listen(port, () => {
  console.log(`argus  http://localhost:${port}   ws://localhost:${port}/ws`);
  console.log(`  live events: postgres |  canned events: ${mock ? "ON (MOCK=0 to stop)" : "off"}`);
  for (const a of lanAddresses()) console.log(`  this network: http://${a}:${port}`);
});

/**
 * HTTPS, if a certificate has been generated.
 *
 * A PHONE cannot use its camera over plain http: getUserMedia is only exposed
 * in a secure context, and an IP address is never one. This laptop is exempt
 * because localhost always counts, which is why the local webcam works with no
 * certificate at all.
 *
 * Served here rather than behind a proxy so the /ws and /ws/cam upgrades work
 * unchanged, and so `npm run dev` remains the only command.
 */
const CERT = process.env.ARGUS_CERT ?? ".certs/cert.pem";
const KEY = process.env.ARGUS_KEY ?? ".certs/key.pem";
if (existsSync(CERT) && existsSync(KEY)) {
  const httpsPort = Number(process.env.HTTPS_PORT ?? 3443);
  const secure = createHttpsServer(
    { cert: readFileSync(CERT), key: readFileSync(KEY) },
    (req, res) => {
      const path = new URL(req.url ?? "/", "http://x").pathname;
      const stream = /^\/cam-stream\/([A-Za-z0-9_-]+)$/.exec(path);
      if (stream) return void mjpeg(stream[1]!, res);
      void handle(req, res);
    },
  );
  secure.on("upgrade", upgrade);
  secure.listen(httpsPort, () => {
    for (const a of lanAddresses()) {
      console.log(`  phones: https://${a}:${httpsPort}/devices  (accept the certificate)`);
    }
  });
} else {
  console.log(`  no ${CERT}: phone cameras need HTTPS — run ./scripts/dev-https.sh`);
}
