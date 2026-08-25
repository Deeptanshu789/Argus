/**
 * Ingest supervisor. Spawns one Python inference sidecar per camera, parses the
 * newline-delimited JSON it emits, and publishes to Redis.
 *
 * This is the ONLY process that touches Python. Everything downstream —
 * association, analytics, API, WebSocket, UI — is TypeScript.
 *
 * It runs out of process from the Next.js server on purpose: decoding video and
 * running YOLO pins a CPU core, and doing that inside the server process would
 * stall the dashboard.
 *
 *   npm run worker
 */
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { SidecarEvent } from "@/contract";

const PYTHON = process.env.ARGUS_PYTHON ?? ".venv/bin/python";
const CAMERAS = (process.env.ARGUS_CAMERAS ?? "CAM1=demo/cam1.mp4,CAM2=demo/cam2.mp4")
  .split(",")
  .map((s) => {
    const [id, source] = s.split("=");
    return { id: id!.trim(), source: source!.trim() };
  });

/** ponytail: fixed 2s backoff. Exponential backoff if a flaky RTSP source shows up. */
const RESTART_MS = 2000;

function start(cam: { id: string; source: string }) {
  const proc = spawn(
    PYTHON,
    ["ml/sidecar.py", "--camera", cam.id, "--source", cam.source, "--fps", "5"],
    { stdio: ["ignore", "pipe", "pipe"] },
  );

  createInterface({ input: proc.stdout }).on("line", (line) => {
    if (!line.trim()) return;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      // A sidecar that prints to stdout instead of stderr is a bug worth seeing,
      // not worth crashing the supervisor over.
      console.warn(`[${cam.id}] non-JSON stdout: ${line.slice(0, 120)}`);
      return;
    }
    const parsed = SidecarEvent.safeParse(raw);
    if (!parsed.success) {
      console.warn(`[${cam.id}] contract violation:`, parsed.error.issues[0]?.message);
      return;
    }
    void handle(parsed.data);
  });

  // Sidecar diagnostics belong on stderr and are passed through, not swallowed.
  createInterface({ input: proc.stderr }).on("line", (l) => console.error(`[${cam.id}] ${l}`));

  proc.on("exit", (code) => {
    console.error(`[${cam.id}] sidecar exited (${code}), restarting in ${RESTART_MS}ms`);
    setTimeout(() => start(cam), RESTART_MS);
  });

  return proc;
}

async function handle(ev: SidecarEvent) {
  switch (ev.event) {
    case "ready":
      console.log(`[${ev.camera_id}] ready at ${ev.fps} fps`);
      return;
    case "error":
      console.error(`[${ev.camera_id}] ${ev.detail}`);
      return;
    case "detection":
      // TODO(Dev A): insert into `detections`, publish a `detection` message.
      return;
    case "track_closed":
      // TODO(Dev A): insert into `tracks`, then hand the embedding to the
      // association engine (src/server/association.ts). Track exit is the only
      // moment cross-camera matching needs the embedding.
      return;
  }
}

for (const cam of CAMERAS) {
  console.log(`spawning sidecar for ${cam.id} <- ${cam.source}`);
  start(cam);
}
