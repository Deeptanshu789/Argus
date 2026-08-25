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
import { SidecarEvent, type CameraLink } from "@/contract";
import { associateArrival, toHop, type TrackRecord } from "@/server/association";
import { LINKS } from "@/server/mock";

const PYTHON = process.env.ARGUS_PYTHON ?? ".venv/bin/python";
const CAMERAS = (process.env.ARGUS_CAMERAS ?? "CAM1=demo/cam1.mp4,CAM2=demo/cam2.mp4")
  .split(",")
  .map((s) => {
    const [id, source] = s.split("=");
    return { id: id!.trim(), source: source!.trim() };
  });

/** ponytail: fixed 2s backoff. Exponential backoff if a flaky RTSP source shows up. */
const RESTART_MS = 2000;

/**
 * How far back to look for a vehicle's previous camera. Anything older than the
 * slowest link in the graph cannot be a candidate, so holding more is waste.
 */
const CANDIDATE_WINDOW_S = 900;

/**
 * ponytail: in-memory ring of recently closed tracks, and the camera graph read
 * from the mock fixtures. Correct for a 4-camera demo and it makes the hour-10
 * gate reachable before the database layer exists.
 *
 * Dev A replaces both with queries: candidates become
 * `SELECT * FROM tracks WHERE exit_time > now() - interval '15 minutes'`, and
 * the graph becomes `SELECT * FROM camera_links`. Everything below that line —
 * associate(), toHop() — stays exactly as it is.
 */
const recent: TrackRecord[] = [];
const links: readonly CameraLink[] = LINKS;

/**
 * Every camera must produce the same embedding dimension, or Module C's layer 2
 * compares vectors of different lengths, cosine() returns 0, and appearance
 * matching silently stops working. That failure reads as "Re-ID just isn't
 * matching well" and can burn hours. Fail loudly on the first mismatch instead.
 */
let embeddingDim: number | null = null;

function checkEmbeddingDim(cameraId: string, embedding: readonly number[]): boolean {
  if (embeddingDim === null) {
    embeddingDim = embedding.length;
    console.log(`embedding dimension: ${embeddingDim}`);
    return true;
  }
  if (embedding.length !== embeddingDim) {
    console.error(
      `[${cameraId}] embedding is ${embedding.length}-dim but ${embeddingDim}-dim ` +
      `was seen first. Cross-camera appearance matching CANNOT work until every ` +
      `sidecar uses the same ReID model. Dropping this track.`,
    );
    return false;
  }
  return true;
}

function remember(t: TrackRecord) {
  recent.push(t);
  const cutoff = Date.now() - CANDIDATE_WINDOW_S * 1000;
  while (recent.length && Date.parse(recent[0]!.entry_time) < cutoff) recent.shift();
}

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
      // TODO(Dev A): insert into `detections`; publish a `detection` message to
      // Redis so server.ts broadcast()s it.
      return;

    case "track_closed": {
      if (!checkEmbeddingDim(ev.camera_id, ev.embedding)) return;

      // Track exit is the only moment cross-camera matching needs the
      // embedding, which is why the sidecar computes it here and nowhere else.
      const track: TrackRecord = {
        id: `${ev.camera_id}:${ev.track_id}`,
        camera_id: ev.camera_id,
        track_id: ev.track_id,
        plate_text: ev.plate_text,
        plate_conf: ev.plate_conf,
        embedding: ev.embedding,
        color_hist: ev.color_hist,
        entry_time: ev.entry_time,
        exit_time: ev.exit_time,
      };

      // Search BACKWARDS. A vehicle leaves CAM1 at 10:00 and reaches CAM3 at
      // 10:03, so CAM3's track closes last — by the time this event arrives,
      // the CAM1 track is already in `recent`. Looking forward would find
      // nothing, because the continuation has not been observed yet.
      const candidates = recent.filter(
        (c) => c.camera_id !== track.camera_id && c.exit_time! < track.entry_time,
      );
      const match = associateArrival(track, candidates, links);
      if (match) {
        const hop = toHop(match);
        console.log(
          `[MATCH] ${match.from.camera_id}->${match.to.camera_id} ` +
          `${match.to.plate_text ?? "(no plate)"} via ${hop.method} ` +
          `conf ${hop.confidence} in ${hop.travel_time_s}s ` +
          `[${match.agreed.join("+")}]`,
        );
        // TODO(Dev A): insert into `matches`, extend `trajectories`, publish
        // `match` and `trajectory_update` messages.
      }

      remember(track);
      // TODO(Dev A): insert into `tracks`.
      return;
    }
  }
}

for (const cam of CAMERAS) {
  console.log(`spawning sidecar for ${cam.id} <- ${cam.source}`);
  start(cam);
}
