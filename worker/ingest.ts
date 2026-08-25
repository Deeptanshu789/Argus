/**
 * Ingest supervisor. Spawns one Python inference sidecar per camera, parses the
 * newline-delimited JSON it emits, writes to Postgres, runs Module C
 * (cross-camera association) and Module D (analytics), and publishes to Redis.
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
import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { SidecarEvent, type CameraLink, type VehicleType } from "@/contract";
import { associateArrival, toHop, type TrackRecord } from "@/server/association";
import {
  bucketize, congestionScore, detectStationary, detectVolumeSpike, detectWrongWay,
  type CameraCalibration, type DetectionRow,
} from "@/server/analytics";
import * as db from "@/server/db";
import { publish, publisher } from "@/server/bus";

const PYTHON = process.env.ARGUS_PYTHON ?? ".venv/bin/python";
const CAMERAS = (process.env.ARGUS_CAMERAS ?? "CAM1=demo/cam1.mp4,CAM2=demo/cam2.mp4")
  .split(",")
  .filter(Boolean)
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

/** Detections arrive at 20/sec across four cameras. One INSERT per detection
 *  is 20 round trips a second for rows nobody reads individually. Batch. */
const FLUSH_MS = 1000;
const FLUSH_ROWS = 200;

/** How often the analytics rollup and anomaly sweep run. */
const ROLLUP_MS = 15_000;

const bus = publisher();

let links: readonly CameraLink[] = [];
let calibration = new Map<string, CameraCalibration>();

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

// ----------------------------------------------------------- detection buffer --

type Pending = {
  ts: string; camera_id: string; track_id: string;
  bbox: [number, number, number, number]; conf: number;
};
let buffer: Pending[] = [];

async function flush() {
  if (!buffer.length) return;
  const batch = buffer;
  buffer = [];
  try {
    await db.insertDetections(batch);
  } catch (e) {
    // Losing a detection batch costs a few points on a chart. Killing the
    // supervisor costs the demo. Log and keep ingesting.
    console.error("detection insert failed:", (e as Error).message);
  }
}

// --------------------------------------------------------------- sidecars --

function start(cam: { id: string; source: string }): ChildProcess {
  const proc = spawn(
    PYTHON,
    ["ml/sidecar.py", "--camera", cam.id, "--source", cam.source, "--fps", "5"],
    { stdio: ["ignore", "pipe", "pipe"] },
  );

  createInterface({ input: proc.stdout! }).on("line", (line) => {
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
    void handle(parsed.data).catch((e) =>
      console.error(`[${cam.id}] handler failed:`, (e as Error).message));
  });

  // Sidecar diagnostics belong on stderr and are passed through, not swallowed.
  createInterface({ input: proc.stderr! }).on("line", (l) => console.error(`[${cam.id}] ${l}`));

  proc.on("exit", (code) => {
    if (shuttingDown) return;
    console.error(`[${cam.id}] sidecar exited (${code}), restarting in ${RESTART_MS}ms`);
    setTimeout(() => children.set(cam.id, start(cam)), RESTART_MS);
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
      buffer.push({
        ts: ev.ts, camera_id: ev.camera_id, track_id: ev.track_id,
        bbox: ev.bbox, conf: ev.conf,
      });
      if (buffer.length >= FLUSH_ROWS) void flush();
      await publish(bus, {
        type: "detection",
        data: {
          camera_id: ev.camera_id, track_id: ev.track_id, bbox: ev.bbox,
          vehicle_type: ev.vehicle_type, plate_text: null, conf: ev.conf,
        },
      });
      return;

    case "track_closed": {
      if (!checkEmbeddingDim(ev.camera_id, ev.embedding)) return;

      // The track's own detections must be in the table before anything reads
      // them back — anomaly checks below do exactly that.
      await flush();

      const dbId = await db.upsertTrack({
        camera_id: ev.camera_id, track_id: ev.track_id,
        vehicle_type: ev.vehicle_type, color: ev.color,
        plate_text: ev.plate_text, plate_conf: ev.plate_conf,
        embedding: ev.embedding, color_hist: ev.color_hist,
        entry_time: ev.entry_time, exit_time: ev.exit_time,
      });

      // Track exit is the only moment cross-camera matching needs the
      // embedding, which is why the sidecar computes it here and nowhere else.
      const track: TrackRecord = {
        id: dbId,
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
      // the CAM1 track is already stored. Looking forward would find nothing,
      // because the continuation has not been observed yet.
      const rows = await db.candidateTracks(
        ev.camera_id, ev.entry_time, CANDIDATE_WINDOW_S);
      const candidates: TrackRecord[] = rows.map((r) => ({
        id: String(r.id),
        camera_id: r.camera_id,
        track_id: r.track_id,
        plate_text: r.plate_text,
        plate_conf: r.plate_conf,
        embedding: r.embedding ?? [],
        color_hist: r.color_hist ?? [],
        entry_time: new Date(r.entry_time).toISOString(),
        exit_time: r.exit_time ? new Date(r.exit_time).toISOString() : null,
      }));

      const match = associateArrival(track, candidates, links);
      if (match) {
        const hop = toHop(match);
        console.log(
          `[MATCH] ${match.from.camera_id}->${match.to.camera_id} ` +
          `${match.to.plate_text ?? "(no plate)"} via ${hop.method} ` +
          `conf ${hop.confidence} in ${hop.travel_time_s}s ` +
          `[${match.agreed.join("+")}]`,
        );
        await db.insertMatch({
          from_track: match.from.id, to_track: match.to.id,
          method: hop.method, confidence: hop.confidence,
          travel_time_s: hop.travel_time_s,
        });
        const plate = match.to.plate_text ?? match.from.plate_text;
        const traj = await db.extendTrajectory(
          match.from.id, match.to.id, plate,
          match.from.entry_time, match.to.exit_time);

        await publish(bus, {
          type: "match",
          data: { ...hop, trajectory_id: traj.id, plate_text: plate },
        });
        // Re-read the whole journey: the frontend draws the full path, and a
        // three-camera trip must arrive as one line, not two disconnected hops.
        const [full] = await db.getTrajectories({ limit: 1 });
        if (full && full.id === traj.id) {
          await publish(bus, { type: "trajectory_update", data: { id: full.id, path: full.path } });
        }
      }

      await anomalies(ev.camera_id, ev.track_id, ev.plate_text);
      return;
    }
  }
}

// --------------------------------------------------------------- anomalies --

async function anomalies(cameraId: string, trackId: string, plate: string | null) {
  const cal = calibration.get(cameraId);
  if (!cal) return;
  const rows = await db.trackDetections(cameraId, trackId);
  if (rows.length < 2) return;

  const input = { camera_id: cameraId, track_id: trackId, plate_text: plate, rows };
  for (const found of [detectStationary(input, cal), detectWrongWay(input, cal)]) {
    if (!found) continue;
    const alert = await db.insertAlert({
      camera_id: found.camera_id, kind: found.kind, severity: found.severity,
      track_id: found.track_id, plate_text: found.plate_text, detail: found.detail,
    });
    console.log(`[ALERT] ${found.kind} ${cameraId} ${found.detail}`);
    await publish(bus, { type: "alert", data: alert });
  }
}

// ---------------------------------------------------------------- rollup --

/**
 * Periodic analytics broadcast plus the volume-spike sweep.
 *
 * Volume spikes are the one anomaly that cannot be judged from a single track:
 * it needs a bucket series to compare against, so it lives here rather than in
 * the per-track path.
 */
async function rollup() {
  try {
    const perCamera: Record<string, { vehicle_count: number; congestion_score: number }> = {};
    let cityCount = 0;
    const citySpeeds: number[] = [];

    for (const cam of CAMERAS) {
      const cal = calibration.get(cam.id) ?? { metersPerPixel: 0.05 };
      const rows = await db.trackDetections_recent(cam.id, 3600);
      const buckets = bucketize(rows, cal);
      const latest = buckets[buckets.length - 1];
      const count = latest?.vehicle_count ?? 0;
      perCamera[cam.id] = {
        vehicle_count: count,
        congestion_score: latest?.congestion_score ?? congestionScore(count, null),
      };
      cityCount += count;
      if (latest?.avg_speed_kmh != null) citySpeeds.push(latest.avg_speed_kmh);

      const spike = detectVolumeSpike(cam.id, buckets);
      if (spike && spike.detail !== lastSpike.get(cam.id)) {
        lastSpike.set(cam.id, spike.detail);
        const alert = await db.insertAlert({
          camera_id: spike.camera_id, kind: spike.kind, severity: spike.severity,
          track_id: null, plate_text: null, detail: spike.detail,
        });
        console.log(`[ALERT] volume_spike ${cam.id} ${spike.detail}`);
        await publish(bus, { type: "alert", data: alert });
      }
    }

    await publish(bus, {
      type: "analytics",
      data: {
        ts: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
        per_camera: perCamera,
        city: {
          vehicle_count: cityCount,
          avg_speed_kmh: citySpeeds.length
            ? Math.round((citySpeeds.reduce((a, b) => a + b, 0) / citySpeeds.length) * 10) / 10
            : 0,
        },
      },
    });
  } catch (e) {
    console.error("rollup failed:", (e as Error).message);
  }
}

/** Don't re-alert the same spike every 15 seconds. */
const lastSpike = new Map<string, string>();

// ------------------------------------------------------------------- boot --

const children = new Map<string, ChildProcess>();
let shuttingDown = false;

async function boot() {
  if (!(await db.ping())) {
    console.error(
      "cannot reach the database at DATABASE_URL.\n" +
      "  docker compose up -d db redis && npm run db:setup",
    );
    process.exit(1);
  }
  links = await db.getLinks();
  calibration = await db.getCalibration();
  if (!links.length) {
    console.warn(
      "camera_links is EMPTY. Layer 3 of the association engine has no graph to " +
      "reason about, so it will abstain on every pair and matching falls back to " +
      "plate + Re-ID only. Run: npm run db:setup",
    );
  }
  console.log(`topology: ${calibration.size} cameras, ${links.length} links`);

  setInterval(() => void flush(), FLUSH_MS);
  setInterval(() => void rollup(), ROLLUP_MS);

  for (const cam of CAMERAS) {
    console.log(`spawning sidecar for ${cam.id} <- ${cam.source}`);
    children.set(cam.id, start(cam));
  }
}

/** Flush the buffer on the way out: a Ctrl-C mid-demo should not drop the last
 *  second of detections and leave the charts short. */
for (const sig of ["SIGINT", "SIGTERM"] as const) {
  process.on(sig, () => {
    shuttingDown = true;
    for (const c of children.values()) c.kill();
    void flush().finally(() => process.exit(0));
  });
}

void boot();
