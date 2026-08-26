/**
 * Mock fixtures — every shape in src/contract.ts, served without a database.
 *
 * WHY THIS EXISTS: it is the unblocker. The frontend developer builds every
 * view against these fixtures and never waits on the CV pipeline or the
 * database. Each real route replaces a mock one behind an identical shape.
 *
 * Fixtures are seeded, so reloads give identical data. A chart that reshuffles
 * on refresh makes it impossible to tell a UI bug from new data.
 */
import type {
  Alert, AnalyticsResponse, Camera, CameraLink, Device, SearchResult, Track,
  Trajectory, Upload, UploadResult,
} from "@/contract";
import { VehicleType } from "@/contract";

/** mulberry32 — 4 lines, deterministic, no dependency. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = <T,>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)]!;
const int = (r: () => number, lo: number, hi: number) => lo + Math.floor(r() * (hi - lo + 1));
const iso = (d: Date) => d.toISOString().replace(/\.\d{3}Z$/, "Z");

const NOW = new Date(Math.floor(Date.now() / 1000) * 1000);
const ago = (s: number) => iso(new Date(NOW.getTime() - s * 1000));
const TYPES = VehicleType.options;
const COLORS = ["white", "silver", "black", "red", "blue"] as const;
const STATES = ["KA", "MH", "DL", "TN"] as const;
const LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ";

const plate = (r: () => number) =>
  `${pick(r, STATES)}${String(int(r, 1, 30)).padStart(2, "0")}` +
  `${pick(r, [...LETTERS])}${pick(r, [...LETTERS])}` +
  `${String(int(r, 1, 9999)).padStart(4, "0")}`;

/** A known-good plate the demo and rehearsal can always rely on finding. */
export const DEMO_PLATE = "KA05MR7821";

// Bengaluru MG Road corridor.
export const CAMERAS: Camera[] = [
  { id: "CAM1", name: "MG Road Junction", lat: 12.9752, lon: 77.6068, heading_deg: 90,  status: "online",   stream_url: "/api/mock/stream/CAM1" },
  { id: "CAM2", name: "Trinity Circle",   lat: 12.9730, lon: 77.6198, heading_deg: 45,  status: "online",   stream_url: "/api/mock/stream/CAM2" },
  { id: "CAM3", name: "Ulsoor Gate",      lat: 12.9788, lon: 77.6101, heading_deg: 270, status: "online",   stream_url: "/api/mock/stream/CAM3" },
  // Deliberately not "online" — the UI must render a degraded camera.
  { id: "CAM4", name: "Richmond Circle",  lat: 12.9611, lon: 77.5966, heading_deg: 180, status: "degraded", stream_url: "/api/mock/stream/CAM4" },
];

export const LINKS: CameraLink[] = [
  { from: "CAM1", to: "CAM3", distance_m: 1400, travel_time_s: 168 },
  { from: "CAM3", to: "CAM1", distance_m: 1400, travel_time_s: 168 },
  { from: "CAM1", to: "CAM2", distance_m: 1500, travel_time_s: 200 },
  { from: "CAM2", to: "CAM1", distance_m: 1500, travel_time_s: 200 },
  { from: "CAM1", to: "CAM4", distance_m: 1900, travel_time_s: 240 },
  { from: "CAM4", to: "CAM1", distance_m: 1900, travel_time_s: 240 },
];

export const TRACKS: Track[] = (() => {
  const r = rng(1);
  const out: Track[] = [];
  for (let i = 0; i < 120; i++) {
    const entrySec = int(r, 30, 3600);
    // ~15% unreadable. Re-ID-only matching is exactly what the demo shows off,
    // so the UI must render this case rather than printing "null".
    const readable = r() > 0.15;
    out.push({
      id: String(8000 + i),
      camera_id: pick(r, CAMERAS).id,
      track_id: `T${String(i).padStart(4, "0")}`,
      plate_text: readable ? plate(r) : null,
      plate_conf: readable ? Math.round((0.82 + r() * 0.17) * 100) / 100 : null,
      vehicle_type: pick(r, TYPES),
      color: pick(r, COLORS),
      entry_time: ago(entrySec),
      exit_time: ago(entrySec - int(r, 4, 20)),
      // ~10% unmeasurable: a track too short to time, or an uncalibrated
      // camera. The UI must render a blank speed, not "0 km/h".
      speed_kmh: r() > 0.1 ? Math.round((18 + r() * 45) * 10) / 10 : null,
    });
  }
  out.sort((a, b) => (a.entry_time < b.entry_time ? 1 : -1));
  out[0] = { ...out[0]!, plate_text: DEMO_PLATE, plate_conf: 0.97, camera_id: "CAM1" };
  return out;
})();

export const TRAJECTORIES: Trajectory[] = (() => {
  const r = rng(2);
  const out: Trajectory[] = [];
  for (let i = 0; i < 24; i++) {
    const rest = CAMERAS.slice(1).filter(() => r() > 0.35);
    const cams = [CAMERAS[0]!, ...(rest.length ? rest : [CAMERAS[2]!])];
    const startSec = int(r, 120, 3000);
    let t = 0;
    const path: [number, number, number][] = [];
    const hops = [];
    for (let k = 0; k < cams.length - 1; k++) {
      const a = cams[k]!, b = cams[k + 1]!;
      path.push([a.lon, a.lat, t]);
      const link = LINKS.find((l) => l.from === a.id && l.to === b.id);
      const dt = link ? link.travel_time_s : int(r, 120, 300);
      t += dt;
      const u = r();
      const method = u < 0.65 ? "plate" : u < 0.9 ? "reid" : "spatial_temporal";
      hops.push({
        from_camera: a.id, to_camera: b.id, method,
        confidence: method === "plate" ? 0.99 : Math.round((0.76 + r() * 0.17) * 100) / 100,
        travel_time_s: dt,
      } as const);
    }
    const last = cams[cams.length - 1]!;
    path.push([last.lon, last.lat, t]);
    out.push({
      id: `TRJ-${1000 + i}`,
      plate_text: i === 0 ? DEMO_PLATE : plate(r),
      vehicle_type: pick(r, TYPES),
      started_at: ago(startSec),
      ended_at: ago(startSec - t),
      path,
      hops: [...hops],
    });
  }
  return out;
})();

export const ALERTS: Alert[] = [
  { id: "A-77", ts: ago(180),  camera_id: "CAM2", kind: "stationary",   severity: "warn",     track_id: null, plate_text: "MH04AQ5678", detail: "Stationary for 7m 12s",           acked: false },
  { id: "A-78", ts: ago(540),  camera_id: "CAM4", kind: "wrong_way",    severity: "critical", track_id: null, plate_text: null,         detail: "Vehicle travelling against flow", acked: false },
  { id: "A-79", ts: ago(1260), camera_id: "CAM1", kind: "volume_spike", severity: "info",     track_id: null, plate_text: null,         detail: "Volume 41% above 1h baseline",    acked: true  },
];

function series(cameraId: string, buckets = 12) {
  const r = rng([...cameraId].reduce((a, c) => a + c.charCodeAt(0), 7));
  return Array.from({ length: buckets }, (_, i) => {
    const count = int(r, 40, 120);
    const speed = Math.round((12 + r() * 34) * 10) / 10;
    const weights = [0.7, 0.04, 0.09, 0.14, 0.03];
    return {
      ts: ago(300 * (buckets - i)),
      vehicle_count: count,
      avg_speed_kmh: speed,
      // density x inverse speed, clamped to 0-100
      congestion_score: Math.round(Math.min(100, (count / 1.2) * (30 / speed)) * 10) / 10,
      by_type: Object.fromEntries(TYPES.map((t, k) => [t, Math.floor(count * weights[k]!)])),
    };
  });
}

// -------------------------------------------------------------- queries --

export const getCameras = () => CAMERAS;
export const getLinks = () => LINKS;

export function getTracks(o: { camera?: string; since?: string; limit?: number } = {}) {
  let rows = TRACKS;
  if (o.camera) rows = rows.filter((t) => t.camera_id === o.camera);
  if (o.since) rows = rows.filter((t) => t.entry_time >= o.since!);
  return rows.slice(0, o.limit ?? 100);
}

export function getTrajectories(o: { since?: string; limit?: number } = {}) {
  let rows = TRAJECTORIES;
  if (o.since) rows = rows.filter((t) => t.started_at >= o.since!);
  return rows.slice(0, o.limit ?? 50);
}

/**
 * A miss is a 200 with empty arrays, never a 404 — "no such plate" is a normal
 * answer, and a 404 makes the UI render an error state for it.
 */
export function search(raw: string): SearchResult {
  // Prefix, matching src/server/db.ts. An operator types what they saw, and
  // that is rarely the whole plate.
  const key = raw.replace(/[^A-Za-z0-9]/g, "").toUpperCase();
  const hit = (p: string | null) => p !== null && key !== "" && p.startsWith(key);
  const sightings = TRACKS
    .filter((t) => hit(t.plate_text))
    .map((t) => ({ camera_id: t.camera_id, ts: t.entry_time, confidence: t.plate_conf }))
    .sort((a, b) => (a.ts < b.ts ? -1 : 1));
  const last = sightings[sightings.length - 1];
  return {
    plate_text: key,
    trajectories: TRAJECTORIES.filter((t) => hit(t.plate_text)),
    sightings,
    last_seen: last ? { camera_id: last.camera_id, ts: last.ts } : null,
  };
}

// ----------------------------------------------------------------- devices --
//
// A phone paired as a camera. The mock cannot receive video, so a mock device
// never leaves "waiting" unless a URL is attached — which is honest: that is
// exactly what the operator sees live until something connects.

const DEVICES: Device[] = [
  {
    id: "1",
    code: "K7M2QP",
    camera_id: "PHONE1",
    label: "South gate phone",
    kind: "browser",
    source_url: null,
    status: "live",
    created_at: ago(600),
    paired_at: ago(540),
    last_frame_at: ago(2),
    pair_url: "/cam/K7M2QP",
  },
  {
    id: "2",
    code: "R4TXBN",
    camera_id: "PHONE2",
    label: null,
    kind: null,
    source_url: null,
    status: "waiting",
    created_at: ago(120),
    paired_at: null,
    last_frame_at: null,
    pair_url: "/cam/R4TXBN",
  },
];

const CODE_CHARS = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";

export const getDevices = () => DEVICES.filter((d) => d.status !== "revoked");

export const getDeviceByCode = (code: string) =>
  DEVICES.find((d) => d.code === code.toUpperCase() && d.status !== "revoked") ?? null;

export function createDevice(label: string | null): Device {
  const r = rng(DEVICES.length + 11);
  const code = Array.from({ length: 6 },
    () => CODE_CHARS[Math.floor(r() * CODE_CHARS.length)]).join("");
  const d: Device = {
    id: String(DEVICES.length + 1),
    code,
    camera_id: `PHONE${DEVICES.length + 1}`,
    label,
    kind: null,
    source_url: null,
    status: "waiting",
    created_at: iso(new Date()),
    paired_at: null,
    last_frame_at: null,
    pair_url: `/cam/${code}`,
  };
  DEVICES.unshift(d);
  return d;
}

export function pairDeviceUrl(code: string, url: string): Device | null {
  const d = getDeviceByCode(code);
  if (!d) return null;
  d.kind = "url";
  d.source_url = url;
  d.status = "live";
  d.paired_at = d.paired_at ?? iso(new Date());
  return d;
}

export function revokeDevice(id: string): boolean {
  const d = DEVICES.find((x) => x.id === id && x.status !== "revoked");
  if (!d) return false;
  d.status = "revoked";
  d.kind = null;
  d.source_url = null;
  return true;
}

// ----------------------------------------------------------------- uploads --
//
// The mock cannot decode video, and pretending otherwise would be a lie the UI
// then has to be built against. What it CAN do is give the upload views their
// shapes: one finished upload with results, so the page can be styled and
// reviewed without a worker, a database or a file on disk.
//
// A mock POST accepts the files and returns a `pending` upload that never
// progresses, which is exactly what happens live when the worker is not
// running — the honest failure to design for.

const UPLOADS: Upload[] = [
  {
    id: "1",
    created_at: ago(900),
    label: "North and east gate",
    status: "done",
    gap_seconds: 168,
    error: null,
    sources: [
      { camera_id: "UP1-1", filename: "north-gate.mp4", status: "done",
        error: null, tracks: 16, plates: 9 },
      { camera_id: "UP1-2", filename: "east-gate.mp4", status: "done",
        error: null, tracks: 14, plates: 7 },
    ],
  },
];

export const getUploads = (limit = 20) => UPLOADS.slice(0, Math.max(limit, 1));

export function getUpload(id: string): UploadResult | null {
  const upload = UPLOADS.find((u) => u.id === id);
  if (!upload) return null;
  const r = rng(7);
  const cams = upload.sources.map((s) => s.camera_id);
  const plates = Array.from({ length: 12 }, (_, i) => {
    const readable = r() > 0.3;
    return {
      camera_id: cams[i % cams.length]!,
      track_id: `T${String(i).padStart(3, "0")}`,
      plate_text: readable ? plate(r) : null,
      plate_conf: readable ? Math.round((0.82 + r() * 0.17) * 100) / 100 : null,
      vehicle_type: pick(r, TYPES),
      entry_time: ago(int(r, 60, 880)),
      speed_kmh: r() > 0.15 ? Math.round((20 + r() * 40) * 10) / 10 : null,
    };
  });
  // Only journeys BETWEEN the uploaded videos, which is the whole promise of
  // the results page.
  const trajectories = TRAJECTORIES.slice(0, 2).map((t, i) => ({
    ...t,
    id: `U${i}`,
    hops: t.hops.slice(0, 1).map((h) => ({ ...h, from_camera: cams[0]!, to_camera: cams[1]! })),
    path: t.path.slice(0, 2),
  }));
  return { upload, plates, trajectories };
}

export function createUpload(filenames: string[], gapSeconds: number | null,
                             label: string | null): Upload {
  const id = String(UPLOADS.length + 1);
  const upload: Upload = {
    id,
    created_at: iso(new Date()),
    label,
    status: "pending",
    gap_seconds: gapSeconds,
    error: null,
    sources: filenames.map((filename, i) => ({
      camera_id: `UP${id}-${i + 1}`,
      filename,
      status: "pending" as const,
      error: null,
      tracks: 0,
      plates: 0,
    })),
  };
  UPLOADS.unshift(upload);
  return upload;
}

export function getAnalytics(camera?: string): AnalyticsResponse {
  let rows;
  if (camera) {
    rows = series(camera);
  } else {
    const per = CAMERAS.map((c) => series(c.id));
    rows = per[0]!.map((_, i) => {
      const b = per.map((p) => p[i]!);
      return {
        ts: b[0]!.ts,
        vehicle_count: b.reduce((s, x) => s + x.vehicle_count, 0),
        avg_speed_kmh: Math.round((b.reduce((s, x) => s + x.avg_speed_kmh, 0) / b.length) * 10) / 10,
        congestion_score: Math.round((b.reduce((s, x) => s + x.congestion_score, 0) / b.length) * 10) / 10,
        by_type: Object.fromEntries(TYPES.map((t) => [t, b.reduce((s, x) => s + (x.by_type[t] ?? 0), 0)])),
      };
    });
  }
  return {
    camera_id: camera ?? null,
    series: rows,
    totals: {
      vehicle_count: rows.reduce((s, x) => s + x.vehicle_count, 0),
      avg_speed_kmh: Math.round((rows.reduce((s, x) => s + (x.avg_speed_kmh ?? 0), 0) / rows.length) * 10) / 10,
    },
  };
}

export const getAlerts = (acked?: boolean, limit = 200) =>
  (acked === undefined ? ALERTS : ALERTS.filter((a) => a.acked === acked))
    .slice(0, Math.min(Math.max(limit, 1), 500));

export function ackAlert(id: string) {
  const a = ALERTS.find((x) => x.id === id);
  if (a) a.acked = true;
  return { ok: true };
}

/** Canned WebSocket event loop. The real hub replaces this; shapes stay identical. */
export async function* mockEvents(): AsyncGenerator<import("@/contract").ServerMessage> {
  const r = rng(3);
  const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
  for (let tick = 1; ; tick++) {
    for (let i = 0; i < 5; i++) {  // ~5/s, matching the real 5 FPS budget
      const t = pick(r, TRACKS);
      const x = int(r, 80, 900), y = int(r, 60, 500);
      yield { type: "detection", data: {
        camera_id: t.camera_id, track_id: t.track_id,
        bbox: [x, y, x + int(r, 90, 180), y + int(r, 60, 130)],
        vehicle_type: t.vehicle_type, plate_text: t.plate_text,
        conf: t.plate_conf ?? Math.round((0.6 + r() * 0.3) * 100) / 100,
      }};
      await sleep(200);
    }
    if (tick % 5 === 0) {
      const trj = pick(r, TRAJECTORIES);
      const hop = trj.hops[0];
      if (hop) {
        yield { type: "match", data: { ...hop, trajectory_id: trj.id, plate_text: trj.plate_text } };
        yield { type: "trajectory_update", data: { id: trj.id, path: trj.path } };
      }
      yield { type: "analytics", data: {
        ts: iso(new Date()),
        per_camera: Object.fromEntries(CAMERAS.map((c) => [c.id, {
          vehicle_count: int(r, 40, 120),
          congestion_score: Math.round((10 + r() * 85) * 10) / 10,
        }])),
        city: { vehicle_count: int(r, 200, 400), avg_speed_kmh: Math.round((18 + r() * 20) * 10) / 10 },
      }};
    }
    if (tick % 20 === 0) yield { type: "alert", data: pick(r, ALERTS) };
  }
}
