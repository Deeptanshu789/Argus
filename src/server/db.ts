/**
 * The database layer. Every SQL statement in the application lives here.
 *
 * postgres.js tagged templates, not an ORM. TimescaleDB's hypertables and
 * `time_bucket()` are raw SQL either way, and an ORM schema file would be a
 * second definition of `db/schema.sql` that can silently drift from it — the
 * exact failure `src/contract.ts` exists to prevent on the API side.
 *
 * Every reader returns a shape from the contract. The API routes then validate
 * against the zod schema before responding, so a column rename cannot reach a
 * client as malformed JSON.
 */
import postgres from "postgres";
import type {
  Alert, AnalyticsResponse, Camera, CameraLink, Hop, MatchMethod,
  SearchResult, Track, Trajectory, VehicleType,
} from "@/contract";
import { bucketize, type CameraCalibration, type DetectionRow } from "@/server/analytics";

export const DATABASE_URL =
  process.env.DATABASE_URL ?? "postgresql://argus:argus@localhost:5432/argus";

/**
 * ponytail: one pool, module singleton. Next dev-mode module reloads would leak
 * pools without the globalThis pin, and a leaked pool exhausts Postgres
 * connections after a dozen edits — a confusing failure to debug mid-build.
 */
const g = globalThis as unknown as { __argus_sql?: postgres.Sql };
export const sql: postgres.Sql =
  g.__argus_sql ?? (g.__argus_sql = postgres(DATABASE_URL, {
    max: 10,
    idle_timeout: 20,
    connect_timeout: 10,
    onnotice: () => {},          // schema.sql is chatty about IF NOT EXISTS
  }));

/**
 * Postgres TIMESTAMPTZ comes back as a Date; the contract says ISO 8601 UTC.
 *
 * MILLISECONDS ARE KEPT. Detections arrive five per second, so truncating to
 * whole seconds collapses a track's frames onto one instant and estimateSpeed()
 * — which divides by the interval — returns null for every track.
 */
const iso = (d: Date | string | null): string | null =>
  d === null ? null : new Date(d).toISOString();

const isoReq = (d: Date | string): string => iso(d)!;

/** A camera is only as online as its last detection. Never a stored flag: a
 *  crashed sidecar cannot update a column saying it crashed. */
const DEGRADED_AFTER_S = 30;
const OFFLINE_AFTER_S = 300;

export async function ping(): Promise<boolean> {
  try {
    await sql`select 1`;
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------- topology --

export async function getCameras(): Promise<Camera[]> {
  const rows = await sql<{
    id: string; name: string; lat: number; lon: number;
    heading_deg: number | null; source_uri: string | null; age_s: number | null;
  }[]>`
    SELECT c.id, c.name, c.lat, c.lon, c.heading_deg, c.source_uri,
           EXTRACT(EPOCH FROM (now() - d.last_seen))::float AS age_s
      FROM cameras c
      LEFT JOIN LATERAL (
        SELECT max(ts) AS last_seen FROM detections WHERE camera_id = c.id
      ) d ON true
     ORDER BY c.id`;

  return rows.map((r) => ({
    id: r.id,
    name: r.name,
    lat: r.lat,
    lon: r.lon,
    heading_deg: r.heading_deg,
    status:
      r.age_s === null || r.age_s > OFFLINE_AFTER_S ? "offline"
      : r.age_s > DEGRADED_AFTER_S ? "degraded"
      : "online",
    stream_url: r.source_uri ?? "",
  }));
}

export async function getLinks(): Promise<CameraLink[]> {
  const rows = await sql<{
    from_camera: string; to_camera: string; distance_m: number; travel_time_s: number;
  }[]>`SELECT from_camera, to_camera, distance_m, travel_time_s
         FROM camera_links ORDER BY from_camera, to_camera`;
  return rows.map((r) => ({
    from: r.from_camera, to: r.to_camera,
    distance_m: r.distance_m, travel_time_s: r.travel_time_s,
  }));
}

export async function getCalibration(): Promise<Map<string, CameraCalibration>> {
  const rows = await sql<{
    id: string; metres_per_pixel: number | null; expected_flow_deg: number | null;
  }[]>`SELECT id, metres_per_pixel, expected_flow_deg FROM cameras`;
  const out = new Map<string, CameraCalibration>();
  for (const r of rows) {
    out.set(r.id, {
      // ponytail: 0.05 m/px is a placeholder for an UNCALIBRATED camera —
      // roughly a 3.5 m lane spanning 70 px. Speeds from it are indicative
      // only. Measure per camera and write metres_per_pixel to make them real.
      metersPerPixel: r.metres_per_pixel ?? 0.05,
      ...(r.expected_flow_deg === null ? {} : { expectedFlowDeg: r.expected_flow_deg }),
    });
  }
  return out;
}

// ------------------------------------------------------------------ tracks --

interface TrackRow {
  id: string; camera_id: string; track_id: string;
  plate_text: string | null; plate_conf: number | null;
  vehicle_type: string | null; color: string | null;
  entry_time: Date; exit_time: Date | null;
}

const toTrack = (r: TrackRow): Track => ({
  id: String(r.id),
  camera_id: r.camera_id,
  track_id: r.track_id,
  plate_text: r.plate_text,
  plate_conf: r.plate_conf,
  vehicle_type: (r.vehicle_type ?? "car") as VehicleType,
  color: r.color,
  entry_time: isoReq(r.entry_time),
  exit_time: iso(r.exit_time),
});

export async function getTracks(
  o: { camera?: string; since?: string; limit?: number } = {},
): Promise<Track[]> {
  const limit = Math.min(Math.max(o.limit ?? 100, 1), 1000);
  const rows = await sql<TrackRow[]>`
    SELECT id, camera_id, track_id, plate_text, plate_conf, vehicle_type, color,
           entry_time, exit_time
      FROM tracks
     WHERE (${o.camera ?? null}::text IS NULL OR camera_id = ${o.camera ?? null})
       AND (${o.since ?? null}::timestamptz IS NULL
            OR entry_time >= ${o.since ?? null}::timestamptz)
     ORDER BY entry_time DESC
     LIMIT ${limit}`;
  return rows.map(toTrack);
}

/**
 * Candidate previous-camera tracks for the association engine. Bounded by the
 * window, because anything older than the slowest link cannot be a match, and
 * scanning further is pure cost.
 */
export async function candidateTracks(
  cameraId: string, before: string, windowSeconds: number,
) {
  return sql<{
    id: string; camera_id: string; track_id: string;
    plate_text: string | null; plate_conf: number | null;
    embedding: number[] | null; color_hist: number[] | null;
    entry_time: Date; exit_time: Date | null;
  }[]>`
    SELECT id, camera_id, track_id, plate_text, plate_conf, embedding, color_hist,
           entry_time, exit_time
      FROM tracks
     WHERE camera_id <> ${cameraId}
       AND exit_time IS NOT NULL
       AND exit_time < ${before}::timestamptz
       AND exit_time > ${before}::timestamptz - make_interval(secs => ${windowSeconds})
     ORDER BY exit_time DESC
     LIMIT 500`;
}

// ------------------------------------------------------------- trajectories --

/**
 * A trajectory is a chain of track ids. Rebuilding the path means joining each
 * one back to its camera's coordinates, in chain order — which `array_position`
 * preserves and a plain `IN` list does not.
 */
async function hydrate(rows: {
  id: string; plate_text: string | null; track_ids: string[];
  started_at: Date; ended_at: Date | null;
}[]): Promise<Trajectory[]> {
  if (!rows.length) return [];
  const allIds = [...new Set(rows.flatMap((r) => r.track_ids.map(String)))];

  const legs = await sql<{
    id: string; camera_id: string; vehicle_type: string | null;
    entry_time: Date; lat: number; lon: number;
  }[]>`
    SELECT t.id, t.camera_id, t.vehicle_type, t.entry_time, c.lat, c.lon
      FROM tracks t JOIN cameras c ON c.id = t.camera_id
     WHERE t.id = ANY(${allIds}::bigint[])`;
  const legById = new Map(legs.map((l) => [String(l.id), l]));

  const hops = await sql<{
    from_track: string; to_track: string; method: string;
    confidence: number; travel_time_s: number | null;
  }[]>`
    SELECT from_track, to_track, method, confidence, travel_time_s
      FROM matches
     WHERE from_track = ANY(${allIds}::bigint[])
       AND to_track   = ANY(${allIds}::bigint[])`;
  const hopByPair = new Map(hops.map((h) => [`${h.from_track}->${h.to_track}`, h]));

  return rows.map((r) => {
    const chain = r.track_ids.map(String);
    const start = r.started_at.getTime();
    const path: [number, number, number][] = [];
    const out: Hop[] = [];
    let vehicleType: VehicleType = "car";

    for (let i = 0; i < chain.length; i++) {
      const leg = legById.get(chain[i]!);
      if (!leg) continue;
      if (i === 0 && leg.vehicle_type) vehicleType = leg.vehicle_type as VehicleType;
      // [lon, lat, seconds_since_start] — the tuple deck.gl TripsLayer consumes.
      path.push([leg.lon, leg.lat, Math.round((leg.entry_time.getTime() - start) / 1000)]);

      const h = hopByPair.get(`${chain[i - 1]}->${chain[i]}`);
      if (i > 0 && h) {
        out.push({
          from_camera: legById.get(chain[i - 1]!)?.camera_id ?? "",
          to_camera: leg.camera_id,
          method: h.method as MatchMethod,
          confidence: h.confidence,
          travel_time_s: h.travel_time_s ?? 0,
        });
      }
    }

    return {
      id: String(r.id),
      plate_text: r.plate_text,
      vehicle_type: vehicleType,
      started_at: isoReq(r.started_at),
      ended_at: iso(r.ended_at),
      path,
      hops: out,
    };
  });
}

export async function getTrajectories(
  o: { since?: string; limit?: number } = {},
): Promise<Trajectory[]> {
  const limit = Math.min(Math.max(o.limit ?? 50, 1), 500);
  const rows = await sql<{
    id: string; plate_text: string | null; track_ids: string[];
    started_at: Date; ended_at: Date | null;
  }[]>`
    SELECT id, plate_text, track_ids, started_at, ended_at
      FROM trajectories
     WHERE (${o.since ?? null}::timestamptz IS NULL
            OR started_at >= ${o.since ?? null}::timestamptz)
     ORDER BY started_at DESC
     LIMIT ${limit}`;
  return hydrate(rows);
}

// ------------------------------------------------------------------ search --

/**
 * Plate search. A miss is a 200 with empty arrays, never a 404 — "we searched
 * and found nothing" is a result, and the UI renders it differently from an
 * error.
 */
export async function search(raw: string): Promise<SearchResult> {
  const plate = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
  const empty: SearchResult = {
    plate_text: plate, trajectories: [], sightings: [], last_seen: null,
  };
  if (!plate) return empty;

  const sightings = await sql<{ camera_id: string; ts: Date; confidence: number | null }[]>`
    SELECT camera_id, entry_time AS ts, plate_conf AS confidence
      FROM tracks
     WHERE plate_text = ${plate}
     ORDER BY entry_time DESC
     LIMIT 200`;

  const trajRows = await sql<{
    id: string; plate_text: string | null; track_ids: string[];
    started_at: Date; ended_at: Date | null;
  }[]>`
    SELECT id, plate_text, track_ids, started_at, ended_at
      FROM trajectories
     WHERE plate_text = ${plate}
     ORDER BY started_at DESC
     LIMIT 20`;

  const first = sightings[0];
  return {
    plate_text: plate,
    trajectories: await hydrate(trajRows),
    sightings: sightings.map((s) => ({
      camera_id: s.camera_id, ts: isoReq(s.ts), confidence: s.confidence,
    })),
    last_seen: first ? { camera_id: first.camera_id, ts: isoReq(first.ts) } : null,
  };
}

// --------------------------------------------------------------- analytics --

/**
 * Buckets are computed from raw detections through `bucketize()` rather than
 * read from the `analytics` rollup table. One implementation, already
 * selfchecked, and it cannot disagree with the rollup because there is nothing
 * to disagree with. The rollup becomes worth reading when a query over raw
 * detections gets slow — measure before switching.
 */
export async function getAnalytics(
  camera?: string, hours = 6,
): Promise<AnalyticsResponse> {
  const rows = await sql<{
    ts: Date; camera_id: string; track_id: string;
    bbox: number[]; vehicle_type: string | null;
  }[]>`
    SELECT d.ts, d.camera_id, d.track_id, d.bbox, t.vehicle_type
      FROM detections d
      LEFT JOIN tracks t
        ON t.camera_id = d.camera_id AND t.track_id = d.track_id
     WHERE d.ts > now() - make_interval(hours => ${hours})
       AND (${camera ?? null}::text IS NULL OR d.camera_id = ${camera ?? null})
     ORDER BY d.ts`;

  const cal = await getCalibration();
  const detections: DetectionRow[] = rows.map((r) => ({
    ts: isoReq(r.ts),
    camera_id: r.camera_id,
    // track_id is only unique WITHIN a camera. Without the prefix, CAM1's T7
    // and CAM2's T7 merge into one track and the vehicle count halves.
    track_id: `${r.camera_id}:${r.track_id}`,
    bbox: [r.bbox[0] ?? 0, r.bbox[1] ?? 0, r.bbox[2] ?? 0, r.bbox[3] ?? 0],
    vehicle_type: (r.vehicle_type ?? "car") as VehicleType,
  }));

  // One camera means one calibration; city-wide has no single metres-per-pixel,
  // so use the mean rather than pretending one camera's ratio holds everywhere.
  const cals = camera
    ? [cal.get(camera) ?? { metersPerPixel: 0.05 }]
    : [...cal.values()];
  const mpp = cals.length
    ? cals.reduce((s, c) => s + c.metersPerPixel, 0) / cals.length
    : 0.05;

  const series = bucketize(detections, { metersPerPixel: mpp });
  const speeds = series.map((b) => b.avg_speed_kmh).filter((s): s is number => s !== null);

  return {
    camera_id: camera ?? null,
    series,
    totals: {
      vehicle_count: series.reduce((s, b) => s + b.vehicle_count, 0),
      avg_speed_kmh: speeds.length
        ? Math.round((speeds.reduce((a, b) => a + b, 0) / speeds.length) * 10) / 10
        : null,
    },
  };
}

// ------------------------------------------------------------------ alerts --

export async function getAlerts(acked?: boolean): Promise<Alert[]> {
  const rows = await sql<{
    id: string; ts: Date; camera_id: string | null; kind: string; severity: string;
    track_id: string | null; plate_text: string | null; detail: string | null; acked: boolean;
  }[]>`
    SELECT id, ts, camera_id, kind, severity, track_id, plate_text, detail, acked
      FROM alerts
     WHERE (${acked ?? null}::boolean IS NULL OR acked = ${acked ?? null})
     ORDER BY ts DESC
     LIMIT 200`;
  return rows.map((r) => ({
    id: String(r.id),
    ts: isoReq(r.ts),
    camera_id: r.camera_id,
    kind: r.kind as Alert["kind"],
    severity: r.severity as Alert["severity"],
    track_id: r.track_id,
    plate_text: r.plate_text,
    detail: r.detail ?? "",
    acked: r.acked,
  }));
}

export async function ackAlert(id: string): Promise<Alert | null> {
  if (!/^\d+$/.test(id)) return null;
  const rows = await sql<{
    id: string; ts: Date; camera_id: string | null; kind: string; severity: string;
    track_id: string | null; plate_text: string | null; detail: string | null; acked: boolean;
  }[]>`
    UPDATE alerts SET acked = true WHERE id = ${id}::bigint
    RETURNING id, ts, camera_id, kind, severity, track_id, plate_text, detail, acked`;
  const r = rows[0];
  if (!r) return null;
  return {
    id: String(r.id), ts: isoReq(r.ts), camera_id: r.camera_id,
    kind: r.kind as Alert["kind"], severity: r.severity as Alert["severity"],
    track_id: r.track_id, plate_text: r.plate_text,
    detail: r.detail ?? "", acked: r.acked,
  };
}

// ------------------------------------------------------------------ writes --
// Called by worker/ingest.ts only. The API never writes.

export async function insertDetections(
  batch: readonly {
    ts: string; camera_id: string; track_id: string;
    bbox: [number, number, number, number]; conf: number;
  }[],
): Promise<void> {
  if (!batch.length) return;
  await sql`INSERT INTO detections ${sql(batch.map((d) => ({
    ts: d.ts, camera_id: d.camera_id, track_id: d.track_id,
    bbox: d.bbox as unknown as number[], conf: d.conf,
  })), "ts", "camera_id", "track_id", "bbox", "conf")}`;
}

/**
 * Upsert, keyed on (camera_id, track_id): a sidecar restart can re-emit a track
 * that is already stored, and a plain INSERT would abort the batch on the
 * unique constraint rather than updating the row.
 */
export async function upsertTrack(t: {
  camera_id: string; track_id: string; vehicle_type: string; color: string | null;
  plate_text: string | null; plate_conf: number | null;
  embedding: readonly number[]; color_hist: readonly number[];
  entry_time: string; exit_time: string | null;
}): Promise<string> {
  const rows = await sql<{ id: string }[]>`
    INSERT INTO tracks (camera_id, track_id, vehicle_type, color, plate_text,
                        plate_conf, embedding, color_hist, entry_time, exit_time)
    VALUES (${t.camera_id}, ${t.track_id}, ${t.vehicle_type}, ${t.color},
            ${t.plate_text}, ${t.plate_conf}, ${t.embedding as number[]},
            ${t.color_hist as number[]}, ${t.entry_time}::timestamptz,
            ${t.exit_time}::timestamptz)
    ON CONFLICT (camera_id, track_id) DO UPDATE SET
      plate_text = COALESCE(EXCLUDED.plate_text, tracks.plate_text),
      plate_conf = COALESCE(EXCLUDED.plate_conf, tracks.plate_conf),
      embedding  = EXCLUDED.embedding,
      color_hist = EXCLUDED.color_hist,
      exit_time  = EXCLUDED.exit_time
    RETURNING id`;
  return String(rows[0]!.id);
}

export async function insertMatch(m: {
  from_track: string; to_track: string; method: string;
  confidence: number; travel_time_s: number;
}): Promise<void> {
  await sql`
    INSERT INTO matches (from_track, to_track, method, confidence, travel_time_s)
    VALUES (${m.from_track}::bigint, ${m.to_track}::bigint, ${m.method},
            ${m.confidence}, ${m.travel_time_s})
    ON CONFLICT (from_track, to_track) DO NOTHING`;
}

/**
 * Extend the trajectory that already ends at `fromTrack`, or open a new one.
 *
 * Appending to the existing chain is what makes a journey across three cameras
 * ONE trajectory rather than two disconnected hops — the property the whole
 * cross-camera story rests on.
 */
export async function extendTrajectory(
  fromTrack: string, toTrack: string,
  plate: string | null, startedAt: string, endedAt: string | null,
): Promise<{ id: string; track_ids: string[] }> {
  // Idempotence first. A sidecar restart re-emits the same track_closed, the
  // same match is re-derived, and this function is called again. `matches` is
  // protected by its unique constraint; without this check a trajectory
  // [1,2] would not be found by a lookup for one ENDING at 1, and a second
  // identical [1,2] would be created on every replay.
  const already = await sql<{ id: string; track_ids: string[] }[]>`
    SELECT id, track_ids FROM trajectories
     WHERE array_position(track_ids, ${fromTrack}::bigint) IS NOT NULL
       AND track_ids[array_position(track_ids, ${fromTrack}::bigint) + 1]
           = ${toTrack}::bigint
     LIMIT 1`;
  if (already[0]) {
    return { id: String(already[0].id), track_ids: already[0].track_ids.map(String) };
  }

  const found = await sql<{ id: string; track_ids: string[] }[]>`
    SELECT id, track_ids FROM trajectories
     WHERE track_ids[array_length(track_ids, 1)] = ${fromTrack}::bigint
     ORDER BY updated_at DESC LIMIT 1`;

  const existing = found[0];
  if (existing) {
    const rows = await sql<{ id: string; track_ids: string[] }[]>`
      UPDATE trajectories
         SET track_ids = array_append(track_ids, ${toTrack}::bigint),
             plate_text = COALESCE(trajectories.plate_text, ${plate}),
             ended_at = ${endedAt}::timestamptz,
             updated_at = now()
       WHERE id = ${existing.id}::bigint
      RETURNING id, track_ids`;
    return { id: String(rows[0]!.id), track_ids: rows[0]!.track_ids.map(String) };
  }

  const rows = await sql<{ id: string; track_ids: string[] }[]>`
    INSERT INTO trajectories (plate_text, track_ids, started_at, ended_at)
    VALUES (${plate}, ARRAY[${fromTrack}::bigint, ${toTrack}::bigint],
            ${startedAt}::timestamptz, ${endedAt}::timestamptz)
    RETURNING id, track_ids`;
  return { id: String(rows[0]!.id), track_ids: rows[0]!.track_ids.map(String) };
}

export async function insertAlert(a: {
  camera_id: string | null; kind: string; severity: string;
  track_id: string | null; plate_text: string | null; detail: string;
}): Promise<Alert> {
  const rows = await sql<{ id: string; ts: Date }[]>`
    INSERT INTO alerts (camera_id, kind, severity, track_id, plate_text, detail)
    VALUES (${a.camera_id}, ${a.kind}, ${a.severity}, ${a.track_id},
            ${a.plate_text}, ${a.detail})
    RETURNING id, ts`;
  return {
    id: String(rows[0]!.id), ts: isoReq(rows[0]!.ts), camera_id: a.camera_id,
    kind: a.kind as Alert["kind"], severity: a.severity as Alert["severity"],
    track_id: a.track_id, plate_text: a.plate_text, detail: a.detail, acked: false,
  };
}

/** Detections for one track, for the anomaly checks. */
export async function trackDetections(
  cameraId: string, trackId: string,
): Promise<DetectionRow[]> {
  const rows = await sql<{
    ts: Date; bbox: number[]; vehicle_type: string | null;
  }[]>`
    SELECT d.ts, d.bbox, t.vehicle_type
      FROM detections d
      LEFT JOIN tracks t ON t.camera_id = d.camera_id AND t.track_id = d.track_id
     WHERE d.camera_id = ${cameraId} AND d.track_id = ${trackId}
     ORDER BY d.ts`;
  return rows.map((r) => ({
    ts: isoReq(r.ts),
    camera_id: cameraId,
    track_id: trackId,
    bbox: [r.bbox[0] ?? 0, r.bbox[1] ?? 0, r.bbox[2] ?? 0, r.bbox[3] ?? 0],
    vehicle_type: (r.vehicle_type ?? "car") as VehicleType,
  }));
}

/** Recent detections for one camera, for the periodic rollup and spike sweep. */
export async function trackDetections_recent(
  cameraId: string, seconds: number,
): Promise<DetectionRow[]> {
  const rows = await sql<{
    ts: Date; track_id: string; bbox: number[]; vehicle_type: string | null;
  }[]>`
    SELECT d.ts, d.track_id, d.bbox, t.vehicle_type
      FROM detections d
      LEFT JOIN tracks t ON t.camera_id = d.camera_id AND t.track_id = d.track_id
     WHERE d.camera_id = ${cameraId}
       AND d.ts > now() - make_interval(secs => ${seconds})
     ORDER BY d.ts`;
  return rows.map((r) => ({
    ts: isoReq(r.ts),
    camera_id: cameraId,
    track_id: r.track_id,
    bbox: [r.bbox[0] ?? 0, r.bbox[1] ?? 0, r.bbox[2] ?? 0, r.bbox[3] ?? 0],
    vehicle_type: (r.vehicle_type ?? "car") as VehicleType,
  }));
}
