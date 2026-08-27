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
  Device, SearchResult, Track, Trajectory, Upload, UploadResult, VehicleType,
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
     WHERE NOT c.is_upload
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

/**
 * The road graph.
 *
 * `includeUploads` defaults to false because /api/cameras/links must describe
 * the same world /api/cameras does: a link naming a camera the caller cannot
 * see is a graph with a dangling edge, and the map would draw a line to
 * nowhere. The WORKER passes true — layer 3 needs an upload's own links to
 * reason about journeys between the operator's videos.
 */
export async function getLinks(
  o: { includeUploads?: boolean } = {},
): Promise<CameraLink[]> {
  const rows = await sql<{
    from_camera: string; to_camera: string; distance_m: number; travel_time_s: number;
  }[]>`SELECT from_camera, to_camera, distance_m, travel_time_s
         FROM camera_links
        WHERE ${o.includeUploads ?? false}
           OR (from_camera NOT IN (SELECT id FROM cameras WHERE is_upload)
              AND to_camera NOT IN (SELECT id FROM cameras WHERE is_upload))
        ORDER BY from_camera, to_camera`;
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
  speed_kmh: number | null;
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
  speed_kmh: r.speed_kmh,
});

export async function getTracks(
  o: { camera?: string; since?: string; limit?: number } = {},
): Promise<Track[]> {
  const limit = Math.min(Math.max(o.limit ?? 100, 1), 1000);
  const rows = await sql<TrackRow[]>`
    SELECT id, camera_id, track_id, plate_text, plate_conf, vehicle_type, color,
           entry_time, exit_time, speed_kmh
      FROM tracks
     WHERE (${o.camera ?? null}::text IS NULL OR camera_id = ${o.camera ?? null})
       -- Uploaded footage has its own results page and must not appear in the
       -- live city view. Asking for one of its cameras by name still works.
       AND (${o.camera ?? null}::text IS NOT NULL
            OR camera_id NOT IN (SELECT id FROM cameras WHERE is_upload))
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
 *
 * Bounded by the CAMERA SET too. An uploaded video is matched only against the
 * other videos in its own upload, and a live camera only against other live
 * cameras. Without this, a vehicle in an operator's clip matches one on a demo
 * camera -- the same plate genuinely appears in both -- and the upload's
 * results fill with journeys through cameras that have nothing to do with the
 * footage they sent. `IS NOT DISTINCT FROM` is what makes NULL (a live camera)
 * match NULL rather than matching nothing.
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
       AND (SELECT upload_id FROM upload_sources u WHERE u.camera_id = tracks.camera_id)
           IS NOT DISTINCT FROM
           (SELECT upload_id FROM upload_sources u WHERE u.camera_id = ${cameraId})
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
      FROM trajectories tr
     WHERE (${o.since ?? null}::timestamptz IS NULL
            OR started_at >= ${o.since ?? null}::timestamptz)
       -- Same reason as getTracks: an uploaded journey belongs to its upload's
       -- page, not to the live map.
       AND NOT EXISTS (
         SELECT 1 FROM tracks t JOIN cameras c ON c.id = t.camera_id
          WHERE t.id = ANY(tr.track_ids) AND c.is_upload)
     ORDER BY started_at DESC
     LIMIT ${limit}`;
  return hydrate(rows);
}

// ------------------------------------------------------------------ search --

/**
 * Plate search. A miss is a 200 with empty arrays, never a 404 — "we searched
 * and found nothing" is a result, and the UI renders it differently from an
 * error.
 *
 * PREFIX match, not exact. The real query is "I saw HR26, something, a white
 * car" — an operator rarely has all ten characters, and OCR itself drops one
 * often enough that an exact match would hide the vehicle it did read. A full
 * plate is a prefix of itself, so exact lookups still work unchanged.
 *
 * ponytail: prefix only, so a query cannot start mid-plate. Postgres uses the
 * plate index for `LIKE 'ABC%'` and cannot for `LIKE '%ABC%'`; switch to
 * pg_trgm if searching by the series letters alone turns out to matter.
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
     WHERE plate_text LIKE ${plate + "%"}
     ORDER BY entry_time DESC
     LIMIT 200`;

  const trajRows = await sql<{
    id: string; plate_text: string | null; track_ids: string[];
    started_at: Date; ended_at: Date | null;
  }[]>`
    SELECT id, plate_text, track_ids, started_at, ended_at
      FROM trajectories
     WHERE plate_text LIKE ${plate + "%"}
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

// ----------------------------------------------------------------- uploads --

/**
 * Where uploaded cameras are drawn.
 *
 * An uploaded video has no location, but `cameras.lat/lon` are NOT NULL and the
 * map draws whatever it is given. Placing them on a short arc near the city
 * centre keeps every existing view working and makes it obvious at a glance
 * that these are not surveyed installations.
 *
 * ponytail: fixed centre, fixed spacing. Add a location field to the upload
 * form if anyone ever needs uploaded footage plotted where it was filmed.
 */
const UPLOAD_ORIGIN = { lat: 12.9600, lon: 77.6300 };
const UPLOAD_SPACING_DEG = 0.004;

export async function createUpload(o: {
  label: string | null;
  gapSeconds: number | null;
  files: readonly { filename: string; path: string }[];
}): Promise<string> {
  if (!o.files.length) throw new Error("an upload needs at least one file");

  const [up] = await sql<{ id: string }[]>`
    INSERT INTO uploads (label, gap_seconds) VALUES (${o.label}, ${o.gapSeconds})
    RETURNING id`;
  const uploadId = String(up!.id);
  const cameraIds: string[] = [];

  for (const [i, f] of o.files.entries()) {
    const cameraId = `UP${uploadId}-${i + 1}`;
    cameraIds.push(cameraId);
    await sql`
      INSERT INTO cameras (id, name, lat, lon, heading_deg, source_uri, is_upload)
      VALUES (${cameraId}, ${f.filename},
              ${UPLOAD_ORIGIN.lat + i * UPLOAD_SPACING_DEG},
              ${UPLOAD_ORIGIN.lon}, NULL, ${f.path}, true)
      ON CONFLICT (id) DO NOTHING`;
    await sql`
      INSERT INTO upload_sources (upload_id, camera_id, filename, path)
      VALUES (${uploadId}::bigint, ${cameraId}, ${f.filename}, ${f.path})`;
  }

  // Only when the operator told us how far apart the cameras are. Inventing a
  // travel time would be worse than having none: layer 3 VETOES a journey it
  // believes impossible, so a wrong number silently deletes real matches, while
  // a missing one merely makes it abstain.
  if (o.gapSeconds !== null && cameraIds.length > 1) {
    for (const from of cameraIds) {
      for (const to of cameraIds) {
        if (from === to) continue;
        await sql`
          INSERT INTO camera_links (from_camera, to_camera, distance_m, travel_time_s)
          VALUES (${from}, ${to}, ${Math.max(1, o.gapSeconds * 10)}, ${o.gapSeconds})
          ON CONFLICT (from_camera, to_camera) DO NOTHING`;
      }
    }
  }
  return uploadId;
}

interface UploadRow {
  id: string; created_at: Date; label: string | null;
  status: string; gap_seconds: number | null; error: string | null;
}

async function withSources(rows: UploadRow[]): Promise<Upload[]> {
  if (!rows.length) return [];
  const ids = rows.map((r) => String(r.id));
  const sources = await sql<{
    upload_id: string; camera_id: string; filename: string;
    status: string; error: string | null; tracks: number; plates: number;
  }[]>`
    SELECT s.upload_id, s.camera_id, s.filename, s.status, s.error,
           count(t.id)::int AS tracks,
           count(t.plate_text)::int AS plates
      FROM upload_sources s
      LEFT JOIN tracks t ON t.camera_id = s.camera_id
     WHERE s.upload_id = ANY(${ids}::bigint[])
     GROUP BY s.id, s.upload_id, s.camera_id, s.filename, s.status, s.error
     ORDER BY s.id`;

  return rows.map((r) => ({
    id: String(r.id),
    created_at: isoReq(r.created_at),
    label: r.label,
    status: r.status as Upload["status"],
    gap_seconds: r.gap_seconds,
    error: r.error,
    sources: sources
      .filter((s) => String(s.upload_id) === String(r.id))
      .map((s) => ({
        camera_id: s.camera_id,
        filename: s.filename,
        status: s.status as Upload["status"],
        error: s.error,
        tracks: s.tracks,
        plates: s.plates,
      })),
  }));
}

export async function getUploads(limit = 20): Promise<Upload[]> {
  const n = Math.min(Math.max(limit, 1), 100);
  const rows = await sql<UploadRow[]>`
    SELECT id, created_at, label, status, gap_seconds, error
      FROM uploads ORDER BY created_at DESC LIMIT ${n}`;
  return withSources(rows);
}

export async function getUpload(id: string): Promise<Upload | null> {
  const rows = await sql<UploadRow[]>`
    SELECT id, created_at, label, status, gap_seconds, error
      FROM uploads WHERE id = ${id}::bigint`;
  return (await withSources(rows))[0] ?? null;
}

/**
 * Everything the results page shows for one upload: the vehicles read out of
 * its videos, and any journey Module C stitched between them.
 */
export async function getUploadResult(id: string): Promise<UploadResult | null> {
  const upload = await getUpload(id);
  if (!upload) return null;
  const cams = upload.sources.map((s) => s.camera_id);
  if (!cams.length) return { upload, plates: [], trajectories: [] };

  const plates = await sql<{
    camera_id: string; track_id: string; plate_text: string | null;
    plate_conf: number | null; vehicle_type: string | null;
    entry_time: Date; speed_kmh: number | null;
  }[]>`
    SELECT camera_id, track_id, plate_text, plate_conf, vehicle_type,
           entry_time, speed_kmh
      FROM tracks
     WHERE camera_id = ANY(${cams})
     -- Vehicles whose plate was read first: that is what the page is for.
     ORDER BY (plate_text IS NULL), entry_time DESC
     LIMIT 500`;

  const trajRows = await sql<{
    id: string; plate_text: string | null; track_ids: string[];
    started_at: Date; ended_at: Date | null;
  }[]>`
    SELECT tr.id, tr.plate_text, tr.track_ids, tr.started_at, tr.ended_at
      FROM trajectories tr
     WHERE EXISTS (SELECT 1 FROM tracks t
                    WHERE t.id = ANY(tr.track_ids) AND t.camera_id = ANY(${cams}))
       -- EVERY leg must belong to this upload. A journey that also passes
       -- through a live camera is not this upload's journey, and showing it
       -- would put footage the operator never sent on their results page.
       AND NOT EXISTS (SELECT 1 FROM tracks t
                        WHERE t.id = ANY(tr.track_ids) AND NOT (t.camera_id = ANY(${cams})))
     ORDER BY tr.started_at DESC
     LIMIT 100`;

  return {
    upload,
    plates: plates.map((p) => ({
      camera_id: p.camera_id,
      track_id: p.track_id,
      plate_text: p.plate_text,
      plate_conf: p.plate_conf,
      vehicle_type: (p.vehicle_type ?? "car") as VehicleType,
      entry_time: isoReq(p.entry_time),
      speed_kmh: p.speed_kmh,
    })),
    trajectories: await hydrate(trajRows),
  };
}

/**
 * Take the oldest pending upload and mark it running, atomically.
 *
 * SKIP LOCKED so two workers never claim the same one. There is only one worker
 * today, but a queue that quietly double-processes is a bad thing to discover
 * during a demo, and the clause costs nothing.
 */
export async function claimPendingUpload(): Promise<Upload | null> {
  const rows = await sql<UploadRow[]>`
    UPDATE uploads SET status = 'running'
     WHERE id = (SELECT id FROM uploads WHERE status = 'pending'
                  ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED)
    RETURNING id, created_at, label, status, gap_seconds, error`;
  return (await withSources(rows))[0] ?? null;
}

/**
 * The on-disk locations of an upload's files.
 *
 * Separate from `getUpload` on purpose: `UploadSource` in the contract has no
 * `path`, because a browser has no business learning where files sit on the
 * server's filesystem. The worker does need it, and the worker is not a
 * browser.
 */
export async function uploadSourcePaths(
  uploadId: string,
): Promise<{ camera_id: string; filename: string; path: string }[]> {
  return sql<{ camera_id: string; filename: string; path: string }[]>`
    SELECT camera_id, filename, path FROM upload_sources
     WHERE upload_id = ${uploadId}::bigint ORDER BY id`;
}

export async function setUploadStatus(
  id: string, status: Upload["status"], error: string | null = null,
): Promise<void> {
  await sql`UPDATE uploads SET status = ${status}, error = ${error}
             WHERE id = ${id}::bigint`;
}

/**
 * Stop an upload the operator no longer wants scanned.
 *
 * The web process cannot kill the sidecars -- they are children of the WORKER,
 * which is a separate process by design. So cancelling is a flag in the
 * database and the worker acts on it; `false` here means the upload had already
 * finished and there was nothing left to stop.
 */
export async function cancelUpload(id: string): Promise<boolean> {
  const rows = await sql<{ id: string }[]>`
    UPDATE uploads SET status = 'cancelled'
     WHERE id = ${id}::bigint AND status IN ('pending', 'running')
    RETURNING id`;
  if (!rows.length) return false;
  await sql`UPDATE upload_sources SET status = 'cancelled'
             WHERE upload_id = ${id}::bigint AND status IN ('pending', 'running')`;
  return true;
}

/** What the worker polls while an upload runs, to notice a cancel request. */
export async function uploadCancelled(id: string): Promise<boolean> {
  const rows = await sql<{ status: string }[]>`
    SELECT status FROM uploads WHERE id = ${id}::bigint`;
  return rows[0]?.status === "cancelled";
}

export async function setUploadSourceStatus(
  cameraId: string, status: Upload["status"], error: string | null = null,
): Promise<void> {
  await sql`UPDATE upload_sources SET status = ${status}, error = ${error}
             WHERE camera_id = ${cameraId}`;
}

/**
 * Uploads older than `days`, oldest first, with their files.
 *
 * Video is the only thing here that grows without bound: a few minutes of
 * 1080p is hundreds of megabytes, nothing deletes it, and a VPS disk fills
 * silently. The DETECTIONS are what the system is for — they are rows, they
 * are small, and they stay. It is the source video that has served its purpose
 * once it has been decoded.
 */
export async function uploadsOlderThan(
  days: number, limit = 100,
): Promise<{ id: string; label: string | null; created_at: Date;
             files: string[]; cameras: string[] }[]> {
  return sql<{ id: string; label: string | null; created_at: Date;
               files: string[]; cameras: string[] }[]>`
    SELECT u.id, u.label, u.created_at,
           coalesce(array_agg(s.path) FILTER (WHERE s.path IS NOT NULL), '{}') AS files,
           coalesce(array_agg(s.camera_id) FILTER (WHERE s.camera_id IS NOT NULL), '{}') AS cameras
      FROM uploads u
      LEFT JOIN upload_sources s ON s.upload_id = u.id
     WHERE u.created_at < now() - make_interval(days => ${days})
       AND u.status IN ('done', 'error')
     GROUP BY u.id
     ORDER BY u.created_at
     LIMIT ${limit}`;
}

/** Uploads interrupted by a worker restart. Left running, they never finish. */
export async function requeueStaleUploads(): Promise<number> {
  const rows = await sql`
    UPDATE uploads SET status = 'pending' WHERE status = 'running' RETURNING id`;
  return rows.length;
}

// ----------------------------------------------------------------- devices --

/**
 * Pairing-code alphabet. No 0/O, no 1/I/L: the code is read off a laptop screen
 * and typed on a phone, and those are the pairs people get wrong.
 */
const CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";
const CODE_LENGTH = 6;

const newCode = () => Array.from(
  { length: CODE_LENGTH },
  () => CODE_ALPHABET[Math.floor(Math.random() * CODE_ALPHABET.length)],
).join("");

/** Where devices are drawn until someone surveys them. See UPLOAD_ORIGIN. */
const DEVICE_ORIGIN = { lat: 12.9700, lon: 77.5900 };
const DEVICE_SPACING_DEG = 0.003;

interface DeviceRow {
  id: string; code: string; camera_id: string; label: string | null;
  kind: string | null; source_url: string | null;
  created_at: Date; paired_at: Date | null; last_frame_at: Date | null;
  revoked: boolean;
}

/**
 * `pair_url` is built by the caller from the REQUEST host, not from a config
 * value: the whole point is that a phone must reach it, and the server has no
 * way of knowing which of its addresses the operator's laptop is being browsed
 * on. Passing "" is correct for callers with no request in hand.
 */
function toDevice(r: DeviceRow, origin = ""): Device {
  const age = r.last_frame_at ? Date.now() - r.last_frame_at.getTime() : null;
  // "live" means SOMETHING IS ARRIVING, for both kinds. A browser device is
  // touched by the frames it pushes; a URL device is touched by the worker when
  // its sidecar opens the stream. Reporting a URL device as live merely because
  // a URL was typed would show a green camera for an app that was closed hours
  // ago, which is worse than showing nothing.
  const status: Device["status"] =
    r.revoked ? "revoked"
    : age !== null && age < DEVICE_STALE_MS ? "live"
    : r.paired_at ? "stale"
    : "waiting";
  return {
    id: String(r.id),
    code: r.code,
    camera_id: r.camera_id,
    label: r.label,
    kind: r.kind as Device["kind"],
    source_url: r.source_url,
    status,
    created_at: isoReq(r.created_at),
    paired_at: iso(r.paired_at),
    last_frame_at: iso(r.last_frame_at),
    pair_url: `${origin}/cam/${r.code}`,
  };
}

/** Longer than frames.ts STALE_MS: last_frame_at is written once a second, so a
 *  tighter bound would flap a perfectly healthy phone in and out of "live". */
const DEVICE_STALE_MS = 20_000;

export async function createDevice(label: string | null): Promise<Device> {
  // Retry on the unique index rather than pre-checking: with 31^6 codes a
  // collision is vanishingly rare, and a SELECT-then-INSERT is a race anyway.
  for (let attempt = 0; attempt < 5; attempt++) {
    const code = newCode();
    const n = await sql<{ n: number }[]>`SELECT count(*)::int n FROM devices`;
    const cameraId = `PHONE${(n[0]?.n ?? 0) + 1}`;
    try {
      await sql`
        INSERT INTO cameras (id, name, lat, lon, heading_deg, source_uri)
        VALUES (${cameraId}, ${label ?? cameraId},
                ${DEVICE_ORIGIN.lat + (n[0]?.n ?? 0) * DEVICE_SPACING_DEG},
                ${DEVICE_ORIGIN.lon}, NULL, NULL)
        ON CONFLICT (id) DO NOTHING`;
      const rows = await sql<DeviceRow[]>`
        INSERT INTO devices (code, camera_id, label)
        VALUES (${code}, ${cameraId}, ${label})
        RETURNING *`;
      return toDevice(rows[0]!);
    } catch (e) {
      if (attempt === 4) throw e;
    }
  }
  throw new Error("could not allocate a pairing code");
}

export async function getDevices(): Promise<Device[]> {
  const rows = await sql<DeviceRow[]>`
    SELECT * FROM devices WHERE NOT revoked ORDER BY created_at DESC LIMIT 50`;
  return rows.map((r) => toDevice(r));
}

export async function getDeviceByCode(code: string): Promise<Device | null> {
  const rows = await sql<DeviceRow[]>`
    SELECT * FROM devices WHERE code = ${code.toUpperCase()} AND NOT revoked`;
  return rows[0] ? toDevice(rows[0]) : null;
}

/**
 * Claim a code. Idempotent: a phone that reconnects re-pairs the same device.
 *
 * Returns null for an unknown or revoked code, which is what makes the code the
 * credential — nothing downstream accepts frames without a device coming back
 * from here.
 */
export async function pairDevice(
  code: string, kind: Device["kind"], sourceUrl: string | null,
): Promise<Device | null> {
  const rows = await sql<DeviceRow[]>`
    UPDATE devices
       SET kind = ${kind}, source_url = ${sourceUrl},
           paired_at = COALESCE(paired_at, now())
     WHERE code = ${code.toUpperCase()} AND NOT revoked
    RETURNING *`;
  return rows[0] ? toDevice(rows[0]) : null;
}

export async function touchDevice(cameraId: string): Promise<void> {
  await sql`UPDATE devices SET last_frame_at = now() WHERE camera_id = ${cameraId}`;
}

export async function revokeDevice(id: string): Promise<boolean> {
  const rows = await sql`
    UPDATE devices SET revoked = true, kind = NULL, source_url = NULL
     WHERE id = ${id}::bigint AND NOT revoked RETURNING id`;
  return rows.length > 0;
}

/** What the worker needs: every device with something to decode right now. */
export async function activeDeviceSources(): Promise<
  { camera_id: string; kind: string; source_url: string | null }[]
> {
  return sql<{ camera_id: string; kind: string; source_url: string | null }[]>`
    SELECT camera_id, kind, source_url
      FROM devices
     WHERE NOT revoked AND kind IS NOT NULL
       AND (kind = 'url'
            OR last_frame_at > now() - make_interval(secs => ${DEVICE_STALE_MS / 1000}))`;
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
       -- City-wide analytics counts live cameras. Uploaded footage is reported
       -- on its own page, and is included here only when asked for by name.
       AND (${camera ?? null}::text IS NOT NULL
            OR d.camera_id NOT IN (SELECT id FROM cameras WHERE is_upload))
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

export async function getAlerts(acked?: boolean, limit = 200): Promise<Alert[]> {
  const n = Math.min(Math.max(limit, 1), 500);
  const rows = await sql<{
    id: string; ts: Date; camera_id: string | null; kind: string; severity: string;
    track_id: string | null; plate_text: string | null; detail: string | null; acked: boolean;
  }[]>`
    SELECT id, ts, camera_id, kind, severity, track_id, plate_text, detail, acked
      FROM alerts
     WHERE (${acked ?? null}::boolean IS NULL OR acked = ${acked ?? null})
       AND (camera_id IS NULL
            OR camera_id NOT IN (SELECT id FROM cameras WHERE is_upload))
     ORDER BY ts DESC
     LIMIT ${n}`;
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

/**
 * The one column that cannot be filled at INSERT time: speed needs the track's
 * whole detection history, and at upsert the last frames are still buffered.
 */
export async function setTrackSpeed(
  cameraId: string, trackId: string, kmh: number,
): Promise<void> {
  await sql`
    UPDATE tracks SET speed_kmh = ${kmh}
     WHERE camera_id = ${cameraId} AND track_id = ${trackId}`;
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
    // A trajectory is a path, and a path cannot visit the same node twice.
    // Appending a track already in the chain produces a cycle, and the
    // timestamps along it then run backwards -- the contract's own ordering
    // check catches it, but only after the bad row is stored.
    //
    // The cause was sidecars reusing track ids after a restart, fixed by
    // RUN_ID in ml/sidecar.py. This stays as the guard that makes an invalid
    // trajectory impossible to write, whatever produces the match.
    if (existing.track_ids.map(String).includes(String(toTrack))) {
      console.warn(
        `trajectory ${existing.id} already contains track ${toTrack}; ` +
        `refusing to append and create a cycle`);
      return { id: String(existing.id), track_ids: existing.track_ids.map(String) };
    }
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
