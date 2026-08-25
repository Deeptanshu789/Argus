/**
 * The API contract, as zod schemas.
 *
 * SINGLE SOURCE OF TRUTH. Types are inferred from these schemas, so the mock,
 * the real routes, the WebSocket hub, the ingest worker and the frontend all
 * derive from one definition. A shape can no longer drift between them without
 * a type error — which is the failure mode this project cannot afford, since
 * two developers build against this boundary in parallel.
 *
 * Changing a schema here IS a contract change. Announce it.
 */
import { z } from "zod";

export const CameraStatus = z.enum(["online", "offline", "degraded"]);
export const VehicleType = z.enum(["car", "bus", "truck", "motorcycle", "auto"]);
export const MatchMethod = z.enum(["plate", "reid", "spatial_temporal"]);
export const AlertKind = z.enum(["stationary", "wrong_way", "volume_spike", "watchlist"]);
export const Severity = z.enum(["info", "warn", "critical"]);

const Iso = z.string().describe("ISO 8601 UTC");

export const Camera = z.object({
  id: z.string(),
  name: z.string(),
  lat: z.number(),
  lon: z.number(),
  heading_deg: z.number().int().nullable(),
  status: CameraStatus,
  stream_url: z.string(),
});

export const CameraLink = z.object({
  from: z.string(),
  to: z.string(),
  distance_m: z.number().int().positive(),
  /** What layer 3 of the association engine checks candidate matches against. */
  travel_time_s: z.number().int().positive(),
});

export const Track = z.object({
  id: z.string(),
  camera_id: z.string(),
  track_id: z.string(),
  /** null when OCR failed. The vehicle is still tracked, via Re-ID. */
  plate_text: z.string().nullable(),
  plate_conf: z.number().min(0).max(1).nullable(),
  vehicle_type: VehicleType,
  color: z.string().nullable(),
  entry_time: Iso,
  exit_time: Iso.nullable(),
});

export const Hop = z.object({
  from_camera: z.string(),
  to_camera: z.string(),
  method: MatchMethod,
  confidence: z.number().min(0).max(1),
  travel_time_s: z.number().int(),
});

export const Trajectory = z.object({
  id: z.string(),
  plate_text: z.string().nullable(),
  vehicle_type: VehicleType,
  started_at: Iso,
  ended_at: Iso.nullable(),
  /** [lon, lat, seconds_since_start] — exactly what deck.gl TripsLayer consumes. */
  path: z.array(z.tuple([z.number(), z.number(), z.number()])),
  hops: z.array(Hop),
});

export const Sighting = z.object({
  camera_id: z.string(),
  ts: Iso,
  confidence: z.number().nullable(),
});

export const SearchResult = z.object({
  plate_text: z.string(),
  trajectories: z.array(Trajectory),
  sightings: z.array(Sighting),
  last_seen: z.object({ camera_id: z.string(), ts: Iso }).nullable(),
});

export const AnalyticsBucket = z.object({
  ts: Iso,
  vehicle_count: z.number().int().nonnegative(),
  avg_speed_kmh: z.number().nullable(),
  congestion_score: z.number().min(0).max(100).nullable(),
  /** Sparse: a bucket only carries the types it actually saw. */
  by_type: z.record(z.string(), z.number().int()),
});

export const AnalyticsResponse = z.object({
  camera_id: z.string().nullable(),
  series: z.array(AnalyticsBucket),
  totals: z.object({
    vehicle_count: z.number().int(),
    avg_speed_kmh: z.number().nullable(),
  }),
});

export const Alert = z.object({
  id: z.string(),
  ts: Iso,
  camera_id: z.string().nullable(),
  kind: AlertKind,
  severity: Severity,
  track_id: z.string().nullable(),
  plate_text: z.string().nullable(),
  detail: z.string(),
  acked: z.boolean(),
});

// ------------------------------------------------------------ websocket --
// Every message is { type, data }. Clients switch on `type` and MUST ignore
// unknown types — that rule is what lets the server add message types without
// breaking a frontend built earlier.

export const DetectionMsg = z.object({
  type: z.literal("detection"),
  data: z.object({
    camera_id: z.string(),
    track_id: z.string(),
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    vehicle_type: VehicleType,
    plate_text: z.string().nullable(),
    conf: z.number(),
  }),
});

export const MatchMsg = z.object({
  type: z.literal("match"),
  data: Hop.extend({ trajectory_id: z.string(), plate_text: z.string().nullable() }),
});

export const TrajectoryUpdateMsg = z.object({
  type: z.literal("trajectory_update"),
  data: Trajectory.pick({ id: true, path: true }),
});

export const AnalyticsMsg = z.object({
  type: z.literal("analytics"),
  data: z.object({
    ts: Iso,
    per_camera: z.record(
      z.string(),
      z.object({ vehicle_count: z.number().int(), congestion_score: z.number() }),
    ),
    city: z.object({ vehicle_count: z.number().int(), avg_speed_kmh: z.number() }),
  }),
});

export const AlertMsg = z.object({ type: z.literal("alert"), data: Alert });

export const ServerMessage = z.discriminatedUnion("type", [
  DetectionMsg, MatchMsg, TrajectoryUpdateMsg, AnalyticsMsg, AlertMsg,
]);

// --------------------------------------------------------- sidecar input --
/**
 * What the Python inference sidecar emits, one JSON object per line on stdout.
 * This is the ONLY place Python and TypeScript meet. Everything downstream —
 * association, analytics, API, UI — is TypeScript.
 */
export const SidecarEvent = z.discriminatedUnion("event", [
  z.object({
    event: z.literal("detection"),
    camera_id: z.string(),
    track_id: z.string(),
    ts: Iso,
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    conf: z.number(),
    vehicle_type: VehicleType,
  }),
  z.object({
    event: z.literal("track_closed"),
    camera_id: z.string(),
    track_id: z.string(),
    entry_time: Iso,
    exit_time: Iso,
    vehicle_type: VehicleType,
    color: z.string().nullable(),
    plate_text: z.string().nullable(),
    plate_conf: z.number().nullable(),
    /**
     * Re-ID appearance vector, computed on track exit only — see CLAUDE.md.
     *
     * Length is NOT pinned here. It is whatever the tracker's ReID model
     * produces, and swapping that model changes the dimension. What matters is
     * that every camera produces the SAME dimension, since Module C compares
     * them across cameras — the worker enforces that at runtime, because a
     * mismatch makes cosine similarity return 0 and layer 2 silently stop
     * firing, which looks like "Re-ID just isn't matching" rather than a bug.
     */
    embedding: z.array(z.number()).min(32),
    color_hist: z.array(z.number()),
  }),
  z.object({ event: z.literal("ready"), camera_id: z.string(), fps: z.number() }),
  z.object({ event: z.literal("error"), camera_id: z.string(), detail: z.string() }),
]);

export type CameraStatus = z.infer<typeof CameraStatus>;
export type VehicleType = z.infer<typeof VehicleType>;
export type MatchMethod = z.infer<typeof MatchMethod>;
export type AlertKind = z.infer<typeof AlertKind>;
export type Severity = z.infer<typeof Severity>;
export type Camera = z.infer<typeof Camera>;
export type CameraLink = z.infer<typeof CameraLink>;
export type Track = z.infer<typeof Track>;
export type Hop = z.infer<typeof Hop>;
export type Trajectory = z.infer<typeof Trajectory>;
export type SearchResult = z.infer<typeof SearchResult>;
export type AnalyticsBucket = z.infer<typeof AnalyticsBucket>;
export type AnalyticsResponse = z.infer<typeof AnalyticsResponse>;
export type Alert = z.infer<typeof Alert>;
export type ServerMessage = z.infer<typeof ServerMessage>;
export type SidecarEvent = z.infer<typeof SidecarEvent>;
