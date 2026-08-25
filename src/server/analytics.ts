/**
 * Module D — traffic analytics and anomaly detection.
 *
 * Pure functions over detection/track rows. No database and no ML: bucket,
 * aggregate, and threshold. The worker calls these and writes the results;
 * BullMQ schedules the periodic rollup.
 *
 * Speed estimation needs CALIBRATION, not just code. A pixel is not a metre,
 * and the conversion differs per camera with mounting height, focal length and
 * viewing angle. `metersPerPixel` is that knob — measure it once per camera
 * against a known landmark (a lane width, a crossing) and store it. There is no
 * value a minimal implementation can guess for you.
 */
import type { AnalyticsBucket, AlertKind, Severity, VehicleType } from "@/contract";

export interface DetectionRow {
  ts: string;
  camera_id: string;
  track_id: string;
  bbox: [number, number, number, number];
  vehicle_type: VehicleType;
}

export interface CameraCalibration {
  /** Ground metres per image pixel at the road surface. MEASURE THIS. */
  metersPerPixel: number;
  /** Compass bearing of legitimate traffic flow, degrees. */
  expectedFlowDeg?: number;
}

/** ponytail: hand-tuned thresholds. First knobs to touch if alerts are noisy. */
export const THRESHOLDS = {
  /** A track must be open this long before "stationary" is even considered. */
  stationarySeconds: 300,
  /** ...and must have moved less than this, in metres, over that time. */
  stationaryMetres: 2,
  /** Heading disagreement above this many degrees counts as wrong-way. */
  wrongWayDeg: 120,
  /** A bucket this many times the rolling baseline is a spike. */
  spikeRatio: 1.4,
  /** Below this many baseline buckets, a spike claim is noise, not signal. */
  spikeMinHistory: 4,
};

const centre = (b: readonly [number, number, number, number]): [number, number] =>
  [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

// ---------------------------------------------------------------- speed --

/**
 * Track speed in km/h from bbox displacement. Returns null when the track is
 * too short to measure — one frame gives no displacement, and guessing zero
 * would drag every average down and inflate the congestion score.
 */
export function estimateSpeed(
  rows: readonly DetectionRow[],
  cal: CameraCalibration,
): number | null {
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => (a.ts < b.ts ? -1 : 1));
  const first = sorted[0]!, last = sorted[sorted.length - 1]!;
  const seconds = (Date.parse(last.ts) - Date.parse(first.ts)) / 1000;
  if (seconds <= 0) return null;
  const [x0, y0] = centre(first.bbox), [x1, y1] = centre(last.bbox);
  const metres = Math.hypot(x1 - x0, y1 - y0) * cal.metersPerPixel;
  return Math.round((metres / seconds) * 3.6 * 10) / 10;
}

/** Bearing of travel in degrees, 0 = up/north in image space. */
export function headingDeg(rows: readonly DetectionRow[]): number | null {
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => (a.ts < b.ts ? -1 : 1));
  const [x0, y0] = centre(sorted[0]!.bbox);
  const [x1, y1] = centre(sorted[sorted.length - 1]!.bbox);
  const dx = x1 - x0, dy = y1 - y0;
  if (Math.hypot(dx, dy) < 1) return null;   // jitter, not motion
  // Image y grows downward, so negate it to get a compass-style bearing.
  return (((Math.atan2(dx, -dy) * 180) / Math.PI) + 360) % 360;
}

/** Smallest absolute angle between two bearings, 0-180. */
export function angleDelta(a: number, b: number): number {
  // The +540/%360/-180 dance folds the difference into [-180, 180] before the
  // abs, which is what makes 350 vs 10 come out as 20 rather than 340.
  return Math.abs((((a - b) % 360) + 540) % 360 - 180);
}

// ---------------------------------------------------------- congestion --

/**
 * 0-100. Density times inverse speed: many vehicles moving slowly is
 * congestion; many vehicles moving freely is just a busy road.
 *
 * `capacity` is vehicles-per-bucket at which the camera's view is full, and
 * `freeFlowKmh` the speed limit. Both are per-camera facts, not constants.
 */
export function congestionScore(
  vehicleCount: number,
  avgSpeedKmh: number | null,
  capacity = 100,
  freeFlowKmh = 40,
): number {
  const density = Math.min(1, vehicleCount / Math.max(capacity, 1));
  // No speed reading means no evidence of slowness — score on density alone
  // rather than inventing a stand-in speed.
  if (avgSpeedKmh === null) return Math.round(density * 100 * 10) / 10;
  const slowness = Math.max(0, Math.min(1, 1 - avgSpeedKmh / Math.max(freeFlowKmh, 1)));
  // Density sets the ceiling, slowness scales within it: a full road at the
  // speed limit is 50, a full road at a crawl is 100, an empty road is 0.
  // Slowness alone can never raise the score, because one slow vehicle on an
  // empty street is not congestion.
  return Math.round(100 * density * (0.5 + 0.5 * slowness) * 10) / 10;
}

// ------------------------------------------------------------ bucketing --

/**
 * Group detections into fixed time buckets. Counts UNIQUE track ids, not
 * detection rows: at 5 FPS one vehicle produces dozens of rows, and counting
 * those would report a car park as a motorway.
 */
export function bucketize(
  rows: readonly DetectionRow[],
  cal: CameraCalibration,
  bucketSeconds = 300,
): AnalyticsBucket[] {
  const buckets = new Map<number, DetectionRow[]>();
  for (const r of rows) {
    const key = Math.floor(Date.parse(r.ts) / 1000 / bucketSeconds) * bucketSeconds;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(r);
  }

  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([key, group]) => {
      const byTrack = new Map<string, DetectionRow[]>();
      for (const r of group) {
        if (!byTrack.has(r.track_id)) byTrack.set(r.track_id, []);
        byTrack.get(r.track_id)!.push(r);
      }

      const speeds = [...byTrack.values()]
        .map((t) => estimateSpeed(t, cal))
        .filter((s): s is number => s !== null);
      const avg = speeds.length
        ? Math.round((speeds.reduce((a, b) => a + b, 0) / speeds.length) * 10) / 10
        : null;

      const byType: Record<string, number> = {};
      for (const [, t] of byTrack) {
        const kind = t[0]!.vehicle_type;
        byType[kind] = (byType[kind] ?? 0) + 1;
      }

      return {
        ts: new Date(key * 1000).toISOString().replace(/\.\d{3}Z$/, "Z"),
        vehicle_count: byTrack.size,
        avg_speed_kmh: avg,
        congestion_score: congestionScore(byTrack.size, avg),
        by_type: byType,
      };
    });
}

// ------------------------------------------------------------ anomalies --

export interface AnomalyInput {
  camera_id: string;
  track_id: string;
  plate_text: string | null;
  rows: readonly DetectionRow[];
}

export interface Anomaly {
  camera_id: string;
  track_id: string;
  plate_text: string | null;
  kind: AlertKind;
  severity: Severity;
  detail: string;
}

export function detectStationary(input: AnomalyInput, cal: CameraCalibration): Anomaly | null {
  const { rows } = input;
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => (a.ts < b.ts ? -1 : 1));
  const seconds = (Date.parse(sorted[sorted.length - 1]!.ts) - Date.parse(sorted[0]!.ts)) / 1000;
  if (seconds < THRESHOLDS.stationarySeconds) return null;

  const [x0, y0] = centre(sorted[0]!.bbox);
  // Max displacement from the start, not first-to-last: a vehicle that leaves
  // and returns to the same spot is not stationary, and comparing endpoints
  // alone would call it that.
  const moved = Math.max(...sorted.map((r) => {
    const [x, y] = centre(r.bbox);
    return Math.hypot(x - x0, y - y0);
  })) * cal.metersPerPixel;
  if (moved >= THRESHOLDS.stationaryMetres) return null;

  const mins = Math.floor(seconds / 60), secs = Math.round(seconds % 60);
  return {
    camera_id: input.camera_id, track_id: input.track_id, plate_text: input.plate_text,
    kind: "stationary", severity: "warn",
    detail: `Stationary for ${mins}m ${secs}s`,
  };
}

export function detectWrongWay(input: AnomalyInput, cal: CameraCalibration): Anomaly | null {
  if (cal.expectedFlowDeg === undefined) return null;  // uncalibrated: no claim
  const heading = headingDeg(input.rows);
  if (heading === null) return null;
  const delta = angleDelta(heading, cal.expectedFlowDeg);
  if (delta < THRESHOLDS.wrongWayDeg) return null;
  return {
    camera_id: input.camera_id, track_id: input.track_id, plate_text: input.plate_text,
    kind: "wrong_way", severity: "critical",
    detail: `Travelling ${Math.round(delta)}° against flow`,
  };
}

/**
 * Compare the newest bucket against the mean of the ones before it. Requires a
 * minimum history: with two prior buckets, "40% above baseline" is noise, and
 * firing on it trains everyone to ignore the alert panel.
 */
export function detectVolumeSpike(
  cameraId: string,
  buckets: readonly AnalyticsBucket[],
): Anomaly | null {
  if (buckets.length < THRESHOLDS.spikeMinHistory + 1) return null;
  const history = buckets.slice(0, -1);
  const latest = buckets[buckets.length - 1]!;
  const baseline = history.reduce((s, b) => s + b.vehicle_count, 0) / history.length;
  if (baseline <= 0) return null;
  const ratio = latest.vehicle_count / baseline;
  if (ratio < THRESHOLDS.spikeRatio) return null;
  return {
    camera_id: cameraId, track_id: "", plate_text: null,
    kind: "volume_spike", severity: "info",
    detail: `Volume ${Math.round((ratio - 1) * 100)}% above ${history.length}-bucket baseline`,
  };
}
