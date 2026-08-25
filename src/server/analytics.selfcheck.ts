/**
 * One runnable check for Module D.
 *
 *   npx tsx src/server/analytics.selfcheck.ts
 */
import assert from "node:assert/strict";
import {
  THRESHOLDS, angleDelta, bucketize, congestionScore, detectStationary, detectVolumeSpike,
  detectWrongWay, estimateSpeed, headingDeg,
  type CameraCalibration, type DetectionRow,
} from "@/server/analytics";
import type { AnalyticsBucket } from "@/contract";

const T0 = Date.parse("2026-08-25T10:00:00Z");
const at = (s: number) => new Date(T0 + s * 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
/** 0.1 m/px: a 1000 px wide view spans 100 m. Plausible for a junction camera. */
const CAL: CameraCalibration = { metersPerPixel: 0.1 };

const det = (o: Partial<DetectionRow> & { ts: string }): DetectionRow => ({
  camera_id: "CAM1", track_id: "T1", bbox: [100, 100, 160, 140], vehicle_type: "car", ...o,
});

// ------------------------------------------------------------------ speed --
{
  // 100 px in 2 s at 0.1 m/px = 5 m/s = 18 km/h.
  const rows = [det({ ts: at(0) }), det({ ts: at(2), bbox: [200, 100, 260, 140] })];
  assert.equal(estimateSpeed(rows, CAL), 18);
  // Calibration is the knob: double m/px, double the speed.
  assert.equal(estimateSpeed(rows, { metersPerPixel: 0.2 }), 36);
}
assert.equal(estimateSpeed([det({ ts: at(0) })], CAL), null,
  "one frame gives no displacement — null, not a fabricated 0 that drags averages down");
assert.equal(estimateSpeed([det({ ts: at(0) }), det({ ts: at(0) })], CAL), null,
  "zero elapsed time must not divide by zero");

// ---------------------------------------------------------------- heading --
{
  const north = [det({ ts: at(0), bbox: [100, 500, 160, 540] }),
                 det({ ts: at(2), bbox: [100, 100, 160, 140] })];  // y decreases = up
  assert.equal(headingDeg(north), 0);
  const east = [det({ ts: at(0) }), det({ ts: at(2), bbox: [500, 100, 560, 140] })];
  assert.equal(headingDeg(east), 90);
}
assert.equal(headingDeg([det({ ts: at(0) }), det({ ts: at(2) })]), null,
  "sub-pixel jitter is not a direction");
assert.equal(angleDelta(0, 180), 180);
assert.equal(angleDelta(350, 10), 20, "must wrap around 360");
assert.equal(angleDelta(90, 90), 0);

// ------------------------------------------------------------- congestion --
assert.equal(congestionScore(0, 40), 0, "empty road");
assert.equal(congestionScore(100, 40, 100, 40), 50, "full road at the speed limit");
assert.equal(congestionScore(100, 0, 100, 40), 100, "full road at a standstill");
assert.ok(congestionScore(2, 1, 100, 40) < 5,
  "one crawling vehicle on an empty street is not congestion");
assert.ok(congestionScore(200, 5, 100, 40) <= 100, "density is clamped");
assert.equal(congestionScore(50, null, 100, 40), 50,
  "no speed reading scores on density alone rather than inventing a speed");

// -------------------------------------------------------------- bucketize --
{
  const rows: DetectionRow[] = [];
  // Three vehicles in bucket 0. One emits 20 detections — counting rows instead
  // of unique tracks would report 22 vehicles instead of 3.
  for (let i = 0; i < 20; i++) {
    rows.push(det({ ts: at(i), track_id: "A", bbox: [100 + i * 10, 100, 160 + i * 10, 140] }));
  }
  rows.push(det({ ts: at(30), track_id: "B", vehicle_type: "bus" }));
  rows.push(det({ ts: at(31), track_id: "B", vehicle_type: "bus", bbox: [150, 100, 210, 140] }));
  rows.push(det({ ts: at(40), track_id: "C", vehicle_type: "truck" }));
  // A fourth in the next 5-minute bucket.
  rows.push(det({ ts: at(400), track_id: "D" }));

  const buckets = bucketize(rows, CAL, 300);
  assert.equal(buckets.length, 2);
  assert.equal(buckets[0]!.vehicle_count, 3, "unique tracks, not detection rows");
  assert.deepEqual(buckets[0]!.by_type, { car: 1, bus: 1, truck: 1 });
  assert.equal(buckets[1]!.vehicle_count, 1);
  assert.ok(buckets[0]!.ts < buckets[1]!.ts, "buckets come out in time order");
  // Track C has a single frame; its unmeasurable speed must not enter the mean.
  assert.ok(buckets[0]!.avg_speed_kmh !== null && buckets[0]!.avg_speed_kmh > 0);
  assert.deepEqual(bucketize([], CAL), []);
}

// ------------------------------------------------------------- stationary --
{
  const rows = Array.from({ length: 10 }, (_, i) =>
    det({ ts: at(i * 40), bbox: [100, 100, 160, 140] }));   // 360 s, no movement
  const a = detectStationary({ camera_id: "CAM1", track_id: "T1", plate_text: "KA01AA1111", rows }, CAL);
  assert.ok(a && a.kind === "stationary" && a.severity === "warn");
  assert.match(a.detail, /^Stationary for 6m 0s$/);
}
{
  const rows = [det({ ts: at(0) }), det({ ts: at(100) })];
  assert.equal(detectStationary({ camera_id: "CAM1", track_id: "T1", plate_text: null, rows }, CAL),
    null, `under ${THRESHOLDS.stationarySeconds}s does not fire`);
}
{
  // Drives away and comes back to the same pixel. Comparing only first-to-last
  // displacement would call this stationary; max-from-start does not.
  const rows = [
    det({ ts: at(0),   bbox: [100, 100, 160, 140] }),
    det({ ts: at(180), bbox: [900, 100, 960, 140] }),
    det({ ts: at(360), bbox: [100, 100, 160, 140] }),
  ];
  assert.equal(detectStationary({ camera_id: "CAM1", track_id: "T1", plate_text: null, rows }, CAL),
    null, "a vehicle that left and returned is not stationary");
}

// -------------------------------------------------------------- wrong way --
{
  const cal: CameraCalibration = { metersPerPixel: 0.1, expectedFlowDeg: 0 };  // flow north
  const southbound = [det({ ts: at(0), bbox: [100, 100, 160, 140] }),
                      det({ ts: at(4), bbox: [100, 600, 160, 640] })];         // heads south
  const a = detectWrongWay({ camera_id: "CAM1", track_id: "T1", plate_text: null, rows: southbound }, cal);
  assert.ok(a && a.kind === "wrong_way" && a.severity === "critical");

  const northbound = [det({ ts: at(0), bbox: [100, 600, 160, 640] }),
                      det({ ts: at(4), bbox: [100, 100, 160, 140] })];
  assert.equal(detectWrongWay({ camera_id: "CAM1", track_id: "T1", plate_text: null, rows: northbound }, cal),
    null, "travelling with the flow is not an alert");

  assert.equal(detectWrongWay({ camera_id: "CAM1", track_id: "T1", plate_text: null, rows: southbound }, CAL),
    null, "an uncalibrated camera makes no wrong-way claim at all");
}

// ------------------------------------------------------------ volume spike --
{
  const b = (n: number, i: number): AnalyticsBucket => ({
    ts: at(i * 300), vehicle_count: n, avg_speed_kmh: 30, congestion_score: 40, by_type: {},
  });
  const flat = [b(50, 0), b(52, 1), b(48, 2), b(50, 3)];
  assert.equal(detectVolumeSpike("CAM1", [...flat, b(51, 4)]), null, "steady volume is not a spike");
  const spike = detectVolumeSpike("CAM1", [...flat, b(90, 4)]);
  assert.ok(spike && spike.kind === "volume_spike" && spike.severity === "info");
  assert.match(spike.detail, /above 4-bucket baseline/);
  assert.equal(detectVolumeSpike("CAM1", [b(50, 0), b(90, 1)]), null,
    "too little history — firing here trains everyone to ignore the alert panel");
  assert.equal(detectVolumeSpike("CAM1", [b(0, 0), b(0, 1), b(0, 2), b(0, 3), b(5, 4)]), null,
    "zero baseline must not divide by zero");
}

console.log("analytics selfcheck ok");
