/**
 * One runnable check for the fixtures. Fails if a shape drifts from the
 * contract, or if the cases the UI depends on stop existing.
 *
 *   npm run selfcheck
 */
import assert from "node:assert/strict";
import {
  Alert, AnalyticsResponse, Camera, CameraLink, SearchResult, Track, Trajectory,
  Device, Upload, UploadResult,
} from "@/contract";
import * as m from "@/server/mock";

// Every fixture validates against the contract it claims to implement.
Camera.array().parse(m.getCameras());
CameraLink.array().parse(m.getLinks());
Track.array().parse(m.getTracks({ limit: 500 }));
Trajectory.array().parse(m.getTrajectories({ limit: 100 }));
Alert.array().parse(m.getAlerts());
AnalyticsResponse.parse(m.getAnalytics());
AnalyticsResponse.parse(m.getAnalytics("CAM1"));
SearchResult.parse(m.search(m.DEMO_PLATE));

const ids = new Set(m.getCameras().map((c) => c.id));
assert.equal(ids.size, 4);
for (const l of m.getLinks()) {
  assert.ok(ids.has(l.from) && ids.has(l.to), `link references unknown camera: ${l.from}->${l.to}`);
}
assert.ok(m.getCameras().some((c) => c.status !== "online"), "UI must handle a degraded camera");

const tracks = m.getTracks({ limit: 500 });
assert.ok(tracks.some((t) => t.plate_text === null), "need unreadable-plate cases");
for (const t of tracks) {
  assert.equal(t.plate_text === null, t.plate_conf === null, `${t.id}: plate text/conf disagree`);
}
assert.deepEqual(m.getTracks({ camera: "CAM1" }).filter((t) => t.camera_id !== "CAM1"), []);

for (const trj of m.getTrajectories({ limit: 100 })) {
  assert.equal(trj.path.length, trj.hops.length + 1, `${trj.id}: path/hop count mismatch`);
  const ts = trj.path.map((p) => p[2]);
  assert.deepEqual(ts, [...ts].sort((a, b) => a - b),
    `${trj.id}: TripsLayer needs monotonically increasing timestamps`);
}

const hit = m.search(m.DEMO_PLATE);
assert.ok(hit.sightings.length && hit.last_seen, "demo plate must always be findable");
// Prefix search: an operator types the part of the plate they saw.
const partial = m.search(m.DEMO_PLATE.slice(0, 4));
assert.ok(partial.sightings.length >= hit.sightings.length,
  "a prefix must find at least what the full plate finds");
assert.ok(m.search(m.DEMO_PLATE.toLowerCase().replace(/(.{2})/, "$1 ")).sightings.length,
  "case and spacing must not change the result");
assert.deepEqual(m.search("").sightings, [],
  "an empty query matches nothing, not everything");

const miss = m.search("ZZ99ZZ9999");
assert.deepEqual(miss.trajectories, []);
assert.equal(miss.last_seen, null, "a miss is an empty 200, never a 404");

// Devices: a phone paired as a camera.
Device.array().parse(m.getDevices());
const dev = m.getDevices()[0]!;
assert.ok(dev.pair_url.includes(dev.code), "the pair link must carry the code");
assert.ok(m.getDeviceByCode(dev.code.toLowerCase()),
  "a code typed in lower case on a phone must still resolve");
assert.equal(m.getDeviceByCode("ZZZZZZ"), null, "an unknown code is null");

const issued = m.createDevice("smoke");
Device.parse(issued);
assert.equal(issued.status, "waiting", "a fresh code is waiting until something connects");
assert.equal(issued.kind, null, "a fresh code has not been claimed by either route");
assert.ok(!/[O0IL1]/.test(issued.code),
  `code ${issued.code} mixes characters that are misread off a screen`);
assert.equal(m.pairDeviceUrl(issued.code, "rtsp://x/y")?.kind, "url");
assert.ok(m.revokeDevice(issued.id), "revoke must succeed once");
assert.equal(m.getDeviceByCode(issued.code), null, "a revoked code stops resolving");
assert.equal(m.revokeDevice(issued.id), false, "revoking twice is not a second success");

// Uploads: the mock must answer the same shapes as the live API, or the upload
// page renders "no mock route: uploads" the moment NEXT_PUBLIC_MOCK is set.
const ups = m.getUploads();
assert.ok(ups.length, "the mock must ship at least one upload to design against");
const detail = m.getUpload(ups[0]!.id);
assert.ok(detail, "getUpload must find the upload getUploads just listed");
UploadResult.parse(detail);
const upCams = new Set(detail!.upload.sources.map((s) => s.camera_id));
assert.ok(detail!.plates.every((p) => upCams.has(p.camera_id)),
  "an upload's results must only contain its own cameras");
assert.ok(
  detail!.trajectories.every((t) =>
    t.hops.every((h) => upCams.has(h.from_camera) && upCams.has(h.to_camera))),
  "an upload's journeys must not leave its own cameras");
assert.equal(m.getUpload("nope"), null, "an unknown upload is null, not a throw");

const made = m.createUpload(["a.mp4", "b.mp4"], 168, "test");
Upload.parse(made);
assert.equal(made.status, "pending", "a new upload is pending until a worker runs it");
assert.equal(made.sources.length, 2);

const a = m.getAnalytics();
assert.equal(a.series.length, 12);
assert.ok(a.totals.vehicle_count > 0);
assert.ok(a.series.every((s) => (s.congestion_score ?? 0) >= 0 && (s.congestion_score ?? 0) <= 100));

assert.ok(m.getAlerts(false).every((x) => !x.acked));
m.ackAlert("A-77");
assert.ok(!m.getAlerts(false).some((x) => x.id === "A-77"), "ack must remove it from unacked");

console.log("mock selfcheck ok");
