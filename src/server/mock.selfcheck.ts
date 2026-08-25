/**
 * One runnable check for the fixtures. Fails if a shape drifts from the
 * contract, or if the cases the UI depends on stop existing.
 *
 *   npm run selfcheck
 */
import assert from "node:assert/strict";
import { Alert, AnalyticsResponse, Camera, CameraLink, SearchResult, Track, Trajectory } from "@/contract";
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
const miss = m.search("ZZ99ZZ9999");
assert.deepEqual(miss.trajectories, []);
assert.equal(miss.last_seen, null, "a miss is an empty 200, never a 404");

const a = m.getAnalytics();
assert.equal(a.series.length, 12);
assert.ok(a.totals.vehicle_count > 0);
assert.ok(a.series.every((s) => (s.congestion_score ?? 0) >= 0 && (s.congestion_score ?? 0) <= 100));

assert.ok(m.getAlerts(false).every((x) => !x.acked));
m.ackAlert("A-77");
assert.ok(!m.getAlerts(false).some((x) => x.id === "A-77"), "ack must remove it from unacked");

console.log("mock selfcheck ok");
