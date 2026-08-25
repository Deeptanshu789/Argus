#!/usr/bin/env python3
"""Mock API — every shape in the wiki's API-Contract, served from fixtures.

WHY THIS EXISTS: it is the hour 0-2 unblocker. Dev B builds the entire frontend
against these shapes and never waits on the real pipeline. Dev A owns this file
and replaces each route with the real implementation behind identical shapes.

    uvicorn backend.mock:app --reload --port 8000

Then the frontend sets VITE_MOCK=1 and points at /api/mock. Switching to live
data is one environment variable.

Fixtures are seeded (Random(0)) so every reload gives the same data — a chart
that reshuffles on refresh makes it impossible to tell a UI bug from new data.
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Argus mock API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

RNG = random.Random(0)
NOW = datetime.now(timezone.utc).replace(microsecond=0)
TYPES = ["car", "bus", "truck", "motorcycle"]
COLORS = ["white", "silver", "black", "red", "blue"]
STATES = ["KA", "MH", "DL", "TN"]


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def plate(r: random.Random) -> str:
    return (f"{r.choice(STATES)}{r.randint(1, 30):02d}"
            f"{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
            f"{r.randint(1, 9999):04d}")


# Bengaluru MG Road corridor — four cameras roughly along one route.
CAMERAS = [
    {"id": "CAM1", "name": "MG Road Junction",   "lat": 12.9752, "lon": 77.6068,
     "heading_deg": 90,  "status": "online", "stream_url": "/api/mock/stream/CAM1"},
    {"id": "CAM2", "name": "Trinity Circle",     "lat": 12.9730, "lon": 77.6198,
     "heading_deg": 45,  "status": "online", "stream_url": "/api/mock/stream/CAM2"},
    {"id": "CAM3", "name": "Ulsoor Gate",        "lat": 12.9788, "lon": 77.6101,
     "heading_deg": 270, "status": "online", "stream_url": "/api/mock/stream/CAM3"},
    {"id": "CAM4", "name": "Richmond Circle",    "lat": 12.9611, "lon": 77.5966,
     "heading_deg": 180, "status": "degraded", "stream_url": "/api/mock/stream/CAM4"},
]
CAM_BY_ID = {c["id"]: c for c in CAMERAS}

LINKS = [
    {"from": "CAM1", "to": "CAM3", "distance_m": 1400, "travel_time_s": 168},
    {"from": "CAM3", "to": "CAM1", "distance_m": 1400, "travel_time_s": 168},
    {"from": "CAM1", "to": "CAM2", "distance_m": 1500, "travel_time_s": 200},
    {"from": "CAM2", "to": "CAM1", "distance_m": 1500, "travel_time_s": 200},
    {"from": "CAM1", "to": "CAM4", "distance_m": 1900, "travel_time_s": 240},
    {"from": "CAM4", "to": "CAM1", "distance_m": 1900, "travel_time_s": 240},
]


def _build_tracks(n: int = 120) -> list[dict]:
    r = random.Random(1)
    out = []
    for i in range(n):
        entry = NOW - timedelta(seconds=r.randint(30, 3600))
        # ~15% have no readable plate — the frontend MUST render this case,
        # because Re-ID-only matching is exactly what we demo in Act 2.
        readable = r.random() > 0.15
        out.append({
            "id": str(8000 + i),
            "camera_id": r.choice(CAMERAS)["id"],
            "track_id": f"T{i:04d}",
            "plate_text": plate(r) if readable else None,
            "plate_conf": round(r.uniform(0.82, 0.99), 2) if readable else None,
            "vehicle_type": r.choices(TYPES, weights=[70, 5, 10, 15])[0],
            "color": r.choice(COLORS),
            "entry_time": iso(entry),
            "exit_time": iso(entry + timedelta(seconds=r.randint(4, 20))),
        })
    return sorted(out, key=lambda t: t["entry_time"], reverse=True)


TRACKS = _build_tracks()
# One known-good plate the demo and the search box can always rely on.
DEMO_PLATE = "KA05MR7821"
TRACKS[0]["plate_text"] = DEMO_PLATE
TRACKS[0]["plate_conf"] = 0.97
TRACKS[0]["camera_id"] = "CAM1"


def _build_trajectories(n: int = 24) -> list[dict]:
    r = random.Random(2)
    out = []
    for i in range(n):
        hops_n = r.randint(1, 3)
        cams = [CAMERAS[0]] + r.sample(CAMERAS[1:], hops_n)
        start = NOW - timedelta(seconds=r.randint(120, 3000))
        t, path, hops = 0, [], []
        for a, b in zip(cams, cams[1:]):
            path.append([a["lon"], a["lat"], t])
            link = next((l for l in LINKS
                         if l["from"] == a["id"] and l["to"] == b["id"]), None)
            dt = link["travel_time_s"] if link else r.randint(120, 300)
            t += dt
            method = r.choices(["plate", "reid", "spatial_temporal"],
                               weights=[65, 25, 10])[0]
            hops.append({
                "from_camera": a["id"], "to_camera": b["id"], "method": method,
                "confidence": 0.99 if method == "plate" else round(r.uniform(0.76, 0.93), 2),
                "travel_time_s": dt,
            })
        path.append([cams[-1]["lon"], cams[-1]["lat"], t])
        out.append({
            "id": f"TRJ-{1000 + i}",
            "plate_text": DEMO_PLATE if i == 0 else plate(r),
            "vehicle_type": r.choice(TYPES),
            "started_at": iso(start),
            "ended_at": iso(start + timedelta(seconds=t)),
            "path": path,
            "hops": hops,
        })
    return out


TRAJECTORIES = _build_trajectories()

ALERTS = [
    {"id": "A-77", "ts": iso(NOW - timedelta(minutes=3)), "camera_id": "CAM2",
     "kind": "stationary", "severity": "warn", "plate_text": "MH04AQ5678",
     "detail": "Stationary for 7m 12s", "acked": False},
    {"id": "A-78", "ts": iso(NOW - timedelta(minutes=9)), "camera_id": "CAM4",
     "kind": "wrong_way", "severity": "critical", "plate_text": None,
     "detail": "Vehicle travelling against flow", "acked": False},
    {"id": "A-79", "ts": iso(NOW - timedelta(minutes=21)), "camera_id": "CAM1",
     "kind": "volume_spike", "severity": "info", "plate_text": None,
     "detail": "Volume 41% above 1h baseline", "acked": True},
]


def _series(camera_id: str, buckets: int = 12) -> list[dict]:
    r = random.Random(hash(camera_id) & 0xFFFF)
    out = []
    for i in range(buckets):
        count = r.randint(40, 120)
        speed = round(r.uniform(12, 46), 1)
        out.append({
            "ts": iso(NOW - timedelta(minutes=5 * (buckets - i))),
            "vehicle_count": count,
            "avg_speed_kmh": speed,
            # density x inverse speed, clamped to 0-100
            "congestion_score": round(min(100.0, count / 1.2 * (30 / speed)), 1),
            "by_type": {t: max(0, int(count * w))
                        for t, w in zip(TYPES, [0.70, 0.04, 0.09, 0.17])},
        })
    return out


# ------------------------------------------------------------------ routes --
P = "/api/mock"


@app.get(P + "/cameras")
def cameras():
    return CAMERAS


@app.get(P + "/cameras/links")
def camera_links():
    return LINKS


@app.get(P + "/tracks")
def tracks(camera: str | None = None, since: str | None = None, limit: int = 100):
    rows = [t for t in TRACKS if camera is None or t["camera_id"] == camera]
    if since:
        rows = [t for t in rows if t["entry_time"] >= since]
    return rows[:limit]


@app.get(P + "/trajectories")
def trajectories(since: str | None = None, limit: int = 50):
    rows = TRAJECTORIES
    if since:
        rows = [t for t in rows if t["started_at"] >= since]
    return rows[:limit]


@app.get(P + "/search")
def search(plate: str):
    """Empty result is 200 with empty arrays, never 404 — 'no such plate' is a
    normal answer, and a 404 makes the UI render an error state for it."""
    key = plate.replace(" ", "").upper()
    trjs = [t for t in TRAJECTORIES if (t["plate_text"] or "") == key]
    sightings = [
        {"camera_id": t["camera_id"], "ts": t["entry_time"], "confidence": t["plate_conf"]}
        for t in TRACKS if (t["plate_text"] or "") == key
    ]
    sightings.sort(key=lambda s: s["ts"])
    return {
        "plate_text": key,
        "trajectories": trjs,
        "sightings": sightings,
        "last_seen": ({"camera_id": sightings[-1]["camera_id"], "ts": sightings[-1]["ts"]}
                      if sightings else None),
    }


@app.get(P + "/analytics")
def analytics(camera: str | None = None, window: str = "1h", bucket: str = "5m"):
    if camera:
        series = _series(camera)
    else:  # city-wide: sum per bucket across cameras
        per_cam = [_series(c["id"]) for c in CAMERAS]
        series = []
        for i in range(len(per_cam[0])):
            rows = [pc[i] for pc in per_cam]
            series.append({
                "ts": rows[0]["ts"],
                "vehicle_count": sum(r["vehicle_count"] for r in rows),
                "avg_speed_kmh": round(sum(r["avg_speed_kmh"] for r in rows) / len(rows), 1),
                "congestion_score": round(sum(r["congestion_score"] for r in rows) / len(rows), 1),
                "by_type": {t: sum(r["by_type"][t] for r in rows) for t in TYPES},
            })
    total = sum(s["vehicle_count"] for s in series)
    return {
        "camera_id": camera,
        "series": series,
        "totals": {
            "vehicle_count": total,
            "avg_speed_kmh": round(sum(s["avg_speed_kmh"] for s in series) / len(series), 1),
        },
    }


@app.get(P + "/alerts")
def alerts(acked: bool | None = None):
    return [a for a in ALERTS if acked is None or a["acked"] == acked]


@app.post(P + "/alerts/{alert_id}/ack")
def ack(alert_id: str):
    for a in ALERTS:
        if a["id"] == alert_id:
            a["acked"] = True
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """Replays a canned event loop. Real hub replaces this; the message shapes
    and the switch-on-`type` rule stay identical."""
    await sock.accept()
    r = random.Random(3)
    try:
        tick = 0
        while True:
            for _ in range(5):  # ~5 detections/s, matching the real 5 FPS budget
                t = r.choice(TRACKS)
                x, y = r.randint(80, 900), r.randint(60, 500)
                await sock.send_json({"type": "detection", "data": {
                    "camera_id": t["camera_id"], "track_id": t["track_id"],
                    "bbox": [x, y, x + r.randint(90, 180), y + r.randint(60, 130)],
                    "vehicle_type": t["vehicle_type"], "plate_text": t["plate_text"],
                    "conf": t["plate_conf"] or round(r.uniform(0.6, 0.9), 2),
                }})
                await asyncio.sleep(0.2)
            tick += 1

            if tick % 5 == 0:  # the money message, every ~5 s
                trj = r.choice(TRAJECTORIES)
                hop = trj["hops"][0]
                await sock.send_json({"type": "match", "data": {
                    "trajectory_id": trj["id"], "plate_text": trj["plate_text"], **hop}})
                await sock.send_json({"type": "trajectory_update",
                                      "data": {"id": trj["id"], "path": trj["path"]}})

            if tick % 5 == 0:
                await sock.send_json({"type": "analytics", "data": {
                    "ts": iso(datetime.now(timezone.utc)),
                    "per_camera": {c["id"]: {
                        "vehicle_count": r.randint(40, 120),
                        "congestion_score": round(r.uniform(10, 95), 1)} for c in CAMERAS},
                    "city": {"vehicle_count": r.randint(200, 400),
                             "avg_speed_kmh": round(r.uniform(18, 38), 1)},
                }})

            if tick % 20 == 0:
                await sock.send_json({"type": "alert", "data": r.choice(ALERTS)})
    except WebSocketDisconnect:
        pass


@app.get("/health")
def health():
    return {"ok": True, "mode": "mock"}


def _selfcheck() -> None:
    """Every contract shape is present and internally consistent."""
    assert {c["id"] for c in CAMERAS} == {"CAM1", "CAM2", "CAM3", "CAM4"}
    for l in LINKS:
        assert l["from"] in CAM_BY_ID and l["to"] in CAM_BY_ID, l
        assert l["travel_time_s"] > 0

    assert any(t["plate_text"] is None for t in TRACKS), "need unreadable-plate cases"
    for t in TRACKS:
        assert t["camera_id"] in CAM_BY_ID
        assert (t["plate_text"] is None) == (t["plate_conf"] is None)

    for trj in TRAJECTORIES:
        assert len(trj["path"]) == len(trj["hops"]) + 1, trj["id"]
        assert [p[2] for p in trj["path"]] == sorted(p[2] for p in trj["path"]), \
            f"{trj['id']}: TripsLayer needs monotonically increasing timestamps"
        for h in trj["hops"]:
            assert h["method"] in {"plate", "reid", "spatial_temporal"}
            assert 0 < h["confidence"] <= 1

    hit = search(DEMO_PLATE)
    assert hit["sightings"] and hit["last_seen"], "demo plate must be findable"
    assert search("ZZ99ZZ9999")["trajectories"] == [], "miss must be an empty 200"

    a = analytics()
    assert len(a["series"]) == 12 and a["totals"]["vehicle_count"] > 0
    assert all(0 <= s["congestion_score"] <= 100 for s in a["series"])
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
