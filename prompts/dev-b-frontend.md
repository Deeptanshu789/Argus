# Antigravity prompt — Dev B (Frontend + Visualization)

Copy everything below the line into Antigravity as your task. It is
self-contained.

---

You are working on **Argus**, a city-wide ANPR and cross-camera vehicle
trajectory tracking system for Smart India Hackathon problem statement SIH26127
(Bharat Electronics Limited). Repo: `https://github.com/Deeptanshu789/Argus`

You are building the entire dashboard. **You are not blocked on anyone** — a
mock API already serves every endpoint you need, with realistic fixtures.

## Setup

```bash
git clone https://github.com/Deeptanshu789/Argus.git && cd Argus
git checkout -b dev-b-frontend

# Terminal 1 — the mock API (already written and tested, just run it)
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.mock:app --reload --port 8000

# Terminal 2 — your app
npm create vite@latest frontend -- --template react-ts
cd frontend && npm i deck.gl maplibre-gl recharts
npm run dev
```

Confirm the mock is live: `curl localhost:8000/api/mock/cameras`

**Read the wiki's API-Contract page before writing any code.** It is the exact
specification of every endpoint and WebSocket message. `backend/mock.py`
implements all of it — read that file too when you want to see real payloads.

## You own

`frontend/` and `demo/`. **Do not touch `backend/`, `ml/`, or `db/`** — another
developer owns them. If a response shape is wrong or missing a field, say so;
do not patch the serializer yourself.

## Wire the API through one constant

```ts
// frontend/src/api.ts
export const API = import.meta.env.VITE_API_URL +
  (import.meta.env.VITE_MOCK ? "/api/mock" : "/api");
export const WS = import.meta.env.VITE_API_URL.replace("http", "ws") + "/ws";
```

`.env.development`: `VITE_API_URL=http://localhost:8000` and `VITE_MOCK=1`.

Switching the whole app from fixtures to live data must be **one environment
variable**. If you hardcode `/api/mock` anywhere else, that breaks. This matters:
when the real backend lands, your code should not change at all.

## The fixtures deliberately include the cases that break naive UIs

Handle all of these from the start — they are not edge cases, they are the demo:

- **~15% of tracks have `plate_text: null`** (and `plate_conf: null`). These
  vehicles are still tracked, via appearance Re-ID. Rendering "null" or crashing
  here breaks the single most impressive moment in the demo. Show something like
  "plate unreadable — tracked by appearance".
- **`GET /search` on an unknown plate returns 200 with empty arrays**, never
  404. Render an empty state, not an error.
- **One camera reports `status: "degraded"`.** Show it.
- **Trajectory hops carry `method`:** `plate`, `reid`, or `spatial_temporal`.
  Colour each leg by method — this is how a judge *sees* the three-layer
  matching engine that differentiates this project.
- **`KA05MR7821`** is a known-good plate that search always finds. Use it when
  rehearsing.

Fixtures are seeded, so reloads give identical data. A chart that reshuffles on
refresh makes it impossible to tell a UI bug from new data — that is
intentional, do not add randomness.

## Build order

### 1. Camera grid

2x2 grid of `<img src={cam.stream_url}>` MJPEG feeds. Detection overlays are
drawn **server-side** — you do not position bounding boxes yourself. That is
deliberate: synchronising box coordinates against video frame timing is a
notorious time sink and there is no budget for it.

Subscribe to the `detection` WebSocket message for a live sidebar of recent
detections per camera, colour-coded by track state (green tracked / yellow new /
red lost).

**Acceptance:** four feeds render, detections stream in, no console errors.

### 2. City map — THE HERO VIEW

deck.gl over MapLibre GL (no API key needed — use a free style URL).

- **`TripsLayer`** for animated trajectories from `GET /api/trajectories`. The
  `path` field is already `[lon, lat, timestamp_seconds]` triples, which is
  exactly what `TripsLayer` consumes — no transformation needed. Timestamps are
  guaranteed monotonically increasing.
- **`HeatmapLayer`** for congestion, refreshed from the `analytics` WebSocket
  message every 5 s.
- **`IconLayer`** for camera positions, styled by `status`.
- Click a trajectory → panel showing plate, every sighting with timestamp, and
  the full route with each hop's `method` and confidence.
- Listen for `trajectory_update` and extend paths live.

**Acceptance:** trajectories visibly draw themselves along the route; clicking
one opens its detail; congestion heat shifts as `analytics` messages arrive.

### 3. Analytics view

From `GET /api/analytics` (omit `camera` for city-wide) plus live `analytics`
WebSocket messages. Recharts:

- Vehicle count time-series (area)
- Vehicle type breakdown (pie) from `by_type`
- Congestion trend (line, 0-100)
- Speed distribution
- Busiest-intersection leaderboard

**Acceptance:** charts populate from the 12 fixture buckets and update live.

### 4. Vehicle search

Plate number in → `GET /api/search`. Show the full trajectory on the map, a
timeline of every sighting, and last-seen location and time. Handle the empty
result cleanly.

**Acceptance:** searching `KA05MR7821` shows a route; searching junk shows an
empty state, not an error.

### 5. Alerts panel

`GET /api/alerts?acked=false`, plus live `alert` messages. Severity colouring
(`info` / `warn` / `critical`), kind labels (`stationary`, `wrong_way`,
`volume_spike`, `watchlist`), and an acknowledge button hitting
`POST /api/alerts/{id}/ack`.

### 6. Camera network data — YOUR DELIVERABLE TO THE BACKEND

Author the real camera topology and hand it over: for each camera an id, name,
lat/lon and heading; for each ordered camera pair a `distance_m` and
`travel_time_s`.

**This is not busywork.** `travel_time_s` is the data that Layer 3 of the
cross-camera matching engine uses to reject physically impossible matches. The
matching engine cannot be tested without it. Author it alongside the map that
visualizes it so the two never drift, and get it to the backend developer early.

### 7. Polish, last

Dark theme, loading and empty states, responsive layout, smooth transitions.

**Polish comes last on purpose.** A beautiful dashboard showing a false
trajectory is worse than a plain one showing a true one.

## WebSocket rules

Every message is `{"type": ..., "data": ...}`. **Switch on `type` and ignore
unknown types** — that rule lets the backend add message types without breaking
you. Handle reconnection: the demo must survive a dropped socket without a page
reload.

## Rules

- Never hardcode API paths outside `api.ts`.
- Never commit `node_modules/` or `dist/` — they are gitignored.
- If you need a field the API does not return, ask for it. Do not scrape it out
  of another endpoint.
- Prefer the simplest thing that works. This is a 36-hour build: no state
  management library until plain React state actually fails, no component
  abstraction with one use.
- Zoom-legible text: plate numbers must be readable from the back of a room.

## Merging

```bash
git add -A && git commit -m "feat(frontend): <what>"
git pull --rebase origin main
git push -u origin dev-b-frontend
```

Commit small and push often — a four-hour local branch is how you get a merge
conflict in a 36-hour build. `frontend/` is a disjoint tree from the backend, so
conflicts should be near zero. Open a PR into `main` as each view's acceptance
check passes; do not wait until everything is done.
