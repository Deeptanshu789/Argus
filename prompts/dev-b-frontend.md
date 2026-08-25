# Antigravity prompt — Dev B (Dashboard + Visualization)

Copy everything below the line into Antigravity as your task. It is
self-contained.

---

You are working on **Argus**, a city-wide ANPR and cross-camera vehicle
trajectory tracking system for Smart India Hackathon problem statement SIH26127
(Bharat Electronics Limited). Repo: `https://github.com/Deeptanshu789/Argus`

You are building the entire dashboard. **You are not blocked on anyone** — a
mock API already serves every endpoint and WebSocket message you need, with
realistic fixtures.

## Setup

```bash
git clone https://github.com/Deeptanshu789/Argus.git && cd Argus
git checkout -b dev-b-dashboard
npm install
cp .env.example .env.local     # NEXT_PUBLIC_MOCK=1 is already set
npm run dev                    # UI + /api + /ws on :3000
```

Confirm: <http://localhost:3000> lists the mock routes and
`curl localhost:3000/api/mock/cameras` returns four cameras.

**Read `src/contract.ts` first.** It holds zod schemas for every response and
every WebSocket message, and all TypeScript types are inferred from it — so your
components are type-checked against the real contract, not against a hand-copied
interface. `src/server/mock.ts` shows what the payloads actually look like.

## The stack

Next.js 15 App Router + TypeScript. deck.gl, MapLibre GL, and Recharts are
already installed. There is **no Python anywhere in your work** — the only
Python in the repo is model training and one CV sidecar process, both owned by
the other developer.

## You own

`src/app/(dashboard)/**`, `src/components/**`, `demo/**`.

**Do not touch `src/server/**`, `src/app/api/**`, `worker/**`, `ml/**`, or
`db/**`** — the other developer owns them. If a response is missing a field, say
so; do not patch the route handler yourself. `src/contract.ts` is shared:
changing it is a contract change, announce it.

## Use the typed client, never bare fetch

`src/lib/api.ts` is the **only** place API paths are constructed:

```ts
import { getCameras, getTrajectories, search, connect } from "@/lib/api";
```

It reads `NEXT_PUBLIC_MOCK` and picks `/api/mock` or `/api`. Hardcode a path
anywhere else and the mock-to-live switch stops working — when the real backend
lands, your code should not change at all.

`connect(onMessage)` gives you the WebSocket with reconnection already handled
and returns a cleanup function for `useEffect`.

## The fixtures deliberately include the cases that break naive UIs

Handle all of these from the start — they are not edge cases, they are the demo:

- **~15% of tracks have `plate_text: null`** (and `plate_conf: null`). These
  vehicles are still tracked, via appearance Re-ID. Rendering "null" or crashing
  here breaks the single most impressive moment in the demo. Show something like
  "plate unreadable — tracked by appearance".
- **`search()` on an unknown plate returns 200 with empty arrays**, never 404.
  Render an empty state, not an error.
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

### 1. Dashboard shell

Replace `src/app/page.tsx` with a route group under `src/app/(dashboard)/`:
grid, map, analytics, search. Dark theme. One `connect()` subscription at the
shell level, fanned out to views — not one socket per component.

### 2. Camera grid

2x2 grid of `<img src={cam.stream_url}>` MJPEG feeds. Detection overlays are
drawn **server-side** — you do not position bounding boxes yourself. That is
deliberate: synchronising box coordinates against video frame timing is a
notorious time sink and there is no budget for it.

Use the `detection` WebSocket message for a live sidebar of recent detections per
camera, colour-coded by track state (green tracked / yellow new / red lost).

**Acceptance:** four feeds render, detections stream in, no console errors.

### 3. City map — THE HERO VIEW

deck.gl over MapLibre GL (no API key needed — use a free style URL).

- **`TripsLayer`** from `getTrajectories()`. The `path` field is already
  `[lon, lat, seconds]` triples, exactly what `TripsLayer` consumes — no
  transformation needed. Timestamps are guaranteed monotonically increasing.
- **`HeatmapLayer`** for congestion, from the `analytics` WebSocket message.
- **`IconLayer`** for cameras, styled by `status`.
- Click a trajectory → panel with plate, every sighting, and each hop's `method`
  and confidence.
- Handle `trajectory_update` to extend paths live.

**Acceptance:** trajectories visibly draw themselves along the route; clicking
one opens its detail; congestion heat shifts as `analytics` messages arrive.

### 4. Analytics view

`getAnalytics()` (omit the camera for city-wide) plus live `analytics` messages.
Recharts: vehicle count time-series, type breakdown pie from `by_type`,
congestion trend (0-100), speed distribution, busiest-intersection leaderboard.

**Acceptance:** charts populate from the 12 fixture buckets and update live.

### 5. Vehicle search

`search(plate)` → trajectory on the map, timeline of every sighting, last-seen
location and time. Handle the empty result cleanly.

**Acceptance:** searching `KA05MR7821` shows a route; junk shows an empty state,
not an error.

### 6. Alerts panel

`getAlerts(false)` plus live `alert` messages. Severity colouring (`info` /
`warn` / `critical`), kind labels (`stationary`, `wrong_way`, `volume_spike`,
`watchlist`), acknowledge button calling `ackAlert(id)`.

### 7. Camera network data — YOUR DELIVERABLE TO THE BACKEND

Author the real camera topology and hand it over: per camera an id, name,
lat/lon and heading; per ordered camera pair a `distance_m` and `travel_time_s`.

**This is not busywork.** `travel_time_s` is the data Layer 3 of the
cross-camera matching engine uses to reject physically impossible matches. The
engine cannot be tested without it. Author it alongside the map that visualizes
it so the two never drift, and get it to the other developer early.

### 8. Polish, last

Loading and empty states, responsive layout, smooth transitions.

**Polish comes last on purpose.** A beautiful dashboard showing a false
trajectory is worse than a plain one showing a true one.

## WebSocket rules

Every message is `{ type, data }` and the union is typed as `ServerMessage`.
**Switch on `type` and ignore unknown types** — that rule lets the server add
message types without breaking you. `connect()` already reconnects; the demo
must survive a dropped socket without a page reload.

## Rules

- Never construct an API path outside `src/lib/api.ts`.
- `npm run check` must pass before every push.
- Never commit `node_modules/` or `.next/` — gitignored, keep it that way.
- If you need a field the API does not return, ask for it. Do not scrape it out
  of another endpoint.
- Prefer the simplest thing that works. 36-hour build: no state management
  library until plain React state actually fails, no component abstraction with
  one use.
- Zoom-legible text: plate numbers must be readable from the back of a room.

## Merging

```bash
npm run check
git add -A && git commit -m "feat(ui): <what>"
git pull --rebase origin main
git push -u origin dev-b-dashboard
```

Commit small and push often — a four-hour local branch is how you get a merge
conflict in a 36-hour build. Your files and the server's are disjoint, so
conflicts should be near zero. Open a PR into `main` as each view's acceptance
check passes; do not wait until everything is done.
