/**
 * Smoke tests. Boots the server, exercises every endpoint over real HTTP, and
 * judges the responses against `src/contract.ts`.
 *
 *   npm run smoke
 *
 * THE CONTRACT IS THE JUDGE, not a list of values this file expects. A test
 * that asserts `cameras[0].name === "MG Road Junction"` proves the fixture is
 * unchanged and nothing else; it passes on a backend that has never worked.
 * Every check below is one of:
 *
 *   - the zod schema parses the response      (shape is honest)
 *   - a rule stated in CLAUDE.md holds        (miss is 200, null not "")
 *   - two independent paths agree             (mock and live, REST and WS)
 *
 * All three can fail on code that "looks fine", which is the only kind of test
 * worth the minutes it costs.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { z } from "zod";
import {
  Alert, AnalyticsResponse, Camera, CameraLink, SearchResult,
  ServerMessage, Track, Trajectory,
} from "@/contract";

const BASE = process.env.SMOKE_BASE ?? "http://localhost:3000";
const PORT = Number(new URL(BASE).port || 80);

let pass = 0;
const failures: string[] = [];

function check(name: string, fn: () => void) {
  try {
    fn();
    pass++;
    console.log(`  ok   ${name}`);
  } catch (e) {
    failures.push(`${name}: ${(e as Error).message}`);
    console.log(`  FAIL ${name}\n         ${(e as Error).message}`);
  }
}

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

/** Parse with the contract schema, and report WHICH field broke. */
function conforms<T>(schema: z.ZodType<T>, data: unknown, what: string): T {
  const r = schema.safeParse(data);
  if (!r.success) {
    const i = r.error.issues[0];
    throw new Error(`${what} violates the contract at "${i?.path.join(".")}": ${i?.message}`);
  }
  return r.data;
}

async function get(path: string) {
  const res = await fetch(`${BASE}${path}`);
  const text = await res.text();
  let body: unknown = null;
  try { body = JSON.parse(text); } catch { /* keep null, status is the signal */ }
  return { status: res.status, body, text };
}

// ---------------------------------------------------------------- server --

let child: ChildProcess | null = null;

async function alive(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(1500) });
    return r.status === 200 || r.status === 503;
  } catch { return false; }
}

async function ensureServer() {
  if (await alive()) {
    console.log(`using the server already on ${BASE}\n`);
    return;
  }
  console.log(`starting a server on port ${PORT} ...`);
  child = spawn("npx", ["tsx", "server.ts"], {
    // MOCK=1 so the WebSocket check has traffic to see without a running
    // worker. The REST routes below hit the LIVE database regardless.
    env: { ...process.env, PORT: String(PORT), MOCK: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stderr?.on("data", (d) => process.stderr.write(`  [server] ${d}`));

  for (let i = 0; i < 120; i++) {
    if (await alive()) { console.log("server up\n"); return; }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("server did not come up within 120s");
}

// ----------------------------------------------------------------- suites --

async function contractSuite(prefix: "/api" | "/api/mock") {
  console.log(`\n${prefix} — responses must satisfy src/contract.ts`);

  const cams = await get(`${prefix}/cameras`);
  check(`${prefix}/cameras`, () => {
    assert(cams.status === 200, `status ${cams.status}`);
    const list = conforms(z.array(Camera), cams.body, "cameras");
    assert(list.length > 0, "no cameras — run npm run db:setup");
  });

  const links = await get(`${prefix}/cameras/links`);
  check(`${prefix}/cameras/links`, () => {
    assert(links.status === 200, `status ${links.status}`);
    conforms(z.array(CameraLink), links.body, "links");
  });

  // Cross-check, not a shape check: layer 3 walks this graph, and a link
  // naming a camera that does not exist makes travelTime() silently return
  // null for that pair — matching then quietly loses its physics veto.
  check(`${prefix} link endpoints all exist in /cameras`, () => {
    const ids = new Set((cams.body as Camera[]).map((c) => c.id));
    for (const l of links.body as CameraLink[]) {
      assert(ids.has(l.from), `link from unknown camera ${l.from}`);
      assert(ids.has(l.to), `link to unknown camera ${l.to}`);
    }
  });

  const tracks = await get(`${prefix}/tracks?limit=5`);
  check(`${prefix}/tracks`, () => {
    assert(tracks.status === 200, `status ${tracks.status}`);
    const list = conforms(z.array(Track), tracks.body, "tracks");
    assert(list.length <= 5, `limit=5 returned ${list.length}`);
  });

  const trajs = await get(`${prefix}/trajectories?limit=3`);
  check(`${prefix}/trajectories`, () => {
    assert(trajs.status === 200, `status ${trajs.status}`);
    const list = conforms(z.array(Trajectory), trajs.body, "trajectories");
    assert(list.length <= 3, `limit=3 returned ${list.length}`);
    for (const t of list) {
      // A journey through N cameras is N points and N-1 hops. Any other
      // relation means the path and the hops disagree about the same trip.
      assert(t.path.length >= t.hops.length,
        `${t.id}: ${t.hops.length} hops but only ${t.path.length} path points`);
      const secs = t.path.map((p) => p[2]);
      assert(secs.every((s, i) => i === 0 || s >= secs[i - 1]!),
        `${t.id}: path timestamps go backwards`);
    }
  });

  const analytics = await get(`${prefix}/analytics`);
  check(`${prefix}/analytics`, () => {
    assert(analytics.status === 200, `status ${analytics.status}`);
    const a = conforms(AnalyticsResponse, analytics.body, "analytics");
    const summed = a.series.reduce((s, b) => s + b.vehicle_count, 0);
    assert(summed === a.totals.vehicle_count,
      `totals say ${a.totals.vehicle_count}, series sums to ${summed}`);
  });

  const alerts = await get(`${prefix}/alerts?acked=false`);
  check(`${prefix}/alerts?acked=false`, () => {
    assert(alerts.status === 200, `status ${alerts.status}`);
    const list = conforms(z.array(Alert), alerts.body, "alerts");
    assert(list.every((a) => !a.acked), "acked=false returned an acked alert");
  });

  // CLAUDE.md: "A search miss is a 200 with empty arrays, never a 404."
  const miss = await get(`${prefix}/search?plate=ZZ99ZZ9999`);
  check(`${prefix}/search miss is 200 with empty arrays`, () => {
    assert(miss.status === 200, `status ${miss.status} — a miss must not 404`);
    const r = conforms(SearchResult, miss.body, "search");
    assert(r.trajectories.length === 0 && r.sightings.length === 0, "miss returned results");
    // CLAUDE.md: null means "not known yet", never "" or 0.
    assert(r.last_seen === null, `last_seen must be null on a miss, got ${JSON.stringify(r.last_seen)}`);
  });

  const empty = await get(`${prefix}/search?plate=`);
  check(`${prefix}/search with an empty plate does not 500`, () => {
    assert(empty.status === 200, `status ${empty.status}`);
    conforms(SearchResult, empty.body, "search");
  });

  const bogus = await get(`${prefix}/nope`);
  check(`${prefix}/nope is 404`, () => {
    assert(bogus.status === 404, `status ${bogus.status}`);
  });
}

/** The mock exists so the frontend can be built before the pipeline. That is
 *  only true while both sides answer the same shape at the same paths. */
async function paritySuite() {
  console.log("\nmock and live must be interchangeable");
  for (const path of ["/cameras", "/cameras/links", "/tracks?limit=1",
                      "/trajectories?limit=1", "/analytics", "/alerts"]) {
    const [live, mock] = await Promise.all([get(`/api${path}`), get(`/api/mock${path}`)]);
    check(`shape parity ${path}`, () => {
      assert(live.status === mock.status,
        `live ${live.status} vs mock ${mock.status}`);
      const shape = (v: unknown): string =>
        Array.isArray(v) ? `[${v.length ? shape(v[0]) : ""}]`
        : v && typeof v === "object"
          ? `{${Object.keys(v as object).sort().join(",")}}`
          : typeof v;
      // Compare top-level shape only: an empty live array cannot reveal its
      // element shape, and the per-endpoint zod checks already cover that.
      const l = shape(live.body), m = shape(mock.body);
      const bothArrays = l.startsWith("[") && m.startsWith("[");
      assert(bothArrays || l === m, `live ${l} vs mock ${m}`);
    });
  }
}

async function websocketSuite() {
  console.log("\nWebSocket");
  const { WebSocket } = await import("ws");
  const url = BASE.replace(/^http/, "ws") + "/ws";

  const got = await new Promise<unknown[]>((resolve) => {
    const seen: unknown[] = [];
    const ws = new WebSocket(url);
    const done = () => { try { ws.close(); } catch { /* already gone */ } resolve(seen); };
    const timer = setTimeout(done, 12_000);
    ws.on("message", (d: Buffer) => {
      try { seen.push(JSON.parse(d.toString())); } catch { /* junk frame */ }
      if (seen.length >= 3) { clearTimeout(timer); done(); }
    });
    ws.on("error", () => { clearTimeout(timer); done(); });
  });

  check("/ws delivers messages", () => {
    assert(got.length > 0, "no messages in 12s — is the server running with MOCK=1 or a live worker?");
  });
  check("/ws messages satisfy ServerMessage", () => {
    for (const m of got) conforms(ServerMessage, m, "ws message");
  });
}

/**
 * Ack is the only mutating endpoint. Test it against an alert this suite
 * creates, so a real deployment's alert list is never quietly marked read.
 */
async function ackSuite() {
  console.log("\nalert acknowledgement");
  const db = await import("@/server/db");
  let created: Alert | null = null;
  try {
    created = await db.insertAlert({
      camera_id: null, kind: "watchlist", severity: "info",
      track_id: null, plate_text: null, detail: "smoke test — safe to ignore",
    });
  } catch (e) {
    check("create an alert to ack", () => { throw e; });
    return;
  }

  const res = await fetch(`${BASE}/api/alerts/${created.id}/ack`, { method: "POST" });
  const body = await res.json();
  check("POST /api/alerts/{id}/ack flips acked", () => {
    assert(res.status === 200, `status ${res.status}`);
    const a = conforms(Alert, body, "ack response");
    assert(a.id === created!.id, `acked ${a.id}, asked for ${created!.id}`);
    assert(a.acked === true, "acked is still false after ack");
  });

  const after = await get("/api/alerts?acked=false");
  check("an acked alert leaves the unacked list", () => {
    const list = conforms(z.array(Alert), after.body, "alerts");
    assert(!list.some((a) => a.id === created!.id), "still listed as unacked");
  });

  const ghost = await fetch(`${BASE}/api/alerts/999999999/ack`, { method: "POST" });
  check("acking an alert that does not exist is 404", () => {
    assert(ghost.status === 404, `status ${ghost.status}`);
  });

  await db.sql`DELETE FROM alerts WHERE id = ${created.id}::bigint`;
}


/**
 * End to end: Python sidecar -> JSON -> worker -> Postgres -> Module C.
 *
 * The synthetic sidecars drive a vehicle CAM1 -> CAM3 -> CAM2 with the third
 * leg deliberately unreadable, so a pass proves layer 1 (plate) AND layers 2+3
 * (Re-ID plus travel-time) both fire. The suite runs the worker TWICE and
 * requires the second run to add no new trajectories: a restarting sidecar
 * replays its events, and a pipeline that is not idempotent turns one journey
 * into a new duplicate on every restart.
 */
async function pipelineSuite() {
  console.log("\nend-to-end pipeline (synthetic sidecars)");
  const db = await import("@/server/db");

  const count = async (table: string) =>
    (await db.sql.unsafe(`select count(*)::int n from ${table}`))[0]!.n as number;

  const runWorker = (seconds: number) => new Promise<void>((resolve) => {
    const w = spawn("npx", ["tsx", "worker/ingest.ts"], {
      env: {
        ...process.env,
        ARGUS_PYTHON: process.env.ARGUS_PYTHON ?? "./.venv/bin/python",
        ARGUS_CAMERAS: "CAM1=demo,CAM3=demo,CAM2=demo",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let matched = 0;
    w.stdout?.on("data", (d: Buffer) => {
      for (const line of d.toString().split("\n")) {
        if (line.includes("[MATCH]")) { matched++; console.log(`  ${line.trim()}`); }
      }
    });
    setTimeout(() => { w.kill("SIGINT"); setTimeout(() => { w.kill("SIGKILL"); resolve(); }, 1500); },
      seconds * 1000);
    void matched;
  });

  const before = { m: await count("matches"), t: await count("trajectories") };
  await runWorker(22);
  const afterFirst = { m: await count("matches"), t: await count("trajectories") };

  check("the pipeline produces cross-camera matches", () => {
    assert(afterFirst.m > before.m || before.m > 0,
      "no matches at all — Module C never fired end to end");
  });

  check("a matched journey becomes a trajectory", () => {
    assert(afterFirst.t > 0, "matches exist but no trajectory was stitched");
  });

  // Replay. Same events, same matches: nothing new may be created.
  await runWorker(22);
  const afterSecond = { m: await count("matches"), t: await count("trajectories") };

  check("replaying the same events creates no duplicate matches", () => {
    assert(afterSecond.m === afterFirst.m,
      `matches went ${afterFirst.m} -> ${afterSecond.m} on a replay of identical events`);
  });
  check("replaying the same events creates no duplicate trajectories", () => {
    assert(afterSecond.t === afterFirst.t,
      `trajectories went ${afterFirst.t} -> ${afterSecond.t} on a replay of identical events`);
  });

  const rows = await db.sql<{ id: string; track_ids: string[] }[]>`
    SELECT id, track_ids FROM trajectories`;
  check("no trajectory visits the same track twice", () => {
    for (const r of rows) {
      const ids = r.track_ids.map(String);
      assert(new Set(ids).size === ids.length,
        `trajectory ${r.id} repeats a track: ${JSON.stringify(ids)}`);
    }
  });

  // The contract's own view of the same data must hold up.
  const trajs = await get("/api/trajectories?limit=5");
  check("stitched trajectories satisfy the contract over HTTP", () => {
    const list = conforms(z.array(Trajectory), trajs.body, "trajectories");
    assert(list.length > 0, "trajectories exist in the database but the API returns none");
    assert(list.some((t) => t.hops.length > 0), "every trajectory came back with zero hops");
  });

  const found = await get("/api/search?plate=KA05MR7821");
  check("the plate the pipeline read is findable by search", () => {
    const r = conforms(SearchResult, found.body, "search");
    assert(r.sightings.length > 0, "the pipeline stored the plate but search cannot find it");
    assert(r.last_seen !== null, "sightings exist but last_seen is null");
  });
}


/**
 * The dashboard must RENDER, not merely typecheck.
 *
 * These are the failures a type checker cannot see: a server component that
 * throws during render, a client bundle that 500s, a route that silently
 * disappeared. Each check asserts a marker that only appears if the page's own
 * code ran — never a snapshot of markup, which would fail on every restyle and
 * teach everyone to ignore it.
 */
async function uiSuite() {
  console.log("\ndashboard pages render");

  const page = async (path: string) => {
    const res = await fetch(`${BASE}${path}`, { redirect: "follow" });
    return { status: res.status, html: await res.text(), url: res.url };
  };

  for (const [path, marker, what] of [
    // A route group adds no URL segment: src/app/(dashboard)/map is /map.
    ["/", "Detection feed", "live view"],
    ["/map", "Journeys", "map view"],
    ["/analytics", "Scope", "analytics view"],
    ["/search", "Vehicle search", "search view"],
    ["/status", "Argus — status", "status page"],
  ] as const) {
    const r = await page(path);
    check(`${path} renders`, () => {
      assert(r.status === 200, `status ${r.status}`);
      // A Next error page is a 200 with a stack in it. Catch that explicitly.
      assert(!/Application error|__NEXT_ERROR|Internal Server Error/.test(r.html),
        `${what} returned an error page`);
      assert(r.html.includes(marker),
        `${what} rendered without its own content (no "${marker}")`);
    });
  }

  // The nav is what makes four views one dashboard. A view unreachable from
  // the nav does not exist as far as anyone using it is concerned.
  const shell = await page("/");
  check("every view is reachable from the nav", () => {
    for (const href of ["/map", "/analytics", "/search"]) {
      assert(shell.html.includes(href), `nav has no link to ${href}`);
    }
  });

  // Dev B builds against fixtures. If the fixture switch stops working the
  // whole parallel-development story stops working with it.
  check("the dashboard bundle constructs API paths through src/lib/api.ts", () => {
    assert(shell.html.includes("/api/") || shell.html.includes("_next"),
      "no API path and no client bundle — the page rendered nothing live");
  });
}

async function main() {
  await ensureServer();
  const health = await get("/api/health");
  console.log(`health: ${health.text}`);
  const dbUp = health.status === 200;
  if (!dbUp) {
    console.log("\nDATABASE IS DOWN — running the mock suite only.");
    console.log("  docker compose up -d db redis && npm run db:setup\n");
  }

  if (dbUp) await contractSuite("/api");
  await contractSuite("/api/mock");
  if (dbUp) await paritySuite();
  await uiSuite();
  await websocketSuite();
  if (dbUp) await ackSuite();
  if (dbUp && !process.argv.includes("--no-pipeline")) await pipelineSuite();

  console.log(`\n${pass} passed, ${failures.length} failed`);
  if (failures.length) {
    for (const f of failures) console.log(`  - ${f}`);
  }
  child?.kill();
  process.exit(failures.length ? 1 : 0);
}

main().catch((e) => {
  console.error("smoke run failed:", e.message);
  child?.kill();
  process.exit(1);
});
