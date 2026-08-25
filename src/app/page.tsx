/**
 * Operator status page. NOT the dashboard — that is `src/app/(dashboard)` and
 * belongs to Dev B (map, camera grid, charts, search).
 *
 * This answers one question: is the system actually working right now? It is
 * what you open after `npm run dev` to see whether the pipeline is alive, which
 * is otherwise only visible by reading worker logs.
 *
 * Server component: it queries the API on the server, so opening this page
 * proves the route handlers, the database and the contract all work — with no
 * client JavaScript involved in that proof.
 */
import { headers } from "next/headers";
import type { Alert, Camera, Track, Trajectory } from "@/contract";

export const dynamic = "force-dynamic";

async function api<T>(path: string): Promise<T | null> {
  const h = await headers();
  const host = h.get("host") ?? "localhost:3000";
  try {
    const res = await fetch(`http://${host}/api${path}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

const C = {
  bg: "#0d1117", panel: "#161b22", line: "#30363d",
  text: "#e6edf3", dim: "#8b949e",
  ok: "#3fb950", warn: "#d29922", bad: "#f85149", link: "#6cb6ff",
};

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{
      background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8,
      padding: "1rem 1.25rem", marginBottom: "1rem",
    }}>
      <h2 style={{ margin: "0 0 .75rem", fontSize: 13, letterSpacing: ".08em",
                   textTransform: "uppercase", color: C.dim }}>{title}</h2>
      {children}
    </section>
  );
}

const dot = (colour: string) => (
  <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4,
                 background: colour, marginRight: 8 }} />
);

const td: React.CSSProperties = {
  padding: ".35rem .75rem .35rem 0", borderBottom: `1px solid ${C.line}`,
  fontSize: 13, whiteSpace: "nowrap",
};

export default async function Page() {
  const [health, cameras, tracks, trajectories, alerts] = await Promise.all([
    api<{ ok: boolean; database: string }>("/health"),
    api<Camera[]>("/cameras"),
    api<Track[]>("/tracks?limit=8"),
    api<Trajectory[]>("/trajectories?limit=5"),
    api<Alert[]>("/alerts?acked=false"),
  ]);

  const dbUp = health?.ok === true;
  const plates = (tracks ?? []).filter((t) => t.plate_text).length;

  return (
    <main style={{
      background: C.bg, color: C.text, minHeight: "100vh", padding: "2.5rem 2rem",
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <h1 style={{ margin: 0, fontSize: 28, letterSpacing: ".02em" }}>Argus</h1>
        <p style={{ color: C.dim, marginTop: ".25rem", fontSize: 13 }}>
          City-wide ANPR and cross-camera vehicle trajectory tracking · SIH26127
        </p>

        <Panel title="System">
          <div style={{ fontSize: 13, lineHeight: 2 }}>
            <div>{dot(dbUp ? C.ok : C.bad)}database {dbUp ? "connected" : "unreachable"}</div>
            <div>{dot(cameras?.length ? C.ok : C.warn)}
              {cameras?.length ?? 0} cameras configured</div>
            <div>{dot(C.ok)}websocket at <code>/ws</code></div>
          </div>
          {!dbUp && (
            <pre style={{ marginTop: ".75rem", color: C.warn, fontSize: 12 }}>
{`docker compose up -d db redis
npm run db:setup`}
            </pre>
          )}
        </Panel>

        <Panel title="Cameras">
          {cameras?.length ? (
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
              <tbody>
                {cameras.map((c) => (
                  <tr key={c.id}>
                    <td style={td}>
                      {dot(c.status === "online" ? C.ok
                         : c.status === "degraded" ? C.warn : C.bad)}
                      {c.id}
                    </td>
                    <td style={{ ...td, width: "100%" }}>{c.name}</td>
                    <td style={{ ...td, color: C.dim }}>{c.status}</td>
                    <td style={{ ...td, color: C.dim }}>
                      {c.lat.toFixed(4)}, {c.lon.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p style={{ color: C.dim, fontSize: 13, margin: 0 }}>
              None. Run <code>npm run db:setup</code>.
            </p>
          )}
          <p style={{ color: C.dim, fontSize: 12, marginBottom: 0 }}>
            Status is derived from the last detection, never a stored flag — a
            crashed sidecar cannot update a column saying it crashed.
          </p>
        </Panel>

        <Panel title={`Recent tracks — ${plates}/${tracks?.length ?? 0} with a plate`}>
          {tracks?.length ? (
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
              <tbody>
                {tracks.map((t) => (
                  <tr key={t.id}>
                    <td style={td}>{t.camera_id}</td>
                    <td style={{ ...td, color: t.plate_text ? C.text : C.dim }}>
                      {/* null means "not read yet", and it must LOOK different
                          from a read that came back empty. */}
                      {t.plate_text ?? "no plate read"}
                    </td>
                    <td style={{ ...td, color: C.dim }}>
                      {t.plate_conf === null ? "" : t.plate_conf.toFixed(2)}
                    </td>
                    <td style={{ ...td, color: C.dim }}>{t.vehicle_type}</td>
                    <td style={{ ...td, color: C.dim, width: "100%" }}>
                      {t.entry_time.replace("T", " ").replace(/\.\d+Z$/, "")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p style={{ color: C.dim, fontSize: 13, margin: 0 }}>
              Nothing yet. Start the pipeline:<br />
              <code style={{ color: C.link }}>
                ARGUS_CAMERAS=&apos;CAM1=demo,CAM3=demo&apos; npm run worker
              </code>
            </p>
          )}
        </Panel>

        <Panel title="Cross-camera journeys — Module C">
          {trajectories?.length ? (
            trajectories.map((t) => (
              <div key={t.id} style={{ fontSize: 13, padding: ".35rem 0",
                                       borderBottom: `1px solid ${C.line}` }}>
                <strong>{t.plate_text ?? "unidentified vehicle"}</strong>{" "}
                <span style={{ color: C.dim }}>
                  {t.hops.length
                    ? t.hops.map((h) => `${h.from_camera} to ${h.to_camera}`).join(", ")
                    : "single camera"}
                </span>
                {t.hops.map((h, i) => (
                  <span key={i} style={{
                    marginLeft: 8, fontSize: 11, padding: "1px 6px", borderRadius: 3,
                    background: h.method === "plate" ? "#1f6feb33" : "#8957e533",
                    color: h.method === "plate" ? C.link : "#d2a8ff",
                  }}>
                    {h.method} {h.confidence.toFixed(2)}
                  </span>
                ))}
              </div>
            ))
          ) : (
            <p style={{ color: C.dim, fontSize: 13, margin: 0 }}>
              No confirmed matches yet. A journey appears once two of the three
              layers — plate text, Re-ID similarity, travel-time feasibility —
              agree on the same vehicle at two cameras.
            </p>
          )}
        </Panel>

        <Panel title={`Open alerts — ${alerts?.length ?? 0}`}>
          {alerts?.length ? (
            alerts.slice(0, 6).map((a) => (
              <div key={a.id} style={{ fontSize: 13, padding: ".3rem 0" }}>
                {dot(a.severity === "critical" ? C.bad
                   : a.severity === "warn" ? C.warn : C.dim)}
                <strong>{a.kind}</strong>{" "}
                <span style={{ color: C.dim }}>{a.camera_id} · {a.detail}</span>
              </div>
            ))
          ) : (
            <p style={{ color: C.dim, fontSize: 13, margin: 0 }}>None.</p>
          )}
        </Panel>

        <Panel title="API">
          <div style={{ display: "flex", flexWrap: "wrap", gap: ".5rem 1.25rem", fontSize: 13 }}>
            {["/api/health", "/api/cameras", "/api/cameras/links", "/api/tracks?limit=5",
              "/api/trajectories?limit=3", "/api/search?plate=KA05MR7821",
              "/api/analytics", "/api/alerts?acked=false", "/api/mock/cameras"]
              .map((r) => (
                <a key={r} href={r} style={{ color: C.link, textDecoration: "none" }}>{r}</a>
              ))}
          </div>
          <p style={{ color: C.dim, fontSize: 12, marginBottom: 0 }}>
            Every response is validated against <code>src/contract.ts</code> before
            it is sent. <code>/api/mock/*</code> serves the same shapes from
            fixtures so the dashboard can be built without the pipeline running.
          </p>
        </Panel>
      </div>
    </main>
  );
}
