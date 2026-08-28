"use client";
/**
 * Dashboard shell: a fixed rail on the left, a stat header across the top.
 *
 * This was a single horizontal nav bar. An operator watching six views needs
 * the view list and the fleet's health visible at the same time as whatever
 * they are reading, which a top bar cannot do without stealing the height the
 * camera grid needs.
 *
 * Client component because the WebSocket badge is live. The views below fetch
 * their own data — nothing is threaded through here, so one slow view cannot
 * stall the others.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Dot, MONO, T } from "@/components/ui";
import { useLive } from "@/components/useLive";
import { getCameras } from "@/lib/api";
import type { Camera } from "@/contract";

// A route GROUP — the parenthesised folder — adds no URL segment. These views
// live at /, /map, /analytics, /search: the dashboard is the application, not a
// section of it. `(dashboard)` only groups them under this layout.
const VIEWS = [
  { href: "/", label: "Live", sub: "camera grid · detections" },
  { href: "/map", label: "Map", sub: "trajectories over the city" },
  { href: "/analytics", label: "Analytics", sub: "counts, speed, congestion" },
  { href: "/search", label: "Search", sub: "one plate, every sighting" },
  { href: "/upload", label: "Upload", sub: "video in, same pipeline" },
  { href: "/devices", label: "Devices", sub: "pair a phone as a camera" },
];

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  // Read what the inline boot script in src/app/layout.tsx already decided,
  // rather than deciding again — a second opinion here would flip the theme
  // one frame after paint, which is the flash that script exists to prevent.
  useEffect(() => {
    const now = document.documentElement.getAttribute("data-theme");
    if (now === "light" || now === "dark") setTheme(now);
  }, []);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("argus-theme", next); } catch { /* private mode */ }
  };
  return { theme, toggle };
}

const railLabel: React.CSSProperties = {
  font: `500 9px/1 ${MONO}`, letterSpacing: ".12em",
  color: T.faint, textTransform: "uppercase",
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const live = useLive();
  const { theme, toggle } = useTheme();
  const [cameras, setCameras] = useState<Camera[]>([]);

  // The fleet counts in the rail. Polled, not pushed: camera status changes on
  // the scale of a sidecar restart, and a WebSocket message per poll interval
  // would cost more than it tells anyone.
  useEffect(() => {
    let alive = true;
    const pull = () => { getCameras().then((c) => { if (alive) setCameras(c); }).catch(() => {}); };
    pull();
    const t = setInterval(pull, 15_000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const fleet = {
    online: cameras.filter((c) => c.status === "online").length,
    degraded: cameras.filter((c) => c.status === "degraded").length,
    offline: cameras.filter((c) => c.status === "offline").length,
  };
  const view = VIEWS.find((v) => (v.href === "/" ? path === "/" : path.startsWith(v.href)));

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: T.bg, color: T.text }}>
      <aside style={{
        width: 198, flex: "0 0 198px", borderRight: `1px solid ${T.line}`,
        background: T.panel, display: "flex", flexDirection: "column",
        position: "sticky", top: 0, height: "100vh",
      }}>
        <div style={{ padding: "16px 16px 14px", borderBottom: `1px solid ${T.line}` }}>
          <Link href="/status" title="System status"
                style={{ display: "flex", alignItems: "center", gap: 8, color: T.text }}>
            <span style={{ width: 11, height: 11, border: `2px solid ${T.text}`,
                           borderRadius: "50%", position: "relative" }}>
              <span style={{ position: "absolute", inset: 2, background: T.text,
                             borderRadius: "50%" }} />
            </span>
            <span style={{ font: "600 17px/1 'IBM Plex Sans'", letterSpacing: ".14em" }}>
              ARGUS
            </span>
          </Link>
          <div style={{ marginTop: 7, font: `400 9px/1.5 ${MONO}`, letterSpacing: ".1em",
                        color: T.faint, textTransform: "uppercase" }}>
            ANPR · trajectory<br />SIH26127 · BEL
          </div>
        </div>

        <nav style={{ padding: "10px 8px", display: "flex", flexDirection: "column", gap: 2 }}>
          {VIEWS.map((v, i) => {
            const active = v.href === "/" ? path === "/" : path.startsWith(v.href);
            return (
              <Link key={v.href} href={v.href} style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "7px 8px", borderRadius: 4, fontSize: 12.5,
                textDecoration: "none",
                color: active ? T.text : T.dim,
                background: active ? T.accentSoft : "transparent",
              }}>
                <span style={{
                  width: 2, height: 13, borderRadius: 1,
                  background: active ? T.accent : "transparent",
                }} />
                <span style={{ font: `400 9.5px/1 ${MONO}`, color: T.faint, width: 14 }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span style={{ flex: 1 }}>{v.label}</span>
              </Link>
            );
          })}
        </nav>

        <div style={{ marginTop: "auto", padding: "12px 16px", borderTop: `1px solid ${T.line}` }}>
          <div style={railLabel}>Fleet</div>
          <div style={{ marginTop: 9, display: "flex", flexDirection: "column", gap: 6 }}>
            {([["online", fleet.online, T.ok],
               ["degraded", fleet.degraded, T.warn],
               ["offline", fleet.offline, T.line2]] as const).map(([label, n, colour]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 7,
                                        font: "400 11.5px/1 'IBM Plex Sans'" }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: colour }} />
                <span style={{ flex: 1, color: T.dim }}>{label}</span>
                <span className="tnum" style={{ font: `500 11.5px ${MONO}` }}>{n}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 14, paddingTop: 12, borderTop: `1px solid ${T.line}`,
                        font: `400 9.5px/1.7 ${MONO}`, color: T.faint }}>
            5 fps/cam · CPU only<br />plate-k12 · PaddleOCR
          </div>
        </div>
      </aside>

      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        <header style={{
          height: 58, flex: "0 0 58px", borderBottom: `1px solid ${T.line}`,
          background: T.panel, display: "flex", alignItems: "stretch",
          position: "sticky", top: 0, zIndex: 20,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 18px",
                        borderRight: `1px solid ${T.line}`, minWidth: 236 }}>
            <Dot colour={live.connected ? T.ok : T.faint} pulse={live.connected} />
            <div>
              <div style={{ font: "500 13px/1.2 'IBM Plex Sans'" }}>{view?.label ?? "Argus"}</div>
              <div style={{ font: `400 9.5px/1.2 ${MONO}`, color: T.faint, letterSpacing: ".06em" }}>
                {live.connected ? (view?.sub ?? "") : "waiting for events"}
              </div>
            </div>
          </div>

          <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "stretch",
                        overflowX: "auto" }}>
            <Stat label="Vehicles now" value={live.city ? String(live.city.vehicle_count) : "—"} />
            <Stat label="Avg speed"
                  value={live.city ? `${live.city.avg_speed_kmh}` : "—"} unit="km/h" />
            <Stat label="Cross-cam matches" value={String(live.matches.length)} />
            <Stat label="Open alerts" value={String(live.alerts.length)} last />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 16px",
                        borderLeft: `1px solid ${T.line}` }}>
            <button onClick={toggle} title="Switch theme" style={{
              background: T.raised, border: `1px solid ${T.line}`,
              borderRadius: 4, padding: "5px 10px", font: `400 10px/1 ${MONO}`,
              letterSpacing: ".08em", color: T.dim, textTransform: "uppercase",
            }}>
              {theme === "dark" ? "light" : "dark"}
            </button>
          </div>
        </header>

        <main style={{ padding: "1.25rem 1.5rem", flex: 1, minWidth: 0 }}>{children}</main>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, last }:
              { label: string; value: string; unit?: string; last?: boolean }) {
  return (
    <div style={{
      padding: "0 16px", height: "100%", display: "flex", flexDirection: "column",
      justifyContent: "center", gap: 3,
      borderRight: last ? undefined : `1px solid ${T.line}`,
    }}>
      <div style={{ font: `500 9px/1 ${MONO}`, letterSpacing: ".11em", color: T.faint,
                    textTransform: "uppercase", whiteSpace: "nowrap" }}>{label}</div>
      <div className="tnum" style={{ font: `500 14px/1 ${MONO}`, whiteSpace: "nowrap" }}>
        {value}
        {unit && <span style={{ font: `400 10px ${MONO}`, color: T.faint }}> {unit}</span>}
      </div>
    </div>
  );
}
