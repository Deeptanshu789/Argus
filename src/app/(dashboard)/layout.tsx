"use client";
/**
 * Dashboard shell. Nav plus a live connection indicator that every view shares.
 *
 * Client component because the WebSocket badge is live. The views below it
 * fetch their own data — nothing is threaded through here, so one slow view
 * cannot stall the others.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Dot, T } from "@/components/ui";
import { useLive } from "@/components/useLive";

// A route GROUP — the parenthesised folder — adds no URL segment. These views
// live at /, /map, /analytics, /search: the dashboard is the application, not a
// section of it. `(dashboard)` only groups them under this layout.
const VIEWS = [
  { href: "/", label: "Live" },
  { href: "/map", label: "Map" },
  { href: "/analytics", label: "Analytics" },
  { href: "/search", label: "Search" },
  { href: "/upload", label: "Upload" },
  { href: "/devices", label: "Devices" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const live = useLive();

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.text,
                  fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      <nav style={{
        display: "flex", alignItems: "center", gap: "1.5rem",
        padding: ".85rem 1.5rem", borderBottom: `1px solid ${T.line}`,
        background: T.panel, position: "sticky", top: 0, zIndex: 10,
      }}>
        <Link href="/status" style={{ color: T.text, textDecoration: "none",
                                      fontWeight: 600, letterSpacing: ".02em" }}
              title="System status">
          Argus
        </Link>
        <div style={{ display: "flex", gap: ".35rem" }}>
          {VIEWS.map((v) => {
            const active = v.href === "/" ? path === "/" : path.startsWith(v.href);
            return (
              <Link key={v.href} href={v.href} style={{
                padding: ".3rem .8rem", borderRadius: 6, fontSize: 13,
                textDecoration: "none",
                color: active ? T.text : T.dim,
                background: active ? T.raised : "transparent",
              }}>{v.label}</Link>
            );
          })}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center",
                      fontSize: 12, color: T.dim }}>
          <Dot colour={live.connected ? T.ok : T.faint} pulse={live.connected} />
          {live.connected ? "live" : "waiting for events"}
          {live.city && (
            <span style={{ marginLeft: "1.25rem" }}>
              {live.city.vehicle_count} vehicles · {live.city.avg_speed_kmh} km/h
            </span>
          )}
        </div>
      </nav>
      <main style={{ padding: "1.5rem", maxWidth: 1400, margin: "0 auto" }}>{children}</main>
    </div>
  );
}
