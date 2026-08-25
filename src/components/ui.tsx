"use client";
/**
 * Shared chrome. One file, because a component-per-file layout for six small
 * primitives buys nothing at this size and costs six imports everywhere.
 */
import type { CSSProperties, ReactNode } from "react";

export const T = {
  bg: "#0b0f14", panel: "#111820", raised: "#161f2a", line: "#22303d",
  text: "#e6edf3", dim: "#8b9bab", faint: "#5b6b7b",
  accent: "#6cb6ff", ok: "#3fb950", warn: "#d29922", bad: "#f85149", reid: "#d2a8ff",
};

export function Panel(
  { title, right, children, style }:
  { title?: string; right?: ReactNode; children: ReactNode; style?: CSSProperties },
) {
  return (
    <section style={{
      background: T.panel, border: `1px solid ${T.line}`, borderRadius: 10,
      padding: "1rem 1.15rem", ...style,
    }}>
      {title && (
        <header style={{ display: "flex", justifyContent: "space-between",
                         alignItems: "center", marginBottom: ".8rem" }}>
          <h2 style={{ margin: 0, fontSize: 12, letterSpacing: ".09em",
                       textTransform: "uppercase", color: T.dim }}>{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export const Dot = ({ colour, pulse }: { colour: string; pulse?: boolean }) => (
  <span style={{
    display: "inline-block", width: 8, height: 8, borderRadius: 4,
    background: colour, marginRight: 8, flex: "0 0 auto",
    boxShadow: pulse ? `0 0 0 4px ${colour}22` : undefined,
  }} />
);

export function Tag({ children, colour = T.dim }: { children: ReactNode; colour?: string }) {
  return (
    <span style={{
      fontSize: 11, padding: "1px 7px", borderRadius: 4, whiteSpace: "nowrap",
      background: `${colour}22`, color: colour, border: `1px solid ${colour}44`,
    }}>{children}</span>
  );
}

export const statusColour = (s: string) =>
  s === "online" ? T.ok : s === "degraded" ? T.warn : T.bad;

export const severityColour = (s: string) =>
  s === "critical" ? T.bad : s === "warn" ? T.warn : T.accent;

/** Layer 1 and layers 2-3 must LOOK different. Which layer confirmed a match is
 *  the whole story of Module C, and a uniform badge hides it. */
export const methodColour = (m: string) =>
  m === "plate" ? T.accent : m === "reid" ? T.reid : T.ok;

export function Empty({ children }: { children: ReactNode }) {
  return <p style={{ color: T.faint, fontSize: 13, margin: 0, lineHeight: 1.6 }}>{children}</p>;
}

/** Seconds since an ISO timestamp, as something a human reads at a glance. */
export function ago(iso: string): string {
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}
