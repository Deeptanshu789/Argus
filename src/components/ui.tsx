"use client";
/**
 * Shared chrome. One file, because a component-per-file layout for six small
 * primitives buys nothing at this size and costs six imports everywhere.
 */
import type { CSSProperties, ReactNode } from "react";

/**
 * Every colour in the application, and none of them are colours.
 *
 * These were hex literals, which meant one hardcoded dark palette and no way
 * to offer a light one -- a control room runs either. Pointing each key at a
 * custom property from src/app/globals.css re-themes all six views without
 * touching a single one of them, and lets an explicit `data-theme` on <html>
 * win over the operating system in both directions.
 *
 * The KEYS are unchanged on purpose. Every view already styles itself through
 * this object, so renaming them would turn a theming change into a rewrite of
 * 1,200 lines of JSX for no gain.
 */
export const T = {
  bg: "var(--bg)",
  panel: "var(--panel)",
  raised: "var(--panel2)",
  sunk: "var(--sunk)",
  line: "var(--line)",
  line2: "var(--line2)",
  text: "var(--ink)",
  dim: "var(--ink2)",
  faint: "var(--ink3)",
  accent: "var(--accent)",
  accentSoft: "var(--accentSoft)",
  ok: "var(--ok)",
  info: "var(--info)",
  warn: "var(--warn)",
  bad: "var(--crit)",
  /** Layer 2 of Module C. Distinct from the accent so a Re-ID match never
   *  reads as a plate match at a glance. */
  reid: "var(--info)",
  shadow: "var(--shadow)",
};

/** Seven steps, cool to hot. Congestion and confidence both ride this. */
export const RAMP = ["var(--r1)", "var(--r2)", "var(--r3)", "var(--r4)",
                     "var(--r5)", "var(--r6)", "var(--r7)"];

/** A 0-1 score to a ramp step. Clamped, because a NaN confidence must pick a
 *  colour rather than render nothing at all. */
export const rampFor = (score: number) =>
  RAMP[Math.min(RAMP.length - 1, Math.max(0, Math.round((score || 0) * (RAMP.length - 1))))]!;

export const MONO = "'IBM Plex Mono', ui-monospace, monospace";

export function Panel(
  { title, right, children, style }:
  { title?: string; right?: ReactNode; children: ReactNode; style?: CSSProperties },
) {
  return (
    <section style={{
      background: T.panel, border: `1px solid ${T.line}`, borderRadius: 6,
      padding: "1rem 1.15rem", boxShadow: T.shadow, ...style,
    }}>
      {title && (
        <header style={{ display: "flex", justifyContent: "space-between",
                         alignItems: "center", marginBottom: ".8rem" }}>
          <h2 style={{ margin: 0, fontSize: 9, letterSpacing: ".12em", fontWeight: 500,
                       fontFamily: MONO,
                       textTransform: "uppercase", color: T.faint }}>{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export const Dot = ({ colour, pulse }: { colour: string; pulse?: boolean }) => (
  <span style={{
    display: "inline-block", width: 7, height: 7, borderRadius: 4,
    background: colour, marginRight: 8, flex: "0 0 auto",
    animation: pulse ? "apulse 1.9s infinite" : undefined,
  }} />
);

export function Tag({ children, colour = T.dim }: { children: ReactNode; colour?: string }) {
  return (
    <span style={{
      fontSize: 10, padding: "1px 6px", borderRadius: 3, whiteSpace: "nowrap",
      fontFamily: MONO, letterSpacing: ".04em",
      background: "color-mix(in srgb, currentColor 12%, transparent)",
      color: colour, border: `1px solid color-mix(in srgb, ${colour} 34%, transparent)`,
    }}>{children}</span>
  );
}

/** A plate, treated the same everywhere. `null` is the system saying it could
 *  not read one -- shown as absent rather than blank, because on real footage
 *  a third of tracked vehicles never get a plate and an empty cell reads as a
 *  bug rather than an ordinary state. */
export function Plate({ text, size = 13 }: { text: string | null; size?: number }) {
  if (!text) {
    return <span style={{ fontFamily: MONO, fontSize: size - 1, color: T.faint }}>no plate</span>;
  }
  return (
    <span className="plate" style={{ fontSize: size, fontWeight: 500, color: T.text }}>
      {text}
    </span>
  );
}

export const statusColour = (s: string) =>
  s === "online" ? T.ok : s === "degraded" ? T.warn : T.faint;

export const severityColour = (s: string) =>
  s === "critical" ? T.bad : s === "warn" ? T.warn : T.info;

/** Layer 1 and layers 2-3 must LOOK different. Which layer confirmed a match is
 *  the whole story of Module C, and a uniform badge hides it. */
export const methodColour = (m: string) =>
  m === "plate" ? T.accent : m === "reid" ? T.reid : T.ok;

/**
 * The 2-of-3 rule, as one glyph.
 *
 * Module C confirms a match only when two of three independent layers agree,
 * and which two is the most informative fact the interface carries. Rendered
 * as a bare string ("plate+reid+spatial_temporal") it is unreadable at a
 * glance and unreadable at all in a dense table. Three fixed slots, filled or
 * hollow, in a fixed order: the shape alone says how well corroborated a hop
 * is, and the colours say which layers did the corroborating.
 */
export function Layers({ method }: { method: string }) {
  const on = new Set(method.split("+"));
  const slots: [string, string][] = [
    ["plate", T.accent], ["reid", T.reid], ["spatial_temporal", T.ok],
  ];
  return (
    <span title={method} style={{ display: "inline-flex", gap: 3, alignItems: "center" }}>
      {slots.map(([key, colour]) => (
        <span key={key} style={{
          width: 6, height: 12, borderRadius: 1,
          background: on.has(key) ? colour : "transparent",
          border: `1px solid ${on.has(key) ? colour : T.line2}`,
        }} />
      ))}
    </span>
  );
}

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
