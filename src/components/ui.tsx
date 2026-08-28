"use client";
/**
 * Shared chrome. One file, because a component-per-file layout for a dozen
 * small primitives buys nothing at this size and costs a dozen imports
 * everywhere.
 *
 * These are the design's vocabulary, not generic widgets. Each one encodes a
 * rule the views would otherwise restate: a plate that could not be read looks
 * different from one that could, a hop shows WHICH layers agreed, a confidence
 * is a number and a bar because the number is exact and the bar is scannable.
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
 * this object, so renaming them would turn a theming change into a rewrite.
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

/** A 0-1 score to a ramp step. Clamped, because a NaN congestion must pick a
 *  colour rather than render nothing at all. */
export const rampFor = (score: number) =>
  RAMP[Math.min(RAMP.length - 1, Math.max(0, Math.round((score || 0) * (RAMP.length - 1))))]!;

export const MONO = "'IBM Plex Mono', ui-monospace, monospace";
export const SANS = "'IBM Plex Sans', system-ui, sans-serif";

/** The one label treatment: small, mono, uppercase, widely tracked. Used for
 *  every panel header and column head, so a label never competes with data. */
export const LABEL: CSSProperties = {
  font: `500 10px/1 ${MONO}`, letterSpacing: ".13em",
  color: T.faint, textTransform: "uppercase",
};

/** Meta text sitting opposite a label: the same size, not shouting. */
export const META: CSSProperties = { font: `400 10px ${MONO}`, color: T.faint };

/** A frame crop we do not have. Every place the design shows a thumbnail, the
 *  real system has no still to show -- the sidecar keeps no JPEG. A hatch says
 *  "image here" without pretending to be one. */
export const HATCH =
  "repeating-linear-gradient(135deg,var(--sunk) 0 4px,var(--panel2) 4px 8px)";

/* ------------------------------------------------------------- surfaces -- */

export function Panel(
  { title, right, children, style, bodyStyle, flush }:
  { title?: ReactNode; right?: ReactNode; children: ReactNode;
    style?: CSSProperties; bodyStyle?: CSSProperties; flush?: boolean },
) {
  return (
    <section style={{
      background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3,
      minWidth: 0, ...style,
    }}>
      {title !== undefined && (
        <header style={{
          padding: "10px 12px", borderBottom: `1px solid ${T.line}`,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          gap: 10,
        }}>
          <div style={LABEL}>{title}</div>
          {right}
        </header>
      )}
      <div style={flush ? bodyStyle : { padding: "12px 14px", ...bodyStyle }}>
        {children}
      </div>
    </section>
  );
}

/** A panel whose header explains itself in a sentence. The analytics charts
 *  need this and the tables do not -- a chart without a stated unit is a
 *  decoration. */
export function Head({ title, sub, right }:
                     { title: string; sub?: string; right?: ReactNode }) {
  return (
    <div style={{ padding: "12px 14px 10px", borderBottom: `1px solid ${T.line}`,
                  display: "flex", alignItems: "flex-start", gap: 12 }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ font: `600 14px/1.2 ${SANS}` }}>{title}</div>
        {sub && <div style={{ marginTop: 5, font: `400 11.5px/1.5 ${SANS}`,
                              color: T.dim }}>{sub}</div>}
      </div>
      {right}
    </div>
  );
}

/** One row in a list. `--rowpad` is set once on :root, so density is a single
 *  change rather than an edit in every table. */
export const rowStyle = (last?: boolean): CSSProperties => ({
  display: "flex", alignItems: "center", gap: 10,
  padding: "var(--rowpad) 12px",
  borderBottom: last ? "none" : `1px solid ${T.line}`,
});

/** A table's column heads. Same widths as its rows, by the caller passing the
 *  same numbers -- a grid template would be tidier and would also stop the
 *  table scrolling horizontally as one unit. */
export function Cols({ children }: { children: ReactNode }) {
  return (
    <div style={{
      display: "flex", padding: "8px 12px", borderBottom: `1px solid ${T.line}`,
      background: T.raised, font: `500 9.5px ${MONO}`, letterSpacing: ".1em",
      color: T.faint, textTransform: "uppercase",
    }}>{children}</div>
  );
}

/* --------------------------------------------------------------- atoms -- */

export const Dot = ({ colour, pulse }: { colour: string; pulse?: boolean }) => (
  <span style={{
    display: "inline-block", width: 7, height: 7, borderRadius: "50%",
    background: colour, flex: "0 0 auto",
    animation: pulse ? "apulse 1.9s infinite" : undefined,
  }} />
);

export function Tag({ children, colour = T.dim }: { children: ReactNode; colour?: string }) {
  return (
    <span style={{
      fontSize: 10, padding: "2px 7px", borderRadius: 2, whiteSpace: "nowrap",
      fontFamily: MONO, letterSpacing: ".06em", textTransform: "uppercase",
      background: "color-mix(in srgb, currentColor 12%, transparent)",
      color: colour, border: `1px solid color-mix(in srgb, ${colour} 34%, transparent)`,
    }}>{children}</span>
  );
}

/**
 * A plate, treated the same everywhere.
 *
 * `null` is the system saying it could not read one, and it gets its own
 * treatment rather than a blank: on real footage most tracked vehicles never
 * yield a plate, and an empty cell reads as a bug rather than the ordinary
 * state it is. Dashed and hatched, so the eye sorts read from unread before it
 * reads any characters.
 */
export function Plate({ text, size = 12.5 }: { text: string | null; size?: number }) {
  if (!text) {
    return (
      <span style={{
        display: "inline-flex", alignItems: "center", padding: "3px 7px",
        border: `1px dashed ${T.line2}`, borderRadius: 2, whiteSpace: "nowrap",
        background: "repeating-linear-gradient(135deg,transparent 0 3px,var(--sunk) 3px 6px)",
        font: `500 ${size - 1.5}px/1 ${MONO}`, letterSpacing: ".05em", color: T.faint,
      }}>NO PLATE</span>
    );
  }
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", padding: "3px 7px",
      border: `1px solid ${T.line2}`, background: T.raised, borderRadius: 2,
      font: `600 ${size}px/1 ${MONO}`, letterSpacing: ".07em", color: T.text,
      whiteSpace: "nowrap",
    }}>{text}</span>
  );
}

export const statusColour = (s: string) =>
  s === "online" || s === "live" || s === "done" ? T.ok
  : s === "degraded" || s === "waiting" || s === "running" || s === "pending" ? T.warn
  : s === "error" ? T.bad
  : s === "stale" ? T.warn
  : T.line2;

export const severityColour = (s: string) =>
  s === "critical" ? T.bad : s === "warn" ? T.warn : T.info;

/** Layer 1 and layers 2-3 must LOOK different. Which layer confirmed a match is
 *  the whole story of Module C, and a uniform badge hides it. */
export const methodColour = (m: string) =>
  m.startsWith("plate") ? T.accent : m.startsWith("reid") ? T.reid : T.ok;

/** An alert's severity, as a bordered chip on its own tinted ground. Four
 *  letters: the word is the least interesting part of an alert row. */
export function Sev({ severity }: { severity: string }) {
  const colour = severityColour(severity);
  const bg = severity === "critical" ? "var(--critBg)"
           : severity === "warn" ? "var(--warnBg)" : "var(--infoBg)";
  return (
    <span style={{
      padding: "3px 6px", borderRadius: 2, background: bg, color: colour,
      border: `1px solid ${colour}`, font: `600 8.5px/1 ${MONO}`,
      letterSpacing: ".08em", flex: "0 0 auto", marginTop: 2,
    }}>{severity.slice(0, 4).toUpperCase()}</span>
  );
}

/* ----------------------------------------------------------- module C --- */

/** Module C reports which layers agreed as a `+`-joined string. Three letters
 *  is the compact form the design reads by: P plate, R Re-ID, T travel time. */
const LETTERS: [string, string][] = [
  ["plate", "P"], ["reid", "R"], ["spatial_temporal", "T"],
];

export const layerCount = (method: string) =>
  LETTERS.filter(([k]) => method.split("+").includes(k)).length;

/**
 * The 2-of-3 rule, as one glyph.
 *
 * Module C confirms a match only when two of three independent layers agree,
 * and which two is the most informative fact the interface carries. Rendered
 * as a bare string ("plate+reid+spatial_temporal") it is unreadable at a
 * glance and unreadable at all in a dense table. Three fixed slots in a fixed
 * order, each either filled or dashed-empty: the shape alone says how well
 * corroborated a hop is, before a single character is read.
 *
 * The rule underneath is solid at 3 of 3 and dotted below, which is the same
 * fact a second time -- deliberately, because it survives being glanced at.
 */
export function Pips({ method, rule = true }: { method: string; rule?: boolean }) {
  const on = new Set(method.split("+"));
  const n = layerCount(method);
  return (
    <span title={method} style={{ display: "inline-flex", flexDirection: "column", gap: 3 }}>
      <span style={{ display: "flex", gap: 3, alignItems: "center" }}>
        {LETTERS.map(([key, letter]) => {
          const lit = on.has(key);
          return (
            <span key={key} style={{
              width: 16, height: 16, display: "inline-grid", placeItems: "center",
              borderRadius: 2, font: `600 9.5px/1 ${MONO}`, letterSpacing: ".02em",
              background: lit ? T.text : "transparent",
              color: lit ? T.panel : T.faint,
              border: `1px ${lit ? "solid" : "dashed"} ${lit ? T.text : T.line2}`,
            }}>{letter}</span>
          );
        })}
        <span style={{ marginLeft: 4, font: `500 9.5px/1 ${MONO}`, letterSpacing: ".04em",
                       color: n === 3 ? T.text : T.faint }}>{n}/3</span>
      </span>
      {rule && <span style={{
        height: 0, width: 62,
        borderTop: n === 3 ? `2px solid ${T.text}` : `2px dotted ${T.line2}`,
      }} />}
    </span>
  );
}

/**
 * A 0-1 value as a number and a bar.
 *
 * Both, always. The number is what an operator writes in a report; the bar is
 * what they compare down a column without reading. A null value is neither --
 * it prints the reason instead, because "0.00" and "never measured" are
 * opposite facts that must not share a shape.
 */
export function Conf({ value, width = 52, colour, missing = "NOT READ" }: {
  value: number | null; width?: number | string; colour?: string; missing?: string;
}) {
  if (value === null || value === undefined) {
    return (
      <span style={{
        font: `500 10.5px ${MONO}`, letterSpacing: ".06em", color: T.faint,
        padding: "2px 6px", border: `1px dashed ${T.line2}`, borderRadius: 2,
        whiteSpace: "nowrap",
      }}>{missing}</span>
    );
  }
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span className="tnum" style={{ font: `500 11.5px ${MONO}` }}>{value.toFixed(2)}</span>
      <Bar value={value} width={width} colour={colour} />
    </span>
  );
}

export function Bar({ value, width = 52, colour }:
                    { value: number; width?: number | string; colour?: string }) {
  return (
    <span style={{ width, height: 4, background: T.sunk, borderRadius: 1,
                   position: "relative", overflow: "hidden", flex: "0 0 auto",
                   display: "inline-block" }}>
      <span style={{
        position: "absolute", left: 0, top: 0, bottom: 0,
        width: `${Math.max(0, Math.min(1, value)) * 100}%`,
        background: colour ?? T.text, borderRadius: 1,
        transition: "width 1.05s cubic-bezier(.2,.75,.2,1)",
      }} />
    </span>
  );
}

/** A headline number with its label above it. The header uses a compact
 *  variant; the search summary uses this one. */
export function Figure({ label, value, unit }:
                       { label: string; value: ReactNode; unit?: string }) {
  return (
    <div>
      <div style={{ ...LABEL, letterSpacing: ".12em", fontSize: 9.5 }}>{label}</div>
      <div className="tnum" style={{ marginTop: 4, font: `500 20px ${MONO}` }}>
        {value}
        {unit && <span style={{ font: `400 12px ${MONO}`, color: T.faint }}> {unit}</span>}
      </div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p style={{ color: T.faint, font: `400 12px/1.7 ${SANS}`, margin: 0 }}>{children}</p>;
}

/** Seconds since an ISO timestamp, as something a human reads at a glance. */
export function ago(iso: string): string {
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

/** Wall-clock, to the second. Timestamps in this system carry milliseconds and
 *  an operator reads none of them. */
export const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString("en-GB", { hour12: false });

/** A travel time, in the units it is actually discussed in. */
export function hopTime(s: number): string {
  return s >= 60 ? `${Math.floor(s / 60)} m ${String(s % 60).padStart(2, "0")} s` : `${s} s`;
}
