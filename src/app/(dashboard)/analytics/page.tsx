"use client";
/**
 * Analytics: volume against speed loss, congestion by camera, the vehicle mix,
 * throughput, and where match confidence comes from.
 *
 * Every number comes from /api/analytics, which runs the same `bucketize()`
 * the worker uses. One implementation, so a chart cannot disagree with an alert
 * about how many vehicles passed.
 *
 * The charts are hand-drawn SVG rather than a chart library. Two of them --
 * the dot matrix and the confidence scatter -- have no library equivalent, and
 * once two are hand-drawn a second rendering stack for the other two is pure
 * weight. Every scale is derived from the data in view, so a demo with forty
 * vehicles and a city with forty thousand both fill the frame.
 */
import { useEffect, useRef, useState } from "react";
import { getAnalytics, getCameras, getTrajectories } from "@/lib/api";
import {
  Empty, Head, LABEL, META, MONO, Panel, RAMP, SANS, T, rampFor,
} from "@/components/ui";
import { useLive, usePoll } from "@/components/useLive";
import type { AnalyticsBucket } from "@/contract";

const clockOf = (iso: string) =>
  new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

/** An absolutely positioned label over an SVG. Text inside the SVG would scale
 *  with the viewBox and stop being 9.5px at any other width. */
function tick(leftPct: number, topPct: number, anchor: "start" | "mid" | "end",
              extra?: React.CSSProperties): React.CSSProperties {
  return {
    position: "absolute", left: `${leftPct}%`, top: `${topPct}%`,
    transform: anchor === "end" ? "translate(-100%,-50%)"
             : anchor === "start" ? "translate(0,-50%)" : "translate(-50%,-50%)",
    whiteSpace: "nowrap", pointerEvents: "none",
    font: `400 9.5px/1 ${MONO}`, color: T.faint,
    fontVariantNumeric: "tabular-nums", ...extra,
  };
}

/* ------------------------------------------------------- the dot matrix -- */

/**
 * Volume above the line, speed lost below it, one dot per unit of each.
 *
 * A dot matrix rather than two stacked line charts because the question an
 * operator actually asks is "are these the same minutes?" -- and a column that
 * grows in both directions at once answers it without reading either axis. The
 * unit is derived from the window so the tallest column is around a dozen dots
 * whether this is a demo clip or a city at rush hour; it is printed, because a
 * dot with an unstated unit is decoration.
 */
function DotMatrix({ series }: { series: AnalyticsBucket[] }) {
  const W = 1180, H = 310, X0 = 54, X1 = 1168, MID = 152;
  const n = series.length;
  const BW = (X1 - X0) / Math.max(1, n);

  const counts = series.map((b) => b.vehicle_count);
  const speeds = series.map((b) => b.avg_speed_kmh).filter((s): s is number => s !== null);
  const free = speeds.length ? Math.max(...speeds) : 0;
  const maxCount = Math.max(1, ...counts);
  const maxLoss = Math.max(1, ...speeds.map((s) => free - s));
  const vPer = Math.max(1, Math.round(maxCount / 12));
  const sPer = Math.max(0.5, Math.round((maxLoss / 8) * 2) / 2);

  const up: { x: number; y: number; fill: string }[] = [];
  const down: { x: number; y: number; fill: string }[] = [];
  const blank: number[] = [];

  series.forEach((b, i) => {
    const x = X0 + i * BW + BW / 2;
    const fill = rampFor((b.congestion_score ?? 0) / 100);
    for (let j = 0; j < Math.max(1, Math.round(b.vehicle_count / vPer)); j++) {
      if (144 - j * 8.6 < 16) break;
      up.push({ x, y: 144 - j * 8.6, fill });
    }
    if (b.avg_speed_kmh === null) { blank.push(x); return; }
    for (let j = 0; j < Math.round((free - b.avg_speed_kmh) / sPer); j++) {
      if (160 + j * 8.6 > 262) break;
      down.push({ x, y: 160 + j * 8.6, fill });
    }
  });

  // The worst bucket names itself. A chart that makes the reader hunt for the
  // peak has not finished its job.
  let worst = 0;
  series.forEach((b, i) => {
    if ((b.congestion_score ?? 0) > (series[worst]!.congestion_score ?? 0)) worst = i;
  });
  const wb = series[worst];
  const wx = X0 + worst * BW + BW / 2;
  const calloutLeft = Math.max(X0, Math.min(wx - 206, X1 - 206));

  const every = Math.max(1, Math.round(n / 12));

  return (
    <div style={{ position: "relative", minWidth: 900 }}>
      {[[56, `${maxCount} veh`], [100, `${Math.round(maxCount / 2)}`], [156, "0"],
        [208, `${sPer * 4} km/h`], [252, `${sPer * 9} lost`]].map(([y, label]) => (
        <div key={label} style={tick(3.9, (y as number) / H * 100, "end")}>{label}</div>
      ))}
      {series.map((b, i) => i % every === 0 ? (
        <div key={b.ts} style={tick((X0 + i * BW + BW / 2) / W * 100, 96.1, "mid")}>
          {clockOf(b.ts)}
        </div>
      ) : null)}
      {wb && (
        <>
          <div style={tick((calloutLeft + 12) / W * 100, 70 / H * 100, "start",
                           { font: `500 11px/1 ${MONO}`, color: T.panel })}>
            {clockOf(wb.ts)} · {wb.vehicle_count} vehicles
          </div>
          <div style={tick((calloutLeft + 12) / W * 100, 87 / H * 100, "start",
                           { font: `400 10px/1 ${SANS}`, color: T.panel, opacity: 0.7 })}>
            {wb.avg_speed_kmh === null ? "speed not measurable"
              : `mean ${wb.avg_speed_kmh.toFixed(1)} km/h`} · congestion{" "}
            {((wb.congestion_score ?? 0) / 100).toFixed(2)}
          </div>
        </>
      )}

      <svg viewBox={`0 0 ${W} ${H}`} style={{ display: "block", width: "100%", height: "auto" }}>
        {[52, 96, 204, 248].map((y) => (
          <line key={y} x1={X0} y1={y} x2={X1} y2={y} stroke={T.line} strokeWidth={1} />
        ))}
        {up.map((d, i) => <circle key={`u${i}`} cx={d.x} cy={d.y} r={3.4} fill={d.fill} />)}
        {down.map((d, i) => <circle key={`d${i}`} cx={d.x} cy={d.y} r={3.4} fill={d.fill} />)}
        <line x1={X0} y1={MID} x2={X1} y2={MID} stroke={T.text} strokeWidth={1} />
        {/* A bucket with no measurable speed is missing data, not zero. An open
            dashed ring says "we did not measure this" where a filled dot at the
            floor would invent a traffic jam. */}
        {blank.map((x, i) => (
          <circle key={`n${i}`} cx={x} cy={272} r={3.4} fill="none"
                  stroke={T.faint} strokeWidth={1} strokeDasharray="1.6 1.6" />
        ))}
        <text x={46} y={275} textAnchor="end" fill={T.faint}
              style={{ font: `400 9px ${MONO}` }}>no speed</text>
        <text x={14} y={96} fill={T.faint} transform="rotate(-90 14 96)"
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".12em" }}>VEHICLES</text>
        <text x={14} y={232} fill={T.faint} transform="rotate(-90 14 232)"
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".12em" }}>SPEED LOST</text>
        {wb && (
          <>
            <rect x={calloutLeft} y={52} width={200} height={46} fill={T.text} rx={2} />
            <line x1={wx} y1={36} x2={wx} y2={136} stroke={T.text} strokeWidth={1} />
          </>
        )}
      </svg>
    </div>
  );
}

/* ------------------------------------------------- throughput and speed -- */

function Throughput({ series }: { series: AnalyticsBucket[] }) {
  const W = 560, H = 250, X0 = 42, X1 = 548, TOP = 38, BOT = 206;
  const n = series.length;
  const LX = (i: number) => X0 + (n < 2 ? 0 : i * ((X1 - X0 - 12) / (n - 1)));
  const counts = series.map((b) => b.vehicle_count);
  const speeds = series.map((b) => b.avg_speed_kmh).filter((s): s is number => s !== null);
  const cMax = Math.max(1, ...counts), cMin = Math.min(...counts, 0);
  const sMax = Math.max(1, ...speeds), sMin = Math.min(...speeds, 0);
  const volY = (v: number) => BOT - ((v - cMin) / Math.max(1, cMax - cMin)) * (BOT - TOP);
  const spY = (v: number) => BOT - ((v - sMin) / Math.max(1, sMax - sMin)) * (BOT - TOP);

  const volPath = series.map((b, i) =>
    `${i ? "L" : "M"} ${LX(i).toFixed(1)} ${volY(b.vehicle_count).toFixed(1)}`).join(" ");
  const volArea = `${volPath} L ${LX(n - 1).toFixed(1)} ${BOT} L ${X0} ${BOT} Z`;

  const segs: { x1: number; y1: number; x2: number; y2: number; gap: boolean }[] = [];
  for (let i = 1; i < n; i++) {
    const a = series[i - 1]!.avg_speed_kmh, b = series[i]!.avg_speed_kmh;
    segs.push({
      x1: LX(i - 1), y1: a === null ? BOT : spY(a),
      x2: LX(i), y2: b === null ? BOT : spY(b),
      gap: a === null || b === null,
    });
  }
  const every = Math.max(1, Math.round(n / 6));

  return (
    <div style={{ position: "relative" }}>
      {[0, 1, 2, 3, 4].map((k) => {
        const y = TOP + k * ((BOT - TOP) / 4);
        return (
          <div key={`l${k}`} style={tick(36 / W * 100, y / H * 100, "end", { color: T.accent })}>
            {Math.round(cMax - k * ((cMax - cMin) / 4))}
          </div>
        );
      })}
      {[0, 1, 2, 3, 4].map((k) => {
        const y = TOP + k * ((BOT - TOP) / 4);
        return (
          <div key={`r${k}`} style={tick(552 / W * 100, y / H * 100, "start", { color: T.dim })}>
            {Math.round(sMax - k * ((sMax - sMin) / 4))}
          </div>
        );
      })}
      {series.map((b, i) => i % every === 0 ? (
        <div key={b.ts} style={tick(LX(i) / W * 100, 236 / H * 100, "mid")}>{clockOf(b.ts)}</div>
      ) : null)}

      <svg viewBox={`0 0 ${W} ${H}`} style={{ display: "block", width: "100%", height: "auto" }}>
        {[0, 1, 2, 3, 4].map((k) => {
          const y = TOP + k * ((BOT - TOP) / 4);
          return <line key={k} x1={X0} y1={y} x2={X1} y2={y} stroke={T.line} strokeWidth={1} />;
        })}
        <path d={volArea} fill={T.accentSoft} />
        <path d={volPath} fill="none" stroke={T.accent} strokeWidth={1.6} />
        {segs.map((s, i) => (
          <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                stroke={s.gap ? T.faint : T.text}
                strokeWidth={s.gap ? 1 : 2.2}
                strokeDasharray={s.gap ? "3 3" : undefined} strokeLinecap="round" />
        ))}
        <text x={X0} y={14} fill={T.accent}
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".1em" }}>VEHICLES / BUCKET</text>
        <text x={X1} y={14} textAnchor="end" fill={T.text}
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".1em" }}>KM/H</text>
      </svg>
    </div>
  );
}

/* ---------------------------------------------------- confidence scatter -- */

type Dot = { tt: number; conf: number; method: string };

/**
 * Every hop plotted by travel time against confidence, marked by the layer
 * that decided it.
 *
 * This is the one chart that argues the project's thesis. Plate reads cluster
 * high and tight; Re-ID sits in a band around its 0.75 cutoff; travel-time-only
 * matches sit lowest and are drawn hollow, because "nothing contradicted this"
 * is a weaker claim than "two things agreed".
 */
function Confidence({ dots }: { dots: Dot[] }) {
  const W = 560, H = 250, X0 = 44, X1 = 548, TOP = 32, BOT = 214;
  const ttMax = Math.max(120, ...dots.map((d) => d.tt));
  const cLo = Math.min(0.6, ...dots.map((d) => d.conf)), cHi = 1.0;
  const X = (tt: number) => X0 + (tt / ttMax) * (X1 - X0 - 4);
  const Y = (c: number) => BOT - ((c - cLo) / (cHi - cLo)) * (BOT - TOP);

  return (
    <div style={{ position: "relative" }}>
      {[0, 1, 2, 3, 4].map((k) => {
        const y = TOP + k * ((BOT - TOP) / 4);
        return (
          <div key={k} style={tick(38 / W * 100, y / H * 100, "end")}>
            {(cHi - k * ((cHi - cLo) / 4)).toFixed(2)}
          </div>
        );
      })}
      {[0, 1, 2, 3, 4, 5, 6].map((k) => {
        const s = Math.round((ttMax / 6) * k);
        return <div key={k} style={tick(X(s) / W * 100, 236 / H * 100, "mid")}>{s}</div>;
      })}

      <svg viewBox={`0 0 ${W} ${H}`} style={{ display: "block", width: "100%", height: "auto" }}>
        {[0, 1, 2, 3, 4].map((k) => {
          const y = TOP + k * ((BOT - TOP) / 4);
          return <line key={k} x1={X0} y1={y} x2={X1} y2={y} stroke={T.line} strokeWidth={1} />;
        })}
        {cLo < 0.75 && (
          <>
            <line x1={X0} y1={Y(0.75)} x2={X1} y2={Y(0.75)} stroke={T.faint}
                  strokeWidth={1} strokeDasharray="4 3" />
            <text x={X1} y={Y(0.75) - 5} textAnchor="end" fill={T.faint}
                  style={{ font: `400 9px ${MONO}` }}>RE-ID CUTOFF 0.75</text>
          </>
        )}
        {dots.map((d, i) => {
          const st = d.method === "spatial_temporal";
          return (
            <circle key={i} cx={X(d.tt)} cy={Y(d.conf)} r={4.2}
                    fill={st ? "none" : d.method.startsWith("plate") ? T.text : T.accent}
                    stroke={st ? T.faint : "none"} strokeWidth={1}
                    opacity={st ? 0.8 : 0.6} />
          );
        })}
        <text x={X0} y={14} fill={T.faint}
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".1em" }}>CONFIDENCE</text>
        <text x={296} y={249} textAnchor="middle" fill={T.faint}
              style={{ font: `500 9px ${MONO}`, letterSpacing: ".1em" }}>
          TRAVEL TIME BETWEEN CAMERAS (SECONDS)
        </text>
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------- the view -- */

/** One congestion sample per camera, kept in the page. */
type Sample = Record<string, number>;
const HEAT_COLS = 36;

export default function AnalyticsView() {
  const [camera, setCamera] = useState<string>("");
  const cameras = usePoll(getCameras, 30_000);
  const data = usePoll(() => getAnalytics(camera || undefined), 10_000);
  const trajectories = usePoll(() => getTrajectories({ limit: 120 }), 20_000);
  const live = useLive();

  // ponytail: the congestion grid is built from live rollups as the page
  // watches, not from history -- a per-camera history query is one /api/analytics
  // call per camera and that endpoint is the expensive one. Add a grouped
  // server-side query if the grid needs to survive a reload.
  const [heat, setHeat] = useState<Sample[]>([]);
  const lastRollup = useRef<string>("");
  useEffect(() => {
    const keys = Object.keys(live.perCamera);
    if (!keys.length) return;
    const stamp = keys.map((k) => `${k}:${live.perCamera[k]!.congestion_score}`).join("|");
    if (stamp === lastRollup.current) return;
    lastRollup.current = stamp;
    const s: Sample = {};
    for (const k of keys) s[k] = live.perCamera[k]!.congestion_score;
    setHeat((h) => [...h, s].slice(-HEAT_COLS));
  }, [live.perCamera]);

  const series = data?.series ?? [];
  const empty = !series.length;

  const mix = Object.entries(
    series.reduce<Record<string, number>>((acc, b) => {
      for (const [k, v] of Object.entries(b.by_type)) acc[k] = (acc[k] ?? 0) + v;
      return acc;
    }, {}),
  ).map(([type, count]) => ({ type, count })).sort((a, b) => b.count - a.count);
  const mixMax = Math.max(1, ...mix.map((m) => m.count));

  const dots: Dot[] = (trajectories ?? []).flatMap((t) =>
    t.hops.map((h) => ({ tt: h.travel_time_s, conf: h.confidence, method: h.method })));

  const heatRows = Array.from(
    new Set(heat.flatMap((s) => Object.keys(s)))).sort();

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Panel flush title="Scope"
             right={<span className="tnum" style={META}>
               {data?.totals.vehicle_count ?? 0} vehicles ·{" "}
               {data?.totals.avg_speed_kmh == null ? "speed not measurable"
                 : `${data.totals.avg_speed_kmh} km/h mean`}
             </span>}>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap", padding: "10px 12px" }}>
          {[{ id: "", name: "All cameras" }, ...(cameras ?? [])].map((c) => {
            const on = camera === c.id;
            return (
              <button key={c.id} onClick={() => setCamera(c.id)} style={{
                padding: "4px 10px", borderRadius: 2, font: `500 10px ${MONO}`,
                letterSpacing: ".08em",
                background: on ? T.text : "transparent",
                color: on ? T.panel : T.dim,
                border: `1px solid ${on ? T.text : T.line2}`,
              }}>{c.id || "ALL"}</button>
            );
          })}
        </div>
      </Panel>

      {empty ? (
        <Panel title="No data yet">
          <Empty>
            Nothing has been ingested in this window. Start the pipeline:<br />
            <code>ARGUS_CAMERAS=&apos;CAM1=demo,CAM3=demo&apos; npm run worker</code>
          </Empty>
        </Panel>
      ) : (
        <>
          <div style={{ border: `1px solid ${T.line}`, background: T.panel, borderRadius: 3,
                        boxShadow: T.shadow }}>
            <div style={{ padding: "13px 16px 11px", borderBottom: `1px solid ${T.line}` }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12,
                            flexWrap: "wrap" }}>
                <h2 style={{ margin: 0, font: `600 17px/1.2 ${SANS}`,
                             letterSpacing: "-0.01em" }}>Volume against speed loss</h2>
                <div style={{ font: `400 12px ${SANS}`, color: T.accent }}>
                  {series.length} five-minute buckets ·{" "}
                  {camera || `${(cameras ?? []).length} cameras`}
                </div>
              </div>
              <p style={{ margin: "7px 0 0", maxWidth: 760, font: `400 12.5px/1.6 ${SANS}`,
                          color: T.dim, textWrap: "pretty" }}>
                Dots above the line are vehicles counted; dots below are speed lost against
                the fastest bucket in this window. Both are shaded by congestion, so a
                column that grows tall in both directions is the shape of a jam forming.
              </p>
              <div style={{ marginTop: 11, display: "flex", alignItems: "center", gap: 9 }}>
                <div style={{ font: `400 9.5px ${MONO}`, color: T.faint,
                              letterSpacing: ".06em" }}>CONGESTION 0.0</div>
                <div style={{ display: "flex", height: 9, width: 150 }}>
                  {RAMP.map((c) => <span key={c} style={{ flex: 1, background: c }} />)}
                </div>
                <div style={{ font: `400 9.5px ${MONO}`, color: T.faint,
                              letterSpacing: ".06em" }}>1.0</div>
              </div>
            </div>
            <div style={{ padding: "8px 10px 4px", overflowX: "auto" }}>
              <DotMatrix series={series} />
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.35fr 1fr", gap: 14,
                        alignItems: "start" }}>
            <Panel flush style={{ minWidth: 0 }} title={undefined}>
              <Head title="Congestion by camera"
                    sub={heat.length
                      ? "One column per rollup since this page opened. Same scale as above."
                      : "Fills in as the worker publishes rollups. Same scale as above."} />
              <div style={{ padding: "12px 14px" }}>
                {heatRows.length ? (
                  <div style={{ display: "grid", gap: 5 }}>
                    {heatRows.map((id) => (
                      <div key={id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 58, flex: "0 0 58px", font: `400 10px ${MONO}`,
                                       color: T.dim, overflow: "hidden",
                                       textOverflow: "ellipsis" }}>{id}</span>
                        <span style={{ flex: 1, display: "flex", gap: 2 }}>
                          {Array.from({ length: HEAT_COLS }, (_, i) => {
                            const s = heat[heat.length - HEAT_COLS + i];
                            const v = s?.[id];
                            return (
                              <span key={i} title={v === undefined ? "no sample"
                                                    : `${(v / 100).toFixed(2)}`}
                                    style={{
                                      flex: 1, height: 15, borderRadius: 1,
                                      background: v === undefined ? "transparent"
                                                : rampFor(v / 100),
                                      border: v === undefined
                                        ? `0.8px dashed ${T.line2}` : "none",
                                    }} />
                            );
                          })}
                        </span>
                      </div>
                    ))}
                    <div style={{ ...META, marginTop: 4 }}>
                      dashed = no rollup seen yet, not zero traffic
                    </div>
                  </div>
                ) : (
                  <Empty>
                    Nothing published yet. The worker emits one rollup per interval;
                    the grid grows a column each time.
                  </Empty>
                )}
              </div>
            </Panel>

            <Panel flush style={{ minWidth: 0 }} title={undefined}>
              <Head title="Vehicle mix"
                    sub="Tracks in this window by class, from the same buckets as above." />
              <div style={{ padding: "12px 14px 16px", display: "flex",
                            flexDirection: "column", gap: 11 }}>
                {mix.length ? mix.map((m) => (
                  <div key={m.type}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8,
                                  marginBottom: 5 }}>
                      <span style={{ width: 82, flex: "0 0 82px", font: `400 12px ${SANS}` }}>
                        {m.type}
                      </span>
                      <span style={{ flex: 1, height: 1, background: T.line }} />
                      <span className="tnum" style={{ font: `500 12px ${MONO}` }}>
                        {m.count.toLocaleString()}
                      </span>
                      <span className="tnum" style={{ width: 52, textAlign: "right",
                                                      ...META }}>
                        {((m.count / mix.reduce((n, x) => n + x.count, 0)) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div style={{ height: 14, background: T.sunk, borderRadius: 1,
                                  overflow: "hidden", position: "relative" }}>
                      <span style={{ position: "absolute", inset: "0 auto 0 0",
                                     width: `${(m.count / mixMax) * 100}%`,
                                     background: T.accent }} />
                    </div>
                  </div>
                )) : <Empty>No vehicle types recorded in this window.</Empty>}
              </div>
            </Panel>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14,
                        alignItems: "start" }}>
            <Panel flush style={{ minWidth: 0 }} title={undefined}>
              <Head title="Throughput and mean speed"
                    sub="Volume filled, speed as the dark line. Dashed where no speed was measurable." />
              <div style={{ padding: "10px 12px 6px" }}>
                <Throughput series={series} />
              </div>
            </Panel>

            <Panel flush style={{ minWidth: 0 }} title={undefined}>
              <Head title="Where the confidence comes from"
                    sub="Every stitched hop, by travel time and match confidence. The marker is the deciding layer." />
              <div style={{ padding: "10px 12px 6px" }}>
                {dots.length ? <Confidence dots={dots} /> : (
                  <div style={{ padding: "12px 2px" }}>
                    <Empty>
                      No cross-camera hop yet. Each one Module C confirms puts a
                      point on this chart.
                    </Empty>
                  </div>
                )}
                <div style={{ display: "flex", gap: 16, padding: "2px 4px 10px",
                              flexWrap: "wrap" }}>
                  {([["plate text", T.text, false], ["OSNet re-id", T.accent, false],
                     ["spatio-temporal only", "transparent", true]] as const).map(
                    ([label, fill, hollow]) => (
                      <div key={label} style={{ display: "flex", alignItems: "center",
                                                gap: 6, ...META }}>
                        <span style={{ width: 9, height: 9, borderRadius: "50%",
                                       background: fill,
                                       border: hollow ? `1px solid ${T.faint}` : "none" }} />
                        {label}
                      </div>
                    ))}
                </div>
              </div>
            </Panel>
          </div>
        </>
      )}

      <div style={{ ...LABEL, letterSpacing: ".06em", textTransform: "none",
                    font: `400 11px/1.6 ${SANS}` }}>
        Speed needs a metres-per-pixel survey per camera. Without one it is an
        estimate from a default scale, not a measurement — which is why an
        unmeasurable bucket prints nothing rather than a zero.
      </div>
    </div>
  );
}
