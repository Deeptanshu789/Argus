"use client";
/**
 * Analytics: volume, speed and congestion over time, plus the vehicle mix.
 *
 * Every number comes from /api/analytics, which runs the same `bucketize()`
 * the worker uses. One implementation, so a chart cannot disagree with an alert
 * about how many vehicles passed.
 */
import { useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { getAnalytics, getCameras } from "@/lib/api";
import { Empty, Panel, T } from "@/components/ui";
import { useLive, usePoll } from "@/components/useLive";

const TYPE_COLOURS: Record<string, string> = {
  car: "#6cb6ff", truck: "#d29922", bus: "#3fb950",
  motorcycle: "#d2a8ff", auto: "#f85149",
};

const axis = { stroke: T.faint, fontSize: 11 };
const tip = {
  contentStyle: { background: T.raised, border: `1px solid ${T.line}`,
                  borderRadius: 6, fontSize: 12 },
  labelStyle: { color: T.dim },
};

const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function AnalyticsView() {
  const [camera, setCamera] = useState<string>("");
  const cameras = usePoll(getCameras, 30_000);
  const data = usePoll(() => getAnalytics(camera || undefined), 10_000);
  const live = useLive();

  const series = (data?.series ?? []).map((b) => ({
    ...b, label: clock(b.ts),
    // Recharts draws a gap for null, which is correct: a bucket with no
    // measurable speed is missing data, not zero. Plotting zero would drag
    // the line to the floor and invent a traffic jam.
    avg_speed_kmh: b.avg_speed_kmh,
  }));

  const mix: { type: string; count: number }[] = Object.entries(
    (data?.series ?? []).reduce<Record<string, number>>((acc, b) => {
      for (const [k, v] of Object.entries(b.by_type)) acc[k] = (acc[k] ?? 0) + v;
      return acc;
    }, {}),
  ).map(([type, count]) => ({ type, count })).sort((a, b) => b.count - a.count);

  const congestionNow = Object.entries(live.perCamera)
    .map(([id, v]) => ({ id, ...v }))
    .sort((a, b) => b.congestion_score - a.congestion_score);

  const empty = !series.length;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <Panel title="Scope" right={
        <span style={{ fontSize: 12, color: T.dim }}>
          {data?.totals.vehicle_count ?? 0} vehicles ·{" "}
          {data?.totals.avg_speed_kmh === null || data?.totals.avg_speed_kmh === undefined
            ? "speed not measurable"
            : `${data.totals.avg_speed_kmh} km/h average`}
        </span>
      }>
        <div style={{ display: "flex", gap: ".4rem", flexWrap: "wrap" }}>
          {[{ id: "", name: "All cameras" }, ...(cameras ?? [])].map((c) => (
            <button key={c.id} onClick={() => setCamera(c.id)} style={{
              padding: ".3rem .75rem", borderRadius: 6, fontSize: 12, cursor: "pointer",
              background: camera === c.id ? T.accent : "transparent",
              color: camera === c.id ? "#04121f" : T.dim,
              border: `1px solid ${camera === c.id ? T.accent : T.line}`,
            }}>{c.id || c.name}</button>
          ))}
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
        <div style={{ display: "grid", gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))" }}>
          <Panel title="Vehicle volume">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={series}>
                <CartesianGrid stroke={T.line} vertical={false} />
                <XAxis dataKey="label" {...axis} />
                <YAxis {...axis} allowDecimals={false} />
                <Tooltip {...tip} />
                <Area type="monotone" dataKey="vehicle_count" name="vehicles"
                      stroke={T.accent} fill={`${T.accent}33`} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Panel>

          <Panel title="Average speed">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={series}>
                <CartesianGrid stroke={T.line} vertical={false} />
                <XAxis dataKey="label" {...axis} />
                <YAxis {...axis} unit=" km/h" />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="avg_speed_kmh" name="km/h"
                      stroke={T.ok} strokeWidth={2} dot={false} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 11, color: T.faint, margin: ".5rem 0 0" }}>
              Gaps are buckets with no measurable speed, not zeros. Speed needs
              per-camera calibration — set <code>metres_per_pixel</code> or the
              scale is indicative only.
            </p>
          </Panel>

          <Panel title="Congestion">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={series}>
                <CartesianGrid stroke={T.line} vertical={false} />
                <XAxis dataKey="label" {...axis} />
                <YAxis {...axis} domain={[0, 100]} />
                <Tooltip {...tip} />
                <Area type="monotone" dataKey="congestion_score" name="score"
                      stroke={T.warn} fill={`${T.warn}33`} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Panel>

          <Panel title="Vehicle mix">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={mix}>
                <CartesianGrid stroke={T.line} vertical={false} />
                <XAxis dataKey="type" {...axis} />
                <YAxis {...axis} allowDecimals={false} />
                <Tooltip {...tip} />
                <Legend wrapperStyle={{ fontSize: 11, color: T.dim }} />
                <Bar dataKey="count" name="vehicles">
                  {mix.map((m) => (
                    <Cell key={m.type} fill={TYPE_COLOURS[m.type] ?? T.dim} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        </div>
      )}

      {congestionNow.length > 0 && (
        <Panel title="Live congestion by camera">
          <div style={{ display: "grid", gap: ".5rem" }}>
            {congestionNow.map((c) => (
              <div key={c.id} style={{ display: "flex", alignItems: "center",
                                       gap: ".75rem", fontSize: 12 }}>
                <span style={{ width: 52, color: T.dim }}>{c.id}</span>
                <div style={{ flex: 1, height: 8, background: T.raised,
                              borderRadius: 4, overflow: "hidden" }}>
                  <div style={{
                    width: `${c.congestion_score}%`, height: "100%",
                    background: c.congestion_score > 66 ? T.bad
                              : c.congestion_score > 33 ? T.warn : T.ok,
                  }} />
                </div>
                <span style={{ width: 88, textAlign: "right", color: T.dim }}>
                  {c.congestion_score.toFixed(0)} · {c.vehicle_count} veh
                </span>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
