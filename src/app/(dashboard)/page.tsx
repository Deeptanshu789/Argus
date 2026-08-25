"use client";
/**
 * Live view: camera grid, the detection feed, confirmed cross-camera matches,
 * and open alerts.
 *
 * The camera "feed" is a synthetic render of the live bounding boxes, not video.
 * Streaming four MJPEG feeds into a browser costs more than the whole inference
 * budget, and the boxes are what the operator actually reads. When a real
 * stream_url exists it goes behind the boxes; nothing else changes.
 */
import { useState } from "react";
import { getAlerts, getCameras, ackAlert } from "@/lib/api";
import { Dot, Empty, Panel, T, Tag, ago, methodColour, severityColour, statusColour }
  from "@/components/ui";
import { useLive, usePoll } from "@/components/useLive";
import type { Alert, Camera } from "@/contract";

/** Bounding boxes drawn at their true aspect. The sidecar reports pixels in the
 *  source frame, so the tile scales them rather than assuming a size. */
function CameraTile({ camera, boxes, count }: {
  camera: Camera;
  boxes: { bbox: [number, number, number, number]; vehicle_type: string; conf: number }[];
  count: number;
}) {
  // The source frame size is not in the contract. Take the extent of what we
  // have seen and never shrink below a sane default, so boxes stay in frame
  // rather than being clipped by a guessed resolution.
  const W = Math.max(960, ...boxes.map((b) => b.bbox[2]));
  const H = Math.max(540, ...boxes.map((b) => b.bbox[3]));

  return (
    <Panel style={{ padding: 0, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: ".5rem",
                    padding: ".6rem .8rem", borderBottom: `1px solid ${T.line}` }}>
        <Dot colour={statusColour(camera.status)} pulse={camera.status === "online"} />
        <strong style={{ fontSize: 13 }}>{camera.id}</strong>
        <span style={{ fontSize: 12, color: T.dim, overflow: "hidden",
                       textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {camera.name}
        </span>
        <span style={{ marginLeft: "auto", fontSize: 11, color: T.faint }}>
          {count} seen
        </span>
      </div>
      <div style={{ position: "relative", aspectRatio: `${W} / ${H}`,
                    background: "#0a0e13",
                    backgroundImage:
                      "repeating-linear-gradient(0deg,#0d131a 0 1px,transparent 1px 28px)," +
                      "repeating-linear-gradient(90deg,#0d131a 0 1px,transparent 1px 28px)" }}>
        {boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.bbox;
          return (
            <div key={i} style={{
              position: "absolute",
              left: `${(x1 / W) * 100}%`, top: `${(y1 / H) * 100}%`,
              width: `${((x2 - x1) / W) * 100}%`, height: `${((y2 - y1) / H) * 100}%`,
              border: `2px solid ${T.ok}`, borderRadius: 2,
              boxShadow: `0 0 12px ${T.ok}44`,
            }}>
              <span style={{
                position: "absolute", top: -17, left: -2, fontSize: 10,
                background: T.ok, color: "#04140a", padding: "1px 4px",
                borderRadius: 2, whiteSpace: "nowrap",
              }}>{b.vehicle_type} {b.conf.toFixed(2)}</span>
            </div>
          );
        })}
        {!boxes.length && (
          <span style={{ position: "absolute", inset: 0, display: "grid",
                         placeItems: "center", color: T.faint, fontSize: 12 }}>
            no vehicles in frame
          </span>
        )}
      </div>
    </Panel>
  );
}

export default function LiveView() {
  const live = useLive();
  const cameras = usePoll(getCameras, 10_000);
  const stored = usePoll(() => getAlerts(false), 15_000);
  const [acked, setAcked] = useState<Set<string>>(new Set());

  // Alerts arrive two ways: over the socket as they fire, and from REST for
  // anything raised before this page opened. Merge, newest first, no repeats.
  const alerts: Alert[] = [
    ...live.alerts,
    ...(stored ?? []).filter((a) => !live.alerts.some((l) => l.id === a.id)),
  ].filter((a) => !acked.has(a.id))
   .sort((a, b) => (a.ts < b.ts ? 1 : -1));

  const onAck = async (id: string) => {
    setAcked((s) => new Set(s).add(id));   // optimistic: the operator clicked it
    try { await ackAlert(id); } catch { setAcked((s) => {
      const n = new Set(s); n.delete(id); return n;   // put it back on failure
    }); }
  };

  // Only the freshest box per track, and only recent ones: a stale box left on
  // screen reads as a vehicle that is still there.
  const FRESH_MS = 3000;
  const byCamera: Record<string, typeof live.detections> = {};
  const seenTracks = new Set<string>();
  for (const d of live.detections) {
    if (Date.now() - d.at > FRESH_MS) continue;
    const key = `${d.camera_id}:${d.track_id}`;
    if (seenTracks.has(key)) continue;
    seenTracks.add(key);
    (byCamera[d.camera_id] ??= []).push(d);
  }

  return (
    <div style={{ display: "grid", gap: "1rem",
                  gridTemplateColumns: "minmax(0, 2.1fr) minmax(280px, 1fr)" }}>
      <div style={{ display: "grid", gap: "1rem",
                    gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        {(cameras ?? []).map((c) => (
          <CameraTile key={c.id} camera={c}
            boxes={byCamera[c.id] ?? []}
            count={live.counts[c.id] ?? 0} />
        ))}
        {!cameras?.length && (
          <Panel title="Cameras">
            <Empty>None configured. Run <code>npm run db:setup</code>.</Empty>
          </Panel>
        )}
      </div>

      <div style={{ display: "grid", gap: "1rem", alignContent: "start" }}>
        <Panel title={`Open alerts — ${alerts.length}`}>
          {alerts.length ? alerts.slice(0, 8).map((a) => (
            <div key={a.id} style={{ display: "flex", alignItems: "flex-start",
                                     gap: ".5rem", padding: ".4rem 0",
                                     borderBottom: `1px solid ${T.line}` }}>
              <Dot colour={severityColour(a.severity)} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13 }}>
                  {a.kind.replace("_", " ")}
                  {a.plate_text && <> · <strong>{a.plate_text}</strong></>}
                </div>
                <div style={{ fontSize: 11, color: T.dim }}>
                  {a.camera_id ?? "city"} · {a.detail} · {ago(a.ts)}
                </div>
              </div>
              <button onClick={() => void onAck(a.id)} style={{
                background: "transparent", border: `1px solid ${T.line}`,
                color: T.dim, borderRadius: 4, fontSize: 11,
                padding: "2px 7px", cursor: "pointer",
              }}>ack</button>
            </div>
          )) : <Empty>Nothing open.</Empty>}
        </Panel>

        <Panel title="Cross-camera matches — Module C">
          {live.matches.length ? live.matches.slice(0, 8).map((m, i) => (
            <div key={i} style={{ padding: ".4rem 0", borderBottom: `1px solid ${T.line}` }}>
              <div style={{ display: "flex", alignItems: "center", gap: ".5rem" }}>
                <strong style={{ fontSize: 13 }}>
                  {m.plate_text ?? "unidentified"}
                </strong>
                <Tag colour={methodColour(m.method)}>
                  {m.method} {m.confidence.toFixed(2)}
                </Tag>
              </div>
              <div style={{ fontSize: 11, color: T.dim }}>
                {m.from_camera} to {m.to_camera} in {m.travel_time_s}s
              </div>
            </div>
          )) : (
            <Empty>
              None yet. A match is confirmed when two of three layers agree —
              plate text, Re-ID similarity, or travel-time feasibility.
            </Empty>
          )}
        </Panel>

        <Panel title="Detection feed">
          <div style={{ maxHeight: 260, overflowY: "auto", fontSize: 12,
                        fontFamily: "ui-monospace, monospace" }}>
            {live.detections.length ? live.detections.slice(0, 30).map((d, i) => (
              <div key={i} style={{ display: "flex", gap: ".6rem", color: T.dim,
                                    padding: "1px 0" }}>
                <span style={{ color: T.accent, width: 46 }}>{d.camera_id}</span>
                <span style={{ width: 34 }}>T{d.track_id}</span>
                <span style={{ flex: 1 }}>{d.vehicle_type}</span>
                <span>{d.conf.toFixed(2)}</span>
              </div>
            )) : <Empty>Waiting. Start the worker: <code>npm run worker</code></Empty>}
          </div>
        </Panel>
      </div>
    </div>
  );
}
