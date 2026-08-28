"use client";
/**
 * Live view: the camera wall, confirmed cross-camera matches, the detection
 * stream and open alerts.
 *
 * The wall is a synthetic render of the live bounding boxes, not video.
 * Streaming every MJPEG feed into a browser costs more than the whole
 * inference budget, and the boxes are what the operator actually reads. When a
 * real stream_url exists it goes behind the boxes; nothing else changes.
 *
 * Layout follows the control-room rule: the thing that must be watched
 * continuously (the wall) takes the width, and the things that are read on
 * arrival (stream, alerts) take a fixed column beside it.
 */
import { useState } from "react";
import { getAlerts, getCameras, ackAlert } from "@/lib/api";
import {
  Conf, Dot, Empty, HATCH, LABEL, META, MONO, Panel, Pips, Plate, SANS, Sev, T,
  ago, clock, hopTime, methodColour, rowStyle, statusColour,
} from "@/components/ui";
import { useLive, usePoll } from "@/components/useLive";
import type { Alert, Camera } from "@/contract";

type Box = { bbox: [number, number, number, number]; vehicle_type: string;
             conf: number; plate_text: string | null };

/** Bounding boxes drawn at their true aspect. The sidecar reports pixels in the
 *  source frame, so the tile scales them rather than assuming a size. */
function CameraTile({ camera, boxes, count }:
                    { camera: Camera; boxes: Box[]; count: number }) {
  // The source frame size is not in the contract. Take the extent of what we
  // have seen and never shrink below a sane default, so boxes stay in frame
  // rather than being clipped by a guessed resolution.
  const W = Math.max(960, ...boxes.map((b) => b.bbox[2]));
  const H = Math.max(540, ...boxes.map((b) => b.bbox[3]));
  const offline = camera.status === "offline";

  return (
    <div style={{ border: `1px solid ${T.line}`, background: T.panel, borderRadius: 3,
                  overflow: "hidden", boxShadow: T.shadow }}>
      <div style={{ position: "relative", aspectRatio: "16 / 10", overflow: "hidden",
                    background: "repeating-linear-gradient(135deg,var(--sunk) 0 6px," +
                                "var(--panel2) 6px 12px)" }}>
        {boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.bbox;
          const read = b.plate_text !== null;
          return (
            <div key={i} style={{
              position: "absolute",
              left: `${(x1 / W) * 100}%`, top: `${(y1 / H) * 100}%`,
              width: `${((x2 - x1) / W) * 100}%`, height: `${((y2 - y1) / H) * 100}%`,
              border: `1.5px ${read ? "solid" : "dashed"} ${read ? T.text : T.faint}`,
              borderRadius: 1,
            }}>
              <span style={{
                position: "absolute", left: -1.5, top: -1.5, padding: "1px 3px",
                maxWidth: "calc(100% + 3px)", whiteSpace: "nowrap", overflow: "hidden",
                textOverflow: "ellipsis",
                background: read ? T.text : T.raised,
                color: read ? T.panel : T.faint,
                border: read ? "none" : `1px dashed ${T.line2}`,
                font: `500 8px/1.45 ${MONO}`, letterSpacing: ".04em",
              }}>{b.plate_text ?? b.vehicle_type}</span>
            </div>
          );
        })}
        {/* A slow sweep, so a tile with no vehicles in frame still reads as a
            live feed rather than a frozen one. Hidden when the camera is down,
            where "nothing is moving" is the fact being reported. */}
        {!offline && (
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: "28%",
            pointerEvents: "none", opacity: 0.16,
            background: "linear-gradient(90deg,transparent,var(--panel) 60%,transparent)",
            animation: "asweep 4.6s linear infinite",
          }} />
        )}
        <div style={{
          position: "absolute", left: 8, right: 8, bottom: 6, ...META,
          letterSpacing: ".08em", whiteSpace: "nowrap", overflow: "hidden",
          textOverflow: "ellipsis",
        }}>
          {offline ? "NO SIGNAL" : `${camera.id} · ${boxes.length} in frame`}
        </div>
      </div>
      <div style={{ padding: "8px 9px", display: "flex", alignItems: "center", gap: 7,
                    borderTop: `1px solid ${T.line}` }}>
        <Dot colour={statusColour(camera.status)} pulse={camera.status === "online"} />
        <span style={{ font: `500 11px ${MONO}`, letterSpacing: ".04em" }}>{camera.id}</span>
        <span style={{ flex: 1, minWidth: 0, font: `400 11px ${SANS}`, color: T.dim,
                       overflow: "hidden", textOverflow: "ellipsis",
                       whiteSpace: "nowrap" }}>{camera.name}</span>
        <span className="tnum" style={META}>{count} seen</span>
      </div>
    </div>
  );
}

export default function LiveView() {
  const live = useLive();
  const cameras = usePoll(getCameras, 10_000);
  const stored = usePoll(() => getAlerts(false), 15_000);
  const [acked, setAcked] = useState<Set<string>>(new Set());
  const [paused, setPaused] = useState(false);
  // The stream freezes on the operator's copy, not on the socket. Holding the
  // rows they were reading is the point; dropping the messages would leave a
  // gap in the record when they unpause.
  const [held, setHeld] = useState<typeof live.detections>([]);

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
  const byCamera: Record<string, Box[]> = {};
  const seenTracks = new Set<string>();
  for (const d of live.detections) {
    if (Date.now() - d.at > FRESH_MS) continue;
    const key = `${d.camera_id}:${d.track_id}`;
    if (seenTracks.has(key)) continue;
    seenTracks.add(key);
    (byCamera[d.camera_id] ??= []).push({
      bbox: d.bbox, vehicle_type: d.vehicle_type, conf: d.conf,
      plate_text: d.plate_text,
    });
  }

  const stream = paused ? held : live.detections;
  const wall = cameras ?? [];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.55fr 0.95fr", gap: 14,
                  alignItems: "start" }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline",
                      justifyContent: "space-between", marginBottom: 9 }}>
          <div style={LABEL}>
            Camera wall · {wall.length} camera{wall.length === 1 ? "" : "s"}
          </div>
          <div style={META}>YOLO11s · mAP50 0.974 · PaddleOCR</div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
          {wall.map((c) => (
            <CameraTile key={c.id} camera={c} boxes={byCamera[c.id] ?? []}
                        count={live.counts[c.id] ?? 0} />
          ))}
        </div>
        {!wall.length && (
          <Panel title="Cameras">
            <Empty>None configured. Run <code>npm run db:setup</code>.</Empty>
          </Panel>
        )}

        <Panel
          style={{ marginTop: 16 }}
          flush
          title={`Cross-camera confirmations · ${live.matches.length}`}
          right={<div style={META}>layer agreement, 2 of 3 required</div>}
        >
          {live.matches.length ? (
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: 660 }}>
                {live.matches.slice(0, 6).map((m, i, all) => (
                  <div key={`${m.trajectory_id}:${i}`}
                       style={{ ...rowStyle(i === all.length - 1), font: `400 12px ${SANS}` }}>
                    <div style={{ width: 150, flex: "0 0 150px" }}>
                      <Plate text={m.plate_text} />
                    </div>
                    <div style={{ width: 140, flex: "0 0 140px", font: `400 11px ${MONO}`,
                                  color: T.dim, letterSpacing: ".03em" }}>
                      {m.from_camera} → {m.to_camera}
                    </div>
                    <div style={{ width: 110, flex: "0 0 110px" }}>
                      <Pips method={m.method} />
                    </div>
                    <div style={{ width: 96, flex: "0 0 96px" }}>
                      <Conf value={m.confidence} width={40}
                            colour={methodColour(m.method)} />
                    </div>
                    <div className="tnum" style={{ flex: 1, minWidth: 0, ...META }}>
                      {hopTime(m.travel_time_s)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ padding: "12px 14px" }}>
              <Empty>
                None yet. A match is confirmed when two of three layers agree —
                plate text, Re-ID similarity, or travel-time feasibility.
              </Empty>
            </div>
          )}
        </Panel>
      </div>

      <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 14 }}>
        <Panel
          flush
          title={
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Dot colour={live.connected ? T.ok : T.faint} pulse={live.connected} />
              Detection stream
            </span>
          }
          right={
            <button onClick={() => { setHeld(live.detections); setPaused(!paused); }}
                    style={{ border: `1px solid ${T.line2}`, background: T.raised,
                             borderRadius: 2, padding: "3px 8px",
                             font: `500 9.5px ${MONO}`, letterSpacing: ".08em",
                             color: T.dim, textTransform: "uppercase" }}>
              {paused ? "resume" : "hold"}
            </button>
          }
        >
          <div style={{ maxHeight: 420, overflowY: "auto" }}>
            {stream.length ? stream.slice(0, 24).map((d, i) => {
              const plate = d.plate_text;
              return (
                <div key={`${d.camera_id}:${d.track_id}:${d.at}`} style={{
                  ...rowStyle(), gap: 9, alignItems: "center",
                  background: plate ? "transparent" : T.raised,
                  animation: i === 0 && !paused ? "aslide .34s ease" : undefined,
                }}>
                  <div style={{ width: 34, height: 26, flex: "0 0 34px",
                                background: HATCH, border: `1px solid ${T.line}` }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <Plate text={plate} />
                    <div style={{ marginTop: 4, ...META, letterSpacing: ".03em" }}>
                      {d.camera_id} · T{d.track_id} · {d.vehicle_type}
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div className="tnum" style={{ font: `500 11.5px ${MONO}` }}>
                      {d.conf.toFixed(2)}
                    </div>
                    <div style={{ marginTop: 4, display: "flex", justifyContent: "flex-end" }}>
                      <span style={{ width: 34, height: 4, background: T.sunk,
                                     borderRadius: 1, overflow: "hidden",
                                     display: "inline-block", position: "relative" }}>
                        <span style={{ position: "absolute", inset: "0 auto 0 0",
                                       width: `${d.conf * 100}%`,
                                       background: plate ? T.text : T.line2 }} />
                      </span>
                    </div>
                  </div>
                </div>
              );
            }) : (
              <div style={{ padding: "12px 14px" }}>
                <Empty>Waiting. Start the worker: <code>npm run worker</code></Empty>
              </div>
            )}
          </div>
        </Panel>

        <Panel flush title={`Alerts · ${alerts.length}`}>
          {alerts.length ? alerts.slice(0, 8).map((a, i, all) => (
            <div key={a.id} style={{
              display: "flex", alignItems: "flex-start", gap: 9, padding: "11px 12px",
              borderBottom: i === all.length - 1 ? "none" : `1px solid ${T.line}`,
            }}>
              <Sev severity={a.severity} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 7,
                              flexWrap: "wrap" }}>
                  <span style={{ font: `500 12px ${SANS}` }}>{a.kind.replace("_", " ")}</span>
                  <span style={META}>{a.camera_id ?? "city"}</span>
                  {a.plate_text && <Plate text={a.plate_text} size={11} />}
                </div>
                <div style={{ marginTop: 3, font: `400 11px/1.45 ${SANS}`, color: T.dim }}>
                  {a.detail}
                </div>
              </div>
              <button onClick={() => void onAck(a.id)} title={ago(a.ts)} style={{
                background: "transparent", border: `1px solid ${T.line}`, color: T.faint,
                borderRadius: 2, font: `500 9.5px ${MONO}`, letterSpacing: ".08em",
                padding: "3px 7px", textTransform: "uppercase",
              }}>ack</button>
              <div className="tnum" style={META}>{clock(a.ts)}</div>
            </div>
          )) : (
            <div style={{ padding: "12px 14px" }}><Empty>Nothing open.</Empty></div>
          )}
        </Panel>
      </div>
    </div>
  );
}
