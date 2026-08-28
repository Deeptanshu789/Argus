"use client";
/**
 * One upload's results: what the pipeline found in the operator's own video,
 * and any journey Module C stitched between two of their files.
 *
 * Scoped to this upload's cameras only. The city cameras are not filtered out
 * here so much as never asked for — this page reads a single upload, so the
 * question of what else is in the database does not arise.
 */
import { use, useState } from "react";
import Link from "next/link";
import { cancelUpload, getUpload } from "@/lib/api";
import {
  Bar, Conf, Cols, Dot, Empty, Empty as Nothing, HATCH, LABEL, META, MONO, Panel,
  Pips, Plate, SANS, T, Tag, ago, hopTime, methodColour, rowStyle, statusColour,
} from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Upload } from "@/contract";

/** Only these two have anything left to stop. */
const stoppable = (s: Upload["status"]) => s === "running" || s === "pending";

/** The pipeline, named in the order it runs. These are the real stages in
 *  ml/sidecar.py, with the real thresholds — a progress bar that invented its
 *  own steps would be the one part of this page nobody could check. */
const STAGES: [string, string][] = [
  ["decode", "source → 5 fps"],
  ["detect", "YOLO11s · plate-k12"],
  ["track", "BoT-SORT + Re-ID"],
  ["read", "PaddleOCR · Indian grammar"],
  ["match", "3 layers, 2 must agree"],
];

export default function UploadResultView({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);
  // Polls while the worker is still decoding, so plates appear as they are read
  // rather than all at once at the end.
  const result = usePoll(() => getUpload(id), 3000);

  if (!result) {
    return <Panel title="Upload"><Empty>Loading…</Empty></Panel>;
  }

  const { upload, plates, trajectories } = result;
  const named = plates.filter((p) => p.plate_text);
  const byCamera = new Map(upload.sources.map((s) => [s.camera_id, s.filename]));
  const tracks = upload.sources.reduce((n, s) => n + s.tracks, 0);
  const readTotal = upload.sources.reduce((n, s) => n + s.plates, 0);
  const finished = upload.sources.filter((s) => s.status === "done").length;

  // There is no frame counter in the contract, so progress is measured in the
  // one unit the worker actually reports: files finished. A percentage
  // interpolated from nothing would be the only dishonest number on the page.
  const done = upload.status === "done" || upload.status === "cancelled"
            || upload.status === "error";
  const progress = done ? 1 : finished / Math.max(1, upload.sources.length);
  const stageReached = upload.status === "pending" ? 0
                     : done ? STAGES.length
                     : Math.min(STAGES.length - 1, 1 + Math.floor(progress * 3));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 14,
                  alignItems: "start" }}>
      <div style={{ minWidth: 0, display: "grid", gap: 14 }}>
        <div style={{ border: `1px solid ${T.line}`, background: T.panel,
                      borderRadius: 3, boxShadow: T.shadow }}>
          <div style={{ padding: "12px 14px", borderBottom: `1px solid ${T.line}`,
                        display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ font: `500 13px ${SANS}`, overflow: "hidden",
                            textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {upload.label ?? upload.sources.map((s) => s.filename).join(", ")
                  ?? `Upload ${upload.id}`}
              </div>
              <div className="tnum" style={{ marginTop: 4, ...META, fontSize: 10.5 }}>
                {upload.sources.map((s) => s.camera_id).join(" · ")} ·{" "}
                {upload.sources.length} file{upload.sources.length === 1 ? "" : "s"} ·{" "}
                {ago(upload.created_at)}
                {upload.gap_seconds !== null && ` · ${upload.gap_seconds}s apart`}
              </div>
            </div>
            <Tag colour={statusColour(upload.status)}>{upload.status}</Tag>
            {stoppable(upload.status) && (
              <button
                disabled={stopping}
                onClick={() => {
                  setStopping(true);
                  setStopError(null);
                  cancelUpload(id)
                    .catch((e: Error) => setStopError(e.message))
                    .finally(() => setStopping(false));
                }}
                style={{ border: `1px solid ${stopping ? T.line2 : T.bad}`,
                         background: "transparent", color: stopping ? T.faint : T.bad,
                         borderRadius: 2, padding: "5px 10px", font: `500 9.5px ${MONO}`,
                         letterSpacing: ".09em", textTransform: "uppercase" }}
              >{stopping ? "stopping…" : "stop scanning"}</button>
            )}
            <Link href="/upload" style={{ ...META, textDecoration: "none" }}>all uploads</Link>
          </div>

          <div style={{ padding: 14 }}>
            {stopError && <p style={{ color: T.bad, font: `400 12px ${SANS}`,
                                      margin: "0 0 10px" }}>{stopError}</p>}
            {upload.error && <p style={{ color: T.bad, font: `400 12px ${SANS}`,
                                         margin: "0 0 10px" }}>{upload.error}</p>}
            {upload.status === "cancelled" && (
              <p style={{ color: T.dim, font: `400 12px ${SANS}`, margin: "0 0 10px" }}>
                Stopped by an operator. Everything read before then is below.
              </p>
            )}

            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <div className="tnum" style={{ font: `500 30px/1 ${MONO}` }}>
                {Math.round(progress * 100)}%
              </div>
              <div className="tnum" style={{ flex: 1, ...META }}>
                {finished} of {upload.sources.length} file
                {upload.sources.length === 1 ? "" : "s"} finished
                {!done && " · roughly a quarter of each clip's own length on CPU"}
              </div>
            </div>
            <div style={{ marginTop: 11, height: 8, background: T.sunk, borderRadius: 1,
                          position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", inset: "0 auto 0 0",
                            width: `${progress * 100}%`,
                            background: upload.status === "error" ? T.bad : T.text,
                            transition: "width 1.05s cubic-bezier(.2,.75,.2,1)" }} />
            </div>

            <div style={{ marginTop: 14, display: "grid",
                          gridTemplateColumns: "repeat(4, 1fr)", gap: 1,
                          background: T.line, border: `1px solid ${T.line}` }}>
              {([["Videos", upload.sources.length],
                 ["Vehicles tracked", tracks],
                 ["Plates read", readTotal],
                 ["Matched across", trajectories.length]] as const).map(([label, value]) => (
                <div key={label} style={{ background: T.panel, padding: "11px 12px" }}>
                  <div style={{ ...LABEL, fontSize: 9, letterSpacing: ".11em" }}>{label}</div>
                  <div className="tnum" style={{ marginTop: 6, font: `500 16px ${MONO}` }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 14, display: "flex", gap: 1, background: T.line,
                          border: `1px solid ${T.line}` }}>
              {STAGES.map(([label, detail], i) => {
                const cleared = i < stageReached;
                const active = !done && i === stageReached;
                return (
                  <div key={label} style={{
                    flex: 1, minWidth: 0, padding: "10px 11px", background: T.panel,
                    borderTop: `2px solid ${active ? T.accent : "transparent"}`,
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <Dot colour={cleared ? T.ok : active ? T.accent : T.line2}
                           pulse={active} />
                      <span style={{ font: `500 11px ${SANS}` }}>{label}</span>
                    </div>
                    <div className="tnum" style={{ marginTop: 5, ...META,
                                                   overflow: "hidden",
                                                   textOverflow: "ellipsis",
                                                   whiteSpace: "nowrap" }}>{detail}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {upload.sources.length > 1 && (
          <Panel flush title="Same vehicle in more than one video"
                 right={<span className="tnum" style={META}>{trajectories.length}</span>}>
            {trajectories.length === 0 ? (
              <div style={{ padding: "12px 14px" }}>
                <Nothing>
                  {upload.status === "done"
                    ? "No vehicle was confirmed in two of these videos. Module C " +
                      "requires two of its three layers to agree, so a single plate " +
                      "read is not enough on its own."
                    : "Nothing yet — matching happens as each vehicle leaves frame."}
                </Nothing>
              </div>
            ) : trajectories.map((t, i) => (
              <div key={t.id} style={{ padding: "var(--rowpad) 14px",
                                       borderBottom: i === trajectories.length - 1
                                         ? "none" : `1px solid ${T.line}` }}>
                <Plate text={t.plate_text} />
                <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
                  {t.hops.map((h, j) => (
                    <div key={j} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <Pips method={h.method} rule={false} />
                      <span style={{ flex: 1, minWidth: 0, font: `400 11px ${MONO}`,
                                     color: T.dim, overflow: "hidden",
                                     textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {byCamera.get(h.from_camera) ?? h.from_camera}
                        {" → "}
                        {byCamera.get(h.to_camera) ?? h.to_camera}
                      </span>
                      <span className="tnum" style={META}>{hopTime(h.travel_time_s)}</span>
                      <Conf value={h.confidence} width={44}
                            colour={methodColour(h.method)} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Panel>
        )}

        <Panel flush title="Every vehicle tracked"
               right={<span className="tnum" style={META}>
                 {named.length} of {plates.length} with a plate
               </span>}>
          {plates.length === 0 ? (
            <div style={{ padding: "12px 14px" }}>
              <Empty>
                {upload.status === "pending"
                  ? "Queued. The worker picks this up within a few seconds — it must " +
                    "be running (npm run worker)."
                  : upload.status === "running"
                    ? "Decoding. Vehicles appear here as each one leaves frame."
                    : "No vehicles were tracked. If the footage clearly contains " +
                      "traffic, the vehicles are probably moving too far between " +
                      "processed frames — see ml/botsort.yaml."}
              </Empty>
            </div>
          ) : (
            <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
              <Cols>
                <div style={{ width: 150, flex: "0 0 150px" }}>Plate</div>
                <div style={{ width: 126, flex: "0 0 126px" }}>Confidence</div>
                <div style={{ width: 96, flex: "0 0 96px" }}>Type</div>
                <div style={{ flex: 1 }}>Video</div>
                <div style={{ width: 78, flex: "0 0 78px", textAlign: "right" }}>Speed</div>
              </Cols>
              {plates.map((p, i) => (
                <div key={`${p.camera_id}:${p.track_id}`}
                     style={{ ...rowStyle(i === plates.length - 1),
                              background: p.plate_text ? "transparent" : T.raised }}>
                  <div style={{ width: 150, flex: "0 0 150px" }}>
                    <Plate text={p.plate_text} />
                  </div>
                  <div style={{ width: 126, flex: "0 0 126px" }}>
                    <Conf value={p.plate_conf} />
                  </div>
                  <div style={{ width: 96, flex: "0 0 96px", font: `400 11.5px ${SANS}`,
                                color: T.dim }}>{p.vehicle_type}</div>
                  <div style={{ flex: 1, minWidth: 0, font: `400 12px ${SANS}`,
                                overflow: "hidden", textOverflow: "ellipsis",
                                whiteSpace: "nowrap" }}>
                    {byCamera.get(p.camera_id) ?? p.camera_id}
                  </div>
                  <div className="tnum" style={{ width: 78, flex: "0 0 78px",
                                                 textAlign: "right", ...META }}>
                    {p.speed_kmh === null ? "—" : `${p.speed_kmh} km/h`}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <div style={{ minWidth: 0, display: "grid", gap: 14 }}>
        <Panel flush title="Plates found in this file">
          {named.length ? named.slice(0, 40).map((p, i) => (
            <div key={`${p.camera_id}:${p.track_id}`}
                 style={{ ...rowStyle(i === Math.min(named.length, 40) - 1), gap: 9 }}>
              <div style={{ width: 40, height: 32, flex: "0 0 40px", background: HATCH,
                            border: `1px solid ${T.line}` }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <Plate text={p.plate_text} />
                <div className="tnum" style={{ marginTop: 4, ...META }}>
                  {new Date(p.entry_time).toLocaleTimeString("en-GB", { hour12: false })}
                  {" · "}{p.vehicle_type}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="tnum" style={{ font: `500 11px ${MONO}` }}>
                  {p.plate_conf === null ? "n/a" : p.plate_conf.toFixed(2)}
                </div>
                <div style={{ marginTop: 4 }}>
                  <Bar value={p.plate_conf ?? 0} width={22}
                       colour={p.plate_conf === null ? T.line2 : T.text} />
                </div>
              </div>
            </div>
          )) : (
            <div style={{ padding: "12px 14px" }}>
              <Empty>No plate has passed validation in this file yet.</Empty>
            </div>
          )}
          <div style={{ padding: "12px 14px", borderTop: `1px solid ${T.line}`,
                        font: `400 11px/1.6 ${SANS}`, color: T.faint }}>
            {plates.length - named.length} of {plates.length} tracked vehicles yielded
            no readable plate. That is the ordinary case on real footage — a plate
            under about 100 pixels wide cannot be read at all, and those vehicles are
            still followed on appearance and travel time.
          </div>
        </Panel>

        <Panel flush title="Per file">
          {upload.sources.map((s, i) => (
            <div key={s.camera_id} style={{ ...rowStyle(i === upload.sources.length - 1),
                                            alignItems: "flex-start" }}>
              <Tag colour={statusColour(s.status)}>{s.status}</Tag>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ font: `400 12px ${SANS}`, overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {s.filename}
                </div>
                <div className="tnum" style={{ marginTop: 4, ...META }}>
                  {s.camera_id} · {s.tracks} vehicles · {s.plates} plates
                </div>
                {s.error && <div style={{ marginTop: 3, font: `400 11px ${SANS}`,
                                          color: T.bad }}>{s.error}</div>}
              </div>
            </div>
          ))}
          <div style={{ padding: "12px 14px", borderTop: `1px solid ${T.line}`,
                        font: `400 11px/1.6 ${SANS}`, color: T.faint }}>
            Speed needs a metres-per-pixel survey of the camera, which an uploaded
            file does not have, so it is an estimate from a default scale rather
            than a measurement.
          </div>
        </Panel>
      </div>
    </div>
  );
}
