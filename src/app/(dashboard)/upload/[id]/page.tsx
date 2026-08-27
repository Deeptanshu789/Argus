"use client";
/**
 * One upload's results: the plates read out of the operator's own video, and
 * any journey Module C stitched between two of their videos.
 *
 * Scoped to this upload's cameras only. The demo cameras are not filtered out
 * here so much as never asked for — this page reads a single upload, so the
 * question of what else is in the database does not arise.
 */
import { use, useState } from "react";
import Link from "next/link";
import { cancelUpload, getUpload } from "@/lib/api";
import { Empty, Panel, T, Tag, ago, methodColour } from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Upload } from "@/contract";

const statusColour = (s: Upload["status"]) =>
  s === "done" ? T.ok : s === "error" ? T.bad : s === "running" ? T.warn : T.dim;

/** Only these two have anything left to stop. */
const stoppable = (s: Upload["status"]) => s === "running" || s === "pending";

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

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <Panel
        title={upload.label ?? `Upload ${upload.id}`}
        right={<Link href="/upload" style={{ color: T.dim, fontSize: 12,
                                             textDecoration: "none" }}>all uploads</Link>}
      >
        <div style={{ display: "flex", gap: ".5rem", alignItems: "center",
                      flexWrap: "wrap", marginBottom: ".8rem" }}>
          <Tag colour={statusColour(upload.status)}>{upload.status}</Tag>
          <span style={{ fontSize: 12, color: T.dim }}>{ago(upload.created_at)}</span>
          {upload.gap_seconds !== null && (
            <Tag colour={T.ok}>{upload.gap_seconds}s between cameras</Tag>
          )}
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
              style={{
                background: "transparent", border: `1px solid ${T.line}`,
                color: stopping ? T.dim : T.bad, borderRadius: 4, fontSize: 11,
                padding: "2px 9px", cursor: stopping ? "default" : "pointer",
                marginLeft: "auto",
              }}
            >{stopping ? "stopping…" : "stop scanning"}</button>
          )}
        </div>

        {stopError && (
          <p style={{ color: T.bad, fontSize: 12, marginBottom: ".8rem" }}>{stopError}</p>
        )}

        {upload.status === "cancelled" && (
          <p style={{ color: T.dim, fontSize: 12, marginBottom: ".8rem" }}>
            Stopped by an operator. Everything read before then is below.
          </p>
        )}

        {upload.error && (
          <p style={{ color: T.bad, fontSize: 12, marginBottom: ".8rem" }}>{upload.error}</p>
        )}

        <div style={{ display: "grid", gap: ".5rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          {upload.sources.map((s) => (
            <div key={s.camera_id} style={{
              background: T.raised, border: `1px solid ${T.line}`,
              borderRadius: 8, padding: ".6rem .75rem",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: ".4rem" }}>
                <Tag colour={statusColour(s.status)}>{s.status}</Tag>
                <code style={{ fontSize: 11, color: T.dim }}>{s.camera_id}</code>
              </div>
              <div style={{ fontSize: 13, marginTop: 5, overflow: "hidden",
                            textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {s.filename}
              </div>
              <div style={{ fontSize: 11, color: T.dim, marginTop: 3 }}>
                {s.tracks} vehicles · <strong style={{ color: T.accent }}>{s.plates}</strong> plates
              </div>
              {s.error && (
                <div style={{ fontSize: 11, color: T.bad, marginTop: 3 }}>{s.error}</div>
              )}
            </div>
          ))}
        </div>
      </Panel>

      {upload.sources.length > 1 && (
        <Panel title={`Same vehicle in more than one video — ${trajectories.length}`}>
          {trajectories.length === 0 ? (
            <Empty>
              {upload.status === "done"
                ? "No vehicle was confirmed in two of these videos. Module C " +
                  "requires two of its three layers to agree, so a single plate " +
                  "read is not enough on its own."
                : "Nothing yet — matching happens as each vehicle leaves frame."}
            </Empty>
          ) : trajectories.map((t) => (
            <div key={t.id} style={{ padding: ".55rem 0",
                                     borderBottom: `1px solid ${T.line}` }}>
              <div style={{ fontSize: 13 }}>
                {t.plate_text ?? <span style={{ color: T.reid }}>plate not read</span>}
              </div>
              <div style={{ display: "flex", gap: ".4rem", marginTop: 5,
                            flexWrap: "wrap", alignItems: "center" }}>
                {t.hops.map((h, i) => (
                  <span key={i} style={{ display: "flex", alignItems: "center",
                                         gap: ".35rem", fontSize: 12 }}>
                    {i === 0 && <code>{byCamera.get(h.from_camera) ?? h.from_camera}</code>}
                    <Tag colour={methodColour(h.method)}>
                      {h.method} {h.confidence.toFixed(2)} · {h.travel_time_s}s
                    </Tag>
                    <code>{byCamera.get(h.to_camera) ?? h.to_camera}</code>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </Panel>
      )}

      <Panel title={`Plates read — ${named.length} of ${plates.length} vehicles`}>
        {plates.length === 0 ? (
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
        ) : (
          <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ color: T.dim, fontSize: 11, textAlign: "left" }}>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Plate</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Conf</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Type</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Video</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Speed</th>
                </tr>
              </thead>
              <tbody>
                {plates.map((p) => (
                  <tr key={`${p.camera_id}:${p.track_id}`}
                      style={{ borderTop: `1px solid ${T.line}` }}>
                    <td style={{ padding: "5px 6px", fontFamily: "ui-monospace, monospace" }}>
                      {p.plate_text ?? <span style={{ color: T.faint }}>not read</span>}
                    </td>
                    <td style={{ padding: "5px 6px", color: T.dim }}>
                      {p.plate_conf === null ? "—" : p.plate_conf.toFixed(2)}
                    </td>
                    <td style={{ padding: "5px 6px", color: T.dim }}>{p.vehicle_type}</td>
                    <td style={{ padding: "5px 6px", color: T.dim, maxWidth: 220,
                                 overflow: "hidden", textOverflow: "ellipsis",
                                 whiteSpace: "nowrap" }}>
                      {byCamera.get(p.camera_id) ?? p.camera_id}
                    </td>
                    <td style={{ padding: "5px 6px", color: T.dim }}>
                      {p.speed_kmh === null ? "—" : `${p.speed_kmh} km/h`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ color: T.faint, fontSize: 11, marginTop: ".8rem", lineHeight: 1.6 }}>
              Speed needs a metres-per-pixel survey of the camera, which an
              uploaded file does not have, so it is an estimate from a default
              scale rather than a measurement.
            </p>
          </div>
        )}
      </Panel>
    </div>
  );
}
