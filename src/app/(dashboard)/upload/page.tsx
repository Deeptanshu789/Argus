"use client";
/**
 * Upload video from this machine and run it through the same pipeline the live
 * cameras use.
 *
 * Each file becomes its own camera, so a two-file upload is a two-camera
 * problem and Module C looks for the same vehicle in both. Uploaded cameras are
 * excluded from the live dashboard, map, analytics and alerts — footage brought
 * in for analysis must not move the city's numbers.
 */
import { useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createUpload, getUploads } from "@/lib/api";
import {
  Empty, LABEL, META, MONO, Panel, SANS, T, Tag, ago, rowStyle, statusColour,
} from "@/components/ui";
import { usePoll } from "@/components/useLive";

export default function UploadView() {
  const uploads = usePoll(() => getUploads(20), 3000);
  const [files, setFiles] = useState<File[]>([]);
  const [gap, setGap] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const submit = async () => {
    if (!files.length || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createUpload(
        files, gap.trim() === "" ? null : Number(gap), label.trim() || null);
      setFiles([]);
      setGap("");
      setLabel("");
      if (input.current) input.current.value = "";
      // Straight to the results, which fill in as the worker decodes. Landing
      // back on the form with nothing visibly different reads as a failure.
      router.push(`/upload/${created.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const totalMB = files.reduce((n, f) => n + f.size, 0) / 1e6;
  const field: React.CSSProperties = {
    display: "block", width: "100%", marginTop: 6, padding: "8px 10px",
    background: T.raised, border: `1px solid ${T.line2}`, borderRadius: 2,
    color: T.text, font: `400 13px ${SANS}`, outline: "none",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 14,
                  alignItems: "start" }}>
      <div style={{ minWidth: 0 }}>
        {/* A real drop target, not a styled file input. The operator's clip is
            already on their desktop; making them navigate a file dialog to it
            is a step the design correctly removes. */}
        <div
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const dropped = Array.from(e.dataTransfer.files)
              .filter((f) => f.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name));
            if (dropped.length) setFiles(dropped);
          }}
          style={{
            padding: "34px 20px", textAlign: "center", borderRadius: 3,
            border: `1.5px dashed ${over ? T.accent : T.line2}`,
            background: over ? T.accentSoft : T.panel,
            transition: "background .15s, border-color .15s",
          }}
        >
          <div style={{ width: 46, height: 46, border: `1.5px solid ${T.line2}`,
                        borderRadius: 2, display: "grid", placeItems: "center",
                        margin: "0 auto" }}>
            <div style={{ width: 0, height: 0, marginLeft: 3,
                          borderLeft: `11px solid ${T.dim}`,
                          borderTop: "7px solid transparent",
                          borderBottom: "7px solid transparent" }} />
          </div>
          <div style={{ marginTop: 14, font: `500 15px ${SANS}` }}>
            Drop a video file to add it as a camera
          </div>
          <div style={{ marginTop: 6, font: `400 12px/1.6 ${SANS}`, color: T.dim,
                        maxWidth: 420, marginLeft: "auto", marginRight: "auto" }}>
            It runs the identical pipeline as a live feed — same detector, same
            reader, same three-layer matcher. Results join the same trajectory graph.
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 9, justifyContent: "center" }}>
            <button onClick={() => input.current?.click()} style={{
              padding: "9px 16px", border: `1px solid ${T.text}`, background: T.text,
              color: T.panel, borderRadius: 2, font: `500 12px ${SANS}`,
            }}>Choose file</button>
            {files.length > 0 && (
              <button onClick={() => { setFiles([]); if (input.current) input.current.value = ""; }}
                      style={{ padding: "9px 16px", border: `1px solid ${T.line2}`,
                               background: "transparent", color: T.dim, borderRadius: 2,
                               font: `500 12px ${SANS}` }}>Clear</button>
            )}
          </div>
          <input
            ref={input} type="file" hidden multiple
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.webm,.m4v"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
          <div style={{ marginTop: 14, font: `400 10px ${MONO}`, color: T.faint,
                        letterSpacing: ".06em" }}>
            MP4 · MOV · MKV · AVI · sampled at 5 fps on CPU
          </div>
        </div>

        {files.length > 0 && (
          <Panel flush style={{ marginTop: 14 }}
                 title={`${files.length} file${files.length === 1 ? "" : "s"} ready`}
                 right={<span className="tnum" style={META}>{totalMB.toFixed(1)} MB</span>}>
            {files.map((f, i) => (
              <div key={i} style={rowStyle(i === files.length - 1)}>
                <Tag colour={T.accent}>camera {i + 1}</Tag>
                <span style={{ flex: 1, minWidth: 0, font: `400 12px ${SANS}`,
                               overflow: "hidden", textOverflow: "ellipsis",
                               whiteSpace: "nowrap" }}>{f.name}</span>
                <span className="tnum" style={META}>{(f.size / 1e6).toFixed(1)} MB</span>
              </div>
            ))}

            <div style={{ padding: "12px 14px", borderTop: `1px solid ${T.line}` }}>
              <label style={{ ...LABEL, fontSize: 9.5 }}>Label (optional)</label>
              <input value={label} onChange={(e) => setLabel(e.target.value)}
                     placeholder="Rasulgarh junction, Tuesday morning" style={field} />

              {files.length > 1 && (
                <>
                  <label style={{ ...LABEL, fontSize: 9.5, display: "block",
                                  marginTop: 14 }}>
                    Travel time between these cameras (seconds, optional)
                  </label>
                  <input value={gap} inputMode="numeric" placeholder="e.g. 180"
                         onChange={(e) => setGap(e.target.value.replace(/[^0-9]/g, ""))}
                         style={field} />
                  <p style={{ margin: "8px 0 0", font: `400 11px/1.6 ${SANS}`,
                              color: T.faint }}>
                    How long a vehicle takes to drive between them. Given it, the
                    travel-time layer can confirm a journey alongside the plate;
                    left blank it abstains, which is better than a guess — a wrong
                    number makes it reject real matches as impossible. The videos
                    are assumed to cover the <strong>same period</strong>, as two
                    cameras at a junction would; this is not a playback offset.
                  </p>
                </>
              )}

              {error && <p style={{ color: T.bad, font: `400 12px ${SANS}`,
                                    marginTop: 12 }}>{error}</p>}

              <button onClick={() => void submit()} disabled={busy} style={{
                marginTop: 14, width: "100%", padding: "10px",
                background: busy ? T.raised : T.text, color: busy ? T.faint : T.panel,
                border: "none", borderRadius: 2, font: `500 13px ${SANS}`,
              }}>
                {busy ? "uploading…" : files.length > 1
                  ? `Analyse ${files.length} videos` : "Analyse video"}
              </button>
              <p style={{ margin: "10px 0 0", font: `400 11px/1.6 ${SANS}`, color: T.faint }}>
                Processing runs at 5 frames a second on CPU, so expect roughly a
                quarter of the clip&apos;s own length per file. The worker
                (<code>npm run worker</code>) must be running — it owns every sidecar.
              </p>
            </div>
          </Panel>
        )}
      </div>

      <Panel flush style={{ minWidth: 0 }} title="Uploads"
             right={<span className="tnum" style={META}>{(uploads ?? []).length}</span>}>
        {(uploads ?? []).length === 0 ? (
          <div style={{ padding: "12px 14px" }}>
            <Empty>
              Nothing uploaded yet. Drop a video on the left; its results stay on
              their own page and never mix with the city cameras.
            </Empty>
          </div>
        ) : (uploads ?? []).map((u, i, all) => {
          const tracks = u.sources.reduce((n, s) => n + s.tracks, 0);
          const plates = u.sources.reduce((n, s) => n + s.plates, 0);
          return (
            <Link key={u.id} href={`/upload/${u.id}`} style={{
              display: "block", textDecoration: "none", color: T.text,
              padding: "var(--rowpad) 14px",
              borderBottom: i === all.length - 1 ? "none" : `1px solid ${T.line}`,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8,
                            flexWrap: "wrap" }}>
                <Tag colour={statusColour(u.status)}>{u.status}</Tag>
                <span style={{ flex: 1, minWidth: 0, font: `500 12.5px ${SANS}`,
                               overflow: "hidden", textOverflow: "ellipsis",
                               whiteSpace: "nowrap" }}>
                  {u.label ?? u.sources.map((s) => s.filename).join(", ") ?? `upload ${u.id}`}
                </span>
                <span style={META}>{ago(u.created_at)}</span>
              </div>
              <div style={{ marginTop: 5, ...META, letterSpacing: ".03em" }}>
                {u.sources.length} video{u.sources.length === 1 ? "" : "s"} ·{" "}
                {tracks} vehicles · {plates} plates read
                {u.gap_seconds !== null && ` · ${u.gap_seconds}s apart`}
              </div>
              {u.error && (
                <div style={{ marginTop: 3, font: `400 11px ${SANS}`, color: T.bad }}>
                  {u.error}
                </div>
              )}
            </Link>
          );
        })}
      </Panel>
    </div>
  );
}
