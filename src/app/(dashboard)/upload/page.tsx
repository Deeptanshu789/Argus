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
import { Empty, Panel, T, Tag, ago } from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Upload } from "@/contract";

const statusColour = (s: Upload["status"]) =>
  s === "done" ? T.ok : s === "error" ? T.bad : s === "running" ? T.warn : T.dim;

export default function UploadView() {
  const uploads = usePoll(() => getUploads(20), 3000);
  const [files, setFiles] = useState<File[]>([]);
  const [gap, setGap] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
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

  return (
    <div style={{ display: "grid", gap: "1rem",
                  gridTemplateColumns: "minmax(0, 420px) minmax(0, 1fr)" }}>
      <Panel title="Analyse a video">
        <input
          ref={input}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.webm,.m4v"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          style={{ width: "100%", fontSize: 13, color: T.dim }}
        />

        {files.length > 0 && (
          <ul style={{ listStyle: "none", padding: 0, margin: ".8rem 0 0",
                       fontSize: 12, color: T.dim }}>
            {files.map((f, i) => (
              <li key={i} style={{ display: "flex", justifyContent: "space-between",
                                   padding: "2px 0" }}>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis",
                               whiteSpace: "nowrap" }}>
                  <Tag colour={T.accent}>camera {i + 1}</Tag> {f.name}
                </span>
                <span style={{ color: T.faint, flex: "0 0 auto", marginLeft: ".5rem" }}>
                  {(f.size / 1e6).toFixed(1)} MB
                </span>
              </li>
            ))}
            <li style={{ color: T.faint, paddingTop: 4 }}>{totalMB.toFixed(1)} MB total</li>
          </ul>
        )}

        <label style={{ display: "block", marginTop: "1rem", fontSize: 12, color: T.dim }}>
          Label (optional)
          <input value={label} onChange={(e) => setLabel(e.target.value)}
                 placeholder="Silk Board, Tuesday morning"
                 style={{ display: "block", width: "100%", marginTop: 4, padding: ".4rem .5rem",
                          background: T.raised, border: `1px solid ${T.line}`,
                          borderRadius: 5, color: T.text, fontSize: 13 }} />
        </label>

        {files.length > 1 && (
          <label style={{ display: "block", marginTop: ".8rem", fontSize: 12, color: T.dim }}>
            Expected travel time between these cameras, in seconds (optional)
            <input value={gap} onChange={(e) => setGap(e.target.value.replace(/[^0-9]/g, ""))}
                   inputMode="numeric" placeholder="e.g. 180"
                   style={{ display: "block", width: "100%", marginTop: 4, padding: ".4rem .5rem",
                            background: T.raised, border: `1px solid ${T.line}`,
                            borderRadius: 5, color: T.text, fontSize: 13 }} />
            <span style={{ display: "block", marginTop: 6, color: T.faint, lineHeight: 1.5 }}>
              How long a vehicle takes to drive between them. Given it, the
              travel-time layer can confirm a journey alongside the plate; left
              blank it abstains, which is better than a guess — a wrong number
              makes it reject real matches as physically impossible.
              <br /><br />
              The videos are assumed to cover the <strong>same period</strong>,
              as two cameras at a junction would. This is not a playback offset:
              the gap between sightings is measured from the footage itself.
            </span>
          </label>
        )}

        {error && (
          <p style={{ color: T.bad, fontSize: 12, marginTop: ".8rem" }}>{error}</p>
        )}

        <button onClick={() => void submit()} disabled={!files.length || busy}
                style={{
                  marginTop: "1rem", width: "100%", padding: ".55rem",
                  background: files.length && !busy ? T.accent : T.raised,
                  color: files.length && !busy ? "#08131f" : T.faint,
                  border: "none", borderRadius: 6, fontSize: 13, fontWeight: 600,
                  cursor: files.length && !busy ? "pointer" : "default",
                }}>
          {busy ? "uploading…" : files.length > 1
            ? `Analyse ${files.length} videos` : "Analyse video"}
        </button>

        <p style={{ color: T.faint, fontSize: 11, marginTop: ".9rem", lineHeight: 1.6 }}>
          Processing runs at 5 frames a second on CPU, so expect roughly a
          quarter of the clip&apos;s own length per file. The worker
          (<code>npm run worker</code>) must be running — it owns every sidecar.
        </p>
      </Panel>

      <Panel title={`Uploads — ${(uploads ?? []).length}`}>
        {(uploads ?? []).length === 0 ? (
          <Empty>
            Nothing uploaded yet. Pick a video on the left; its results stay on
            their own page and never mix with the demo cameras.
          </Empty>
        ) : (
          <div>
            {(uploads ?? []).map((u) => {
              const tracks = u.sources.reduce((n, s) => n + s.tracks, 0);
              const plates = u.sources.reduce((n, s) => n + s.plates, 0);
              return (
                <Link key={u.id} href={`/upload/${u.id}`} style={{
                  display: "block", textDecoration: "none", color: T.text,
                  padding: ".65rem .3rem", borderBottom: `1px solid ${T.line}`,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: ".5rem",
                                flexWrap: "wrap" }}>
                    <Tag colour={statusColour(u.status)}>{u.status}</Tag>
                    <strong style={{ fontSize: 13 }}>
                      {u.label ?? u.sources.map((s) => s.filename).join(", ") ?? `upload ${u.id}`}
                    </strong>
                    <span style={{ fontSize: 11, color: T.faint, marginLeft: "auto" }}>
                      {ago(u.created_at)}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: T.dim, marginTop: 3 }}>
                    {u.sources.length} video{u.sources.length === 1 ? "" : "s"} ·{" "}
                    {tracks} vehicles · {plates} plates read
                    {u.gap_seconds !== null && ` · ${u.gap_seconds}s apart`}
                  </div>
                  {u.error && (
                    <div style={{ fontSize: 11, color: T.bad, marginTop: 3 }}>{u.error}</div>
                  )}
                </Link>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
