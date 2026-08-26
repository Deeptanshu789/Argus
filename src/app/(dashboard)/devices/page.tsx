"use client";
/**
 * Pair a phone, or this laptop, as a live camera.
 *
 * A code is issued here and satisfied in one of two ways, because
 * getUserMedia only works in a secure context: over HTTPS (or on localhost) the
 * phone's own browser captures and pushes frames; over plain http it cannot,
 * and the same code accepts a stream URL from an IP-camera app instead.
 *
 * A paired device is a REAL camera — it appears on the dashboard, the map and
 * the analytics beside CAM1..CAM4. It carries no road links, so layer 3 of the
 * association engine abstains on it rather than pretending to know how far it
 * is from anywhere.
 */
import { useEffect, useState } from "react";
import { createDevice, getDevices, pairDeviceUrl, revokeDevice } from "@/lib/api";
import { Empty, Panel, T, Tag, ago } from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Device } from "@/contract";

const statusColour = (s: Device["status"]) =>
  s === "live" ? T.ok : s === "waiting" ? T.warn : T.faint;

export default function DevicesView() {
  const devices = usePoll(getDevices, 3000);
  const [origin, setOrigin] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [urlFor, setUrlFor] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState("");

  // window is not available while this renders on the server, and the pair URL
  // must be the address the OPERATOR reached this page on — that is the one a
  // phone on the same network can also reach.
  useEffect(() => setOrigin(window.location.origin), []);

  const secure = typeof window !== "undefined" &&
    (window.isSecureContext || window.location.hostname === "localhost");

  const add = async () => {
    setBusy(true);
    setError(null);
    try {
      await createDevice(label.trim() || null);
      setLabel("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const attachUrl = async (code: string) => {
    setError(null);
    try {
      await pairDeviceUrl(code, streamUrl.trim());
      setUrlFor(null);
      setStreamUrl("");
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const lanHint = origin.replace(/^https?:\/\//, "");

  return (
    <div style={{ display: "grid", gap: "1rem",
                  gridTemplateColumns: "minmax(0, 380px) minmax(0, 1fr)" }}>
      <Panel title="Add a camera">
        <label style={{ display: "block", fontSize: 12, color: T.dim }}>
          Name it (optional)
          <input value={label} onChange={(e) => setLabel(e.target.value)}
                 placeholder="South gate phone"
                 style={{ display: "block", width: "100%", marginTop: 4,
                          padding: ".4rem .5rem", background: T.raised,
                          border: `1px solid ${T.line}`, borderRadius: 5,
                          color: T.text, fontSize: 13 }} />
        </label>
        <button onClick={() => void add()} disabled={busy}
                style={{ marginTop: ".8rem", width: "100%", padding: ".55rem",
                         background: busy ? T.raised : T.accent,
                         color: busy ? T.faint : "#08131f", border: "none",
                         borderRadius: 6, fontSize: 13, fontWeight: 600,
                         cursor: busy ? "default" : "pointer" }}>
          {busy ? "issuing…" : "Issue a pairing code"}
        </button>
        {error && <p style={{ color: T.bad, fontSize: 12, marginTop: ".7rem" }}>{error}</p>}

        <div style={{ marginTop: "1.2rem", fontSize: 11, color: T.faint, lineHeight: 1.7 }}>
          <strong style={{ color: T.dim }}>This laptop&apos;s own webcam</strong> works
          straight away — open the pair link below in a new tab.
          <br /><br />
          <strong style={{ color: T.dim }}>A phone</strong> needs one of two things,
          because browsers refuse camera access on a plain http address:
          <br />
          {secure
            ? "This page is on a secure origin, so the phone page will work directly."
            : `This page is on plain http (${lanHint}), so a phone browser will refuse. ` +
              "Either serve over HTTPS — scripts/dev-https.sh — or use the stream-URL " +
              "option on any pending code below with an IP-camera app."}
        </div>
      </Panel>

      <Panel title={`Devices — ${(devices ?? []).length}`}>
        {(devices ?? []).length === 0 ? (
          <Empty>
            No devices yet. Issue a code, then open its link on the phone or on
            this machine.
          </Empty>
        ) : (devices ?? []).map((d) => (
          <div key={d.id} style={{ padding: ".7rem .3rem",
                                   borderBottom: `1px solid ${T.line}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: ".5rem",
                          flexWrap: "wrap" }}>
              <Tag colour={statusColour(d.status)}>{d.status}</Tag>
              <code style={{ fontSize: 18, letterSpacing: ".18em", color: T.accent }}>
                {d.code}
              </code>
              <strong style={{ fontSize: 13 }}>{d.label ?? d.camera_id}</strong>
              <code style={{ fontSize: 11, color: T.dim }}>{d.camera_id}</code>
              {d.kind && <Tag>{d.kind}</Tag>}
              <button onClick={() => void revokeDevice(d.id)} style={{
                marginLeft: "auto", background: "transparent",
                border: `1px solid ${T.line}`, color: T.dim, borderRadius: 4,
                fontSize: 11, padding: "2px 8px", cursor: "pointer" }}>
                revoke
              </button>
            </div>

            <div style={{ fontSize: 11, color: T.dim, marginTop: 5 }}>
              {d.last_frame_at ? `last frame ${ago(d.last_frame_at)}`
                : d.paired_at ? `paired ${ago(d.paired_at)}`
                : "waiting for something to connect"}
              {d.source_url && ` · ${d.source_url}`}
            </div>

            {d.status !== "revoked" && (
              <div style={{ marginTop: ".5rem", display: "flex", gap: ".5rem",
                            flexWrap: "wrap", alignItems: "center" }}>
                <a href={`${origin}${d.pair_url}`} target="_blank" rel="noreferrer"
                   style={{ fontSize: 12, color: T.accent }}>
                  {origin}{d.pair_url}
                </a>
                <button onClick={() => { setUrlFor(urlFor === d.code ? null : d.code); }}
                        style={{ background: "transparent", border: `1px solid ${T.line}`,
                                 color: T.dim, borderRadius: 4, fontSize: 11,
                                 padding: "2px 8px", cursor: "pointer" }}>
                  {urlFor === d.code ? "cancel" : "use a stream URL instead"}
                </button>
              </div>
            )}

            {urlFor === d.code && (
              <div style={{ marginTop: ".5rem", display: "flex", gap: ".4rem" }}>
                <input value={streamUrl} onChange={(e) => setStreamUrl(e.target.value)}
                       placeholder="http://192.168.1.20:8080/video or rtsp://…"
                       style={{ flex: 1, padding: ".35rem .5rem", background: T.raised,
                                border: `1px solid ${T.line}`, borderRadius: 5,
                                color: T.text, fontSize: 12 }} />
                <button onClick={() => void attachUrl(d.code)} style={{
                  background: T.accent, color: "#08131f", border: "none",
                  borderRadius: 5, fontSize: 12, padding: "0 .8rem",
                  fontWeight: 600, cursor: "pointer" }}>attach</button>
              </div>
            )}
          </div>
        ))}
      </Panel>
    </div>
  );
}
