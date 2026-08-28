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
 * the analytics beside the fixed cameras. It carries no road links, so layer 3
 * of the association engine abstains on it rather than pretending to know how
 * far it is from anywhere.
 */
import { useEffect, useState } from "react";
import { createDevice, getDevices, pairDeviceUrl, revokeDevice } from "@/lib/api";
import {
  Cols, Dot, Empty, LABEL, META, MONO, Panel, SANS, T, ago, rowStyle,
} from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Device } from "@/contract";

const colourFor = (s: Device["status"]) =>
  s === "live" ? T.ok : s === "stale" ? T.warn : s === "waiting" ? T.info : T.line2;

export default function DevicesView() {
  const devices = usePoll(getDevices, 3000);
  const [origin, setOrigin] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [urlFor, setUrlFor] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState("");

  // window is not available while this renders on the server, and the pair URL
  // must be the address the OPERATOR reached this page on — that is the one a
  // phone on the same network can also reach.
  useEffect(() => setOrigin(window.location.origin), []);

  const secure = typeof window !== "undefined" &&
    (window.isSecureContext || window.location.hostname === "localhost");

  const list = devices ?? [];
  // The card shows the newest code nobody has claimed. Issuing a second one
  // while the first is still waiting is how an operator ends up reading the
  // wrong six characters down a phone line.
  const pending = list.find((d) => d.status === "waiting") ?? null;
  const pairUrl = pending ? `${origin}${pending.pair_url}` : "";

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

  return (
    <div style={{ display: "grid", gridTemplateColumns: "400px 1fr", gap: 14,
                  alignItems: "start" }}>
      <Panel flush style={{ boxShadow: T.shadow }} title="Pair a phone as a camera">
        <div style={{ padding: "18px 16px" }}>
          <div style={{ font: `400 12px/1.6 ${SANS}`, color: T.dim }}>
            Open the link on the phone. It streams into the same pipeline as
            every other camera — no install, no app.
          </div>

          {pending ? (
            <>
              {/* Character by character, in boxes. A six-character code read
                  aloud or typed on a handset is the one string in this system a
                  human transcribes, and run together it is misread. */}
              <div style={{ marginTop: 15, display: "flex", gap: 6 }}>
                {pending.code.split("").map((ch, i) => (
                  <div key={i} style={{
                    flex: 1, aspectRatio: "3 / 4", display: "grid", placeItems: "center",
                    border: `1px solid ${T.line2}`, background: T.raised, borderRadius: 2,
                    font: `600 24px ${MONO}`,
                  }}>{ch}</div>
                ))}
              </div>

              <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 8,
                            padding: "10px 12px", border: `1px solid ${T.line}`,
                            background: T.raised, borderRadius: 2 }}>
                <a href={pairUrl} target="_blank" rel="noreferrer" style={{
                  flex: 1, minWidth: 0, font: `400 11.5px ${MONO}`, color: T.dim,
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>{pairUrl}</a>
                <button onClick={() => {
                  navigator.clipboard?.writeText(pairUrl).then(
                    () => { setCopied(true); setTimeout(() => setCopied(false), 1500); },
                    () => {},
                  );
                }} style={{ border: `1px solid ${T.line2}`, background: T.panel,
                            borderRadius: 2, padding: "5px 9px", font: `500 9.5px ${MONO}`,
                            letterSpacing: ".08em", color: T.dim }}>
                  {copied ? "COPIED" : "COPY"}
                </button>
              </div>

              <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 10,
                            paddingTop: 14, borderTop: `1px solid ${T.line}` }}>
                <Dot colour={T.warn} pulse />
                <div style={{ flex: 1, font: `400 11.5px ${SANS}`, color: T.dim }}>
                  Waiting for a device. Issued {ago(pending.created_at)}.
                </div>
              </div>
            </>
          ) : (
            <div style={{ marginTop: 15 }}>
              <label style={{ ...LABEL, fontSize: 9.5 }}>Name it (optional)</label>
              <input value={label} onChange={(e) => setLabel(e.target.value)}
                     placeholder="South gate phone"
                     style={{ display: "block", width: "100%", marginTop: 6,
                              padding: "8px 10px", background: T.raised,
                              border: `1px solid ${T.line2}`, borderRadius: 2,
                              color: T.text, font: `400 13px ${SANS}`, outline: "none" }} />
              <button onClick={() => void add()} disabled={busy} style={{
                marginTop: 12, width: "100%", padding: "10px",
                background: busy ? T.raised : T.text, color: busy ? T.faint : T.panel,
                border: "none", borderRadius: 2, font: `500 13px ${SANS}`,
              }}>{busy ? "issuing…" : "Issue a pairing code"}</button>
            </div>
          )}

          {error && <p style={{ color: T.bad, font: `400 12px ${SANS}`,
                                marginTop: 12 }}>{error}</p>}

          <div style={{ marginTop: 16, paddingTop: 14, borderTop: `1px solid ${T.line}`,
                        font: `400 11px/1.7 ${SANS}`, color: T.faint }}>
            <strong style={{ color: T.dim }}>This laptop&apos;s own webcam</strong> works
            straight away — localhost is always a secure context.
            <br /><br />
            <strong style={{ color: T.dim }}>A phone</strong> needs one of two things,
            because browsers refuse camera access on a plain http address.{" "}
            {secure
              ? "This page is on a secure origin, so the phone page will work directly."
              : "This page is on plain http, so a phone browser will refuse. Either " +
                "serve over HTTPS — scripts/dev-https.sh — or attach a stream URL " +
                "from an IP-camera app to a waiting code below."}
          </div>
        </div>
      </Panel>

      <Panel flush style={{ minWidth: 0 }} title="Paired devices"
             right={<span style={META}>stale after 20 s without a frame</span>}>
        {list.length === 0 ? (
          <div style={{ padding: "12px 14px" }}>
            <Empty>
              No devices yet. Issue a code, then open its link on the phone or on
              this machine.
            </Empty>
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <div style={{ minWidth: 660 }}>
              <Cols>
                <div style={{ width: 92, flex: "0 0 92px" }}>Code</div>
                <div style={{ width: 88, flex: "0 0 88px" }}>Camera</div>
                <div style={{ width: 84, flex: "0 0 84px" }}>Kind</div>
                <div style={{ flex: 1 }}>Status</div>
                <div style={{ width: 112, flex: "0 0 112px" }}>Last frame</div>
                <div style={{ width: 96, flex: "0 0 96px", textAlign: "right" }} />
              </Cols>
              {list.map((d, i) => (
                <div key={d.id}>
                  <div style={rowStyle(i === list.length - 1 && urlFor !== d.code)}>
                    <div style={{ width: 92, flex: "0 0 92px", font: `600 12.5px ${MONO}`,
                                  letterSpacing: ".09em" }}>{d.code}</div>
                    <div style={{ width: 88, flex: "0 0 88px", font: `400 11.5px ${MONO}`,
                                  color: T.dim }}>{d.camera_id}</div>
                    <div style={{ width: 84, flex: "0 0 84px", font: `400 11.5px ${SANS}`,
                                  color: T.dim }}>{d.kind ?? "—"}</div>
                    <div style={{ flex: 1, minWidth: 0, display: "flex",
                                  alignItems: "center", gap: 7 }}>
                      <Dot colour={colourFor(d.status)} pulse={d.status === "live"} />
                      <span style={{ font: `400 11.5px ${SANS}` }}>{d.status}</span>
                      {d.label && <span style={META}>{d.label}</span>}
                    </div>
                    <div className="tnum" style={{ width: 112, flex: "0 0 112px",
                                                   font: `400 11.5px ${MONO}`,
                                                   color: T.dim }}>
                      {d.last_frame_at ? ago(d.last_frame_at)
                        : d.paired_at ? `paired ${ago(d.paired_at)}` : "—"}
                    </div>
                    <div style={{ width: 96, flex: "0 0 96px", display: "flex",
                                  gap: 5, justifyContent: "flex-end" }}>
                      {d.status !== "revoked" && (
                        <>
                          <button onClick={() => setUrlFor(urlFor === d.code ? null : d.code)}
                                  title="Attach an IP-camera stream URL instead"
                                  style={{ border: `1px solid ${T.line2}`,
                                           background: "transparent", color: T.dim,
                                           borderRadius: 2, padding: "3px 7px",
                                           font: `500 9.5px ${MONO}`,
                                           letterSpacing: ".08em" }}>URL</button>
                          <button onClick={() => void revokeDevice(d.id)} style={{
                            border: `1px solid ${T.line2}`, background: "transparent",
                            color: T.faint, borderRadius: 2, padding: "3px 7px",
                            font: `500 9.5px ${MONO}`, letterSpacing: ".08em",
                          }}>REVOKE</button>
                        </>
                      )}
                    </div>
                  </div>
                  {urlFor === d.code && (
                    <div style={{ display: "flex", gap: 6, padding: "0 12px 12px" }}>
                      <input value={streamUrl} autoFocus
                             onChange={(e) => setStreamUrl(e.target.value)}
                             placeholder="http://192.168.1.20:8080/video or rtsp://…"
                             style={{ flex: 1, padding: "7px 10px", background: T.raised,
                                      border: `1px solid ${T.line2}`, borderRadius: 2,
                                      color: T.text, font: `400 12px ${MONO}`,
                                      outline: "none" }} />
                      <button onClick={() => void attachUrl(d.code)} style={{
                        background: T.text, color: T.panel, border: "none", borderRadius: 2,
                        padding: "0 14px", font: `500 12px ${SANS}`,
                      }}>Attach</button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        <div style={{ padding: "13px 14px", borderTop: `1px solid ${T.line}`,
                      display: "flex", gap: 22, flexWrap: "wrap" }}>
          {([["live — frames arriving", T.ok],
             ["stale — paired, no frames", T.warn],
             ["waiting — code unused", T.info],
             ["revoked", T.line2]] as const).map(([text, colour]) => (
            <div key={text} style={{ display: "flex", alignItems: "center", gap: 7,
                                     font: `400 11px ${SANS}`, color: T.dim }}>
              <Dot colour={colour} />{text}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
