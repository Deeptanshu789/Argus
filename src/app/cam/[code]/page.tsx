"use client";
/**
 * The page a phone opens. Captures its camera and pushes JPEG frames to the
 * server, which re-serves them as MJPEG for the inference sidecar.
 *
 * Outside the dashboard layout on purpose: whoever is holding the phone is
 * pointing it at a road, not navigating an operator console.
 *
 * FIVE FRAMES A SECOND, and 640 across. That is what the pipeline processes —
 * pushing 30 fps of 1080p would spend the phone's battery and the network on
 * frames the sidecar throws away, and would make the tracker worse, not better,
 * by giving it nothing extra to associate.
 */
import { use, useCallback, useEffect, useRef, useState } from "react";
import { getDeviceByCode } from "@/lib/api";
import { T } from "@/components/ui";
import type { Device } from "@/contract";

const FPS = 5;
const WIDTH = 640;
const QUALITY = 0.7;

type Phase = "checking" | "unknown" | "insecure" | "denied" | "ready" | "streaming" | "lost";

export default function CamPage({ params }: { params: Promise<{ code: string }> }) {
  const { code } = use(params);
  const [device, setDevice] = useState<Device | null>(null);
  const [phase, setPhase] = useState<Phase>("checking");
  const [detail, setDetail] = useState<string>("");
  const [frames, setFrames] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const stream = useRef<MediaStream | null>(null);

  useEffect(() => {
    let alive = true;
    getDeviceByCode(code)
      .then((d) => {
        if (!alive) return;
        if (!d) return setPhase("unknown");
        setDevice(d);
        // Only localhost and https can reach a camera. Saying so up front beats
        // a bare "Permission denied" the operator cannot act on.
        setPhase(window.isSecureContext ? "ready" : "insecure");
      })
      .catch((e: Error) => { if (alive) { setPhase("unknown"); setDetail(e.message); } });
    return () => { alive = false; };
  }, [code]);

  const stop = useCallback(() => {
    socket.current?.close();
    socket.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const startStreaming = async () => {
    setDetail("");
    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({
        // The rear camera on a phone; ignored by a laptop, which has one.
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
        audio: false,
      });
    } catch (e) {
      setPhase("denied");
      setDetail((e as Error).message);
      return;
    }
    stream.current = media;
    if (video.current) {
      video.current.srcObject = media;
      await video.current.play().catch(() => {});
    }

    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}` +
                  `/ws/cam?code=${encodeURIComponent(code)}`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    socket.current = ws;

    ws.onerror = () => { setPhase("lost"); setDetail("the connection failed"); };
    ws.onclose = () => { setPhase((p) => (p === "streaming" ? "lost" : p)); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(String(ev.data)) as { error?: string };
      if (msg.error) { setPhase("unknown"); setDetail(msg.error); ws.close(); }
    };

    ws.onopen = () => {
      setPhase("streaming");
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      let sent = 0;

      const tick = () => {
        const v = video.current;
        if (!ctx || !v || !v.videoWidth || ws.readyState !== WebSocket.OPEN) return;
        canvas.width = WIDTH;
        canvas.height = Math.round((v.videoHeight / v.videoWidth) * WIDTH);
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            if (!blob || ws.readyState !== WebSocket.OPEN) return;
            // Drop this frame rather than queue it if the socket is already
            // backed up: a stale frame is worth nothing to a plate reader.
            if (ws.bufferedAmount > 512 * 1024) return;
            void blob.arrayBuffer().then((b) => {
              ws.send(b);
              setFrames(++sent);
            });
          },
          "image/jpeg", QUALITY,
        );
      };

      const timer = setInterval(tick, 1000 / FPS);
      ws.addEventListener("close", () => clearInterval(timer));
    };
  };

  const box: React.CSSProperties = {
    minHeight: "100vh", background: T.bg, color: T.text, padding: "1.2rem",
    fontFamily: "ui-sans-serif, system-ui, sans-serif",
    display: "flex", flexDirection: "column", gap: "1rem",
  };

  return (
    <div style={box}>
      <header>
        <div style={{ fontWeight: 600 }}>Argus camera</div>
        <div style={{ fontSize: 12, color: T.dim, marginTop: 2 }}>
          code <code style={{ color: T.accent, letterSpacing: ".15em" }}>{code}</code>
          {device && ` · ${device.label ?? device.camera_id}`}
        </div>
      </header>

      <video ref={video} playsInline muted
             style={{ width: "100%", maxWidth: 640, borderRadius: 10,
                      background: "#000", aspectRatio: "4 / 3",
                      objectFit: "cover", border: `1px solid ${T.line}` }} />

      {phase === "checking" && <p style={{ color: T.dim }}>Checking the code…</p>}

      {phase === "unknown" && (
        <p style={{ color: T.bad }}>
          That code is not valid{detail && `: ${detail}`}. Issue a new one on the
          Devices page.
        </p>
      )}

      {phase === "insecure" && (
        <div style={{ color: T.warn, fontSize: 14, lineHeight: 1.6 }}>
          <strong>This browser will not allow camera access here.</strong>
          <p style={{ color: T.dim, fontSize: 13 }}>
            Camera capture needs a secure context, and this page was opened over
            plain <code>http</code>. Two ways forward:
          </p>
          <ol style={{ color: T.dim, fontSize: 13, paddingLeft: "1.1rem" }}>
            <li>Serve Argus over HTTPS — run <code>scripts/dev-https.sh</code> on
              the machine running it, then open the <code>https://</code> address
              and accept the certificate warning once.</li>
            <li>Or install an IP-camera app on this phone and attach its stream
              URL to this code from the Devices page. No certificate needed.</li>
          </ol>
        </div>
      )}

      {phase === "denied" && (
        <p style={{ color: T.bad, fontSize: 14 }}>
          Camera permission was refused{detail && `: ${detail}`}. Allow it in the
          browser&apos;s site settings and reload.
        </p>
      )}

      {phase === "ready" && (
        <button onClick={() => void startStreaming()} style={{
          padding: ".9rem", background: T.accent, color: "#08131f", border: "none",
          borderRadius: 8, fontSize: 16, fontWeight: 600, cursor: "pointer" }}>
          Use this camera
        </button>
      )}

      {phase === "streaming" && (
        <div>
          <p style={{ color: T.ok, fontSize: 15, margin: 0 }}>
            Streaming · {frames} frames sent
          </p>
          <p style={{ color: T.faint, fontSize: 12, lineHeight: 1.6 }}>
            Keep this page open and the screen awake. {FPS} frames a second at{" "}
            {WIDTH}px — the rate the pipeline processes, so anything more would
            be discarded.
          </p>
          <button onClick={() => { stop(); setPhase("ready"); }} style={{
            marginTop: ".6rem", padding: ".6rem 1rem", background: "transparent",
            border: `1px solid ${T.line}`, color: T.dim, borderRadius: 6,
            fontSize: 13, cursor: "pointer" }}>Stop</button>
        </div>
      )}

      {phase === "lost" && (
        <div>
          <p style={{ color: T.bad, fontSize: 14 }}>
            Connection lost{detail && `: ${detail}`}. {frames} frames were sent.
          </p>
          <button onClick={() => void startStreaming()} style={{
            padding: ".7rem 1.1rem", background: T.accent, color: "#08131f",
            border: "none", borderRadius: 8, fontSize: 15, fontWeight: 600,
            cursor: "pointer" }}>Reconnect</button>
        </div>
      )}
    </div>
  );
}
