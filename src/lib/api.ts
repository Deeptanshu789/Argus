/**
 * The ONLY place API paths are constructed. Hardcode a path anywhere else and
 * the mock/live switch stops working.
 *
 * Switching the whole app from fixtures to live data is one env var:
 * NEXT_PUBLIC_MOCK=1 in .env.development, absent in production.
 */
import type {
  Alert, AnalyticsResponse, Camera, CameraLink, SearchResult, ServerMessage, Track, Trajectory,
} from "@/contract";

const BASE = process.env.NEXT_PUBLIC_MOCK === "1" ? "/api/mock" : "/api";

async function get<T>(path: string, params: Record<string, string | number | undefined> = {}) {
  const q = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]),
  );
  const res = await fetch(`${BASE}${path}${q.size ? `?${q}` : ""}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const getCameras = () => get<Camera[]>("/cameras");
export const getLinks = () => get<CameraLink[]>("/cameras/links");
export const getTracks = (p: { camera?: string; since?: string; limit?: number } = {}) =>
  get<Track[]>("/tracks", p);
export const getTrajectories = (p: { since?: string; limit?: number } = {}) =>
  get<Trajectory[]>("/trajectories", p);
export const search = (plate: string) => get<SearchResult>("/search", { plate });
export const getAnalytics = (camera?: string) => get<AnalyticsResponse>("/analytics", { camera });
export const getAlerts = (acked?: boolean) =>
  get<Alert[]>("/alerts", { acked: acked === undefined ? undefined : String(acked) });
export const ackAlert = (id: string) =>
  fetch(`${BASE}/alerts/${id}/ack`, { method: "POST" }).then((r) => r.json());

/**
 * WebSocket with reconnect. The demo must survive a dropped socket without a
 * page reload. Unknown message types are ignored by the caller's switch — that
 * is what lets the server add types without breaking a frontend built earlier.
 */
export function connect(onMessage: (m: ServerMessage) => void): () => void {
  let ws: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  const open = () => {
    if (closed) return;
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data as string) as ServerMessage); } catch { /* ignore junk */ }
    };
    ws.onclose = () => { if (!closed) timer = setTimeout(open, 1000); };
    ws.onerror = () => ws?.close();
  };
  open();

  return () => { closed = true; if (timer) clearTimeout(timer); ws?.close(); };
}
