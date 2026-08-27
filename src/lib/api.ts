/**
 * The ONLY place API paths are constructed. Hardcode a path anywhere else and
 * the mock/live switch stops working.
 *
 * Switching the whole app from fixtures to live data is one env var:
 * NEXT_PUBLIC_MOCK=1 in .env.development, absent in production.
 */
import type {
  Alert, AnalyticsResponse, Camera, CameraLink, SearchResult, ServerMessage, Track, Trajectory,
  Device, Upload, UploadResult,
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
export const getAlerts = (acked?: boolean, limit?: number) =>
  get<Alert[]>("/alerts", {
    acked: acked === undefined ? undefined : String(acked),
    limit: limit === undefined ? undefined : String(limit),
  });
export const getDevices = () => get<Device[]>("/devices");

export const getDeviceByCode = (code: string) =>
  fetch(`${BASE}/devices/${encodeURIComponent(code)}`).then(async (r) => {
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(`device lookup failed (${r.status})`);
    return (await r.json()) as Device;
  });

const postJson = async <T>(path: string, body: unknown): Promise<T> => {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { error?: string }).error ?? `failed (${r.status})`);
  return data as T;
};

export const createDevice = (label: string | null) =>
  postJson<Device>("/devices", { label });

export const pairDeviceUrl = (code: string, url: string) =>
  postJson<Device>(`/devices/${encodeURIComponent(code)}/url`, { url });

export const revokeDevice = (id: string) =>
  postJson<{ ok: true }>(`/devices/${encodeURIComponent(id)}/revoke`, {});

export const getUploads = (limit?: number) =>
  get<Upload[]>("/uploads", { limit: limit === undefined ? undefined : String(limit) });

export const getUpload = (id: string) => get<UploadResult>(`/uploads/${id}`);

/** Stop scanning an upload. Whatever was already read is kept. */
export const cancelUpload = (id: string) =>
  postJson<Upload>(`/uploads/${encodeURIComponent(id)}/cancel`, {});

/**
 * Multipart, not JSON: a video is megabytes of binary and base64 would inflate
 * it by a third for no benefit. The browser sets its own Content-Type boundary,
 * so this must NOT set one.
 */
export const createUpload = (files: File[], gapSeconds: number | null, label: string | null) => {
  const body = new FormData();
  for (const f of files) body.append("files", f);
  if (gapSeconds !== null) body.append("gap_seconds", String(gapSeconds));
  if (label) body.append("label", label);
  return fetch(`${BASE}/uploads`, { method: "POST", body }).then(async (r) => {
    const data = await r.json();
    if (!r.ok) throw new Error((data as { error?: string }).error ?? `upload failed (${r.status})`);
    return data as Upload;
  });
};

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
