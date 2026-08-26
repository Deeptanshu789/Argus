/**
 * In-memory relay from a phone's browser to the inference sidecar.
 *
 * A browser cannot serve RTSP, and the sidecar cannot receive a WebSocket. This
 * bridges them: frames arrive here as JPEG over a WebSocket and leave as an
 * MJPEG HTTP stream, which OpenCV opens with no Python change at all — the
 * sidecar already treats an http:// source as a live stream.
 *
 * LATEST FRAME WINS. There is no queue, deliberately. A phone on a congested
 * network that falls behind should show the sidecar the newest frame, not work
 * through a backlog of stale ones: this is surveillance, where an old frame has
 * no value. That also bounds memory at one frame per device no matter what the
 * network does.
 *
 * ponytail: process-local. Two server processes would each see only their own
 * devices, so if this is ever run behind a load balancer the frames have to
 * move to Redis or the device has to be pinned to one instance.
 */

/** Frames older than this mean the phone went away — the tab was closed, the
 *  screen locked, the network dropped. The sidecar sees the stream end. */
export const STALE_MS = 10_000;

interface Feed {
  latest: Buffer | null;
  at: number;
  /** MJPEG responses waiting for the next frame. */
  waiters: Set<(frame: Buffer) => void>;
  frames: number;
}

const feeds = new Map<string, Feed>();

const feed = (cameraId: string): Feed => {
  let f = feeds.get(cameraId);
  if (!f) {
    f = { latest: null, at: 0, waiters: new Set(), frames: 0 };
    feeds.set(cameraId, f);
  }
  return f;
};

export function putFrame(cameraId: string, jpeg: Buffer): void {
  const f = feed(cameraId);
  f.latest = jpeg;
  f.at = Date.now();
  f.frames++;
  // Hand it to everyone waiting, then clear: each waiter re-subscribes for the
  // next one. Copying the set first because a waiter removes itself when called.
  const waiting = [...f.waiters];
  f.waiters.clear();
  for (const w of waiting) w(jpeg);
}

/** Wait for the next frame, or resolve null if none arrives before the timeout. */
export function nextFrame(cameraId: string, timeoutMs = STALE_MS): Promise<Buffer | null> {
  const f = feed(cameraId);
  return new Promise((resolve) => {
    let done = false;
    const give = (frame: Buffer | null) => {
      if (done) return;
      done = true;
      f.waiters.delete(give as (b: Buffer) => void);
      clearTimeout(timer);
      resolve(frame);
    };
    const timer = setTimeout(() => give(null), timeoutMs);
    f.waiters.add(give as (b: Buffer) => void);
  });
}

export function feedStatus(cameraId: string): { live: boolean; frames: number; ageMs: number } {
  const f = feeds.get(cameraId);
  if (!f || !f.at) return { live: false, frames: 0, ageMs: Infinity };
  const ageMs = Date.now() - f.at;
  return { live: ageMs < STALE_MS, frames: f.frames, ageMs };
}

export function dropFeed(cameraId: string): void {
  const f = feeds.get(cameraId);
  if (!f) return;
  for (const w of f.waiters) w(Buffer.alloc(0));
  feeds.delete(cameraId);
}

export const liveFeeds = (): string[] =>
  [...feeds.entries()].filter(([, f]) => Date.now() - f.at < STALE_MS).map(([id]) => id);
