/**
 * Redis pub/sub between the ingest worker and the web server.
 *
 * They are separate processes on purpose — a CPU-pinned decode loop in the
 * server would stall the dashboard — so the worker cannot call broadcast()
 * directly. One channel carries every ServerMessage; the server relays each one
 * to its WebSocket clients unchanged.
 *
 * ponytail: fire-and-forget pub/sub, no persistence. A client that is offline
 * during an event misses it and refetches over REST on reconnect, which is
 * exactly what a live map needs. Use a Redis Stream instead only if replay of
 * missed events ever becomes a requirement.
 */
import Redis from "ioredis";
import { ServerMessage } from "@/contract";

export const REDIS_URL = process.env.REDIS_URL ?? "redis://localhost:6379/0";
export const CHANNEL = "argus:events";

/** Publishing must never take the process down: Redis is a nice-to-have for
 *  live updates, while the database is the system of record. */
function quiet(r: Redis, label: string): Redis {
  let warned = false;
  r.on("error", (e) => {
    if (warned) return;
    warned = true;
    console.warn(`redis ${label} unavailable (${e.message}); live updates are off`);
  });
  return r;
}

export function publisher(): Redis {
  return quiet(new Redis(REDIS_URL, { maxRetriesPerRequest: null, lazyConnect: false }), "publisher");
}

export async function publish(pub: Redis, msg: ServerMessage): Promise<void> {
  try {
    await pub.publish(CHANNEL, JSON.stringify(msg));
  } catch {
    /* already logged once by quiet() */
  }
}

/**
 * Subscribe and hand each valid message to `onMessage`.
 *
 * Messages are validated against the contract on the way IN, not just on the
 * way out: a malformed publish would otherwise reach every connected browser
 * and break the dashboard's switch, with the bug looking like a frontend fault.
 */
export function subscribe(onMessage: (m: ServerMessage) => void): Redis {
  const sub = quiet(new Redis(REDIS_URL, { maxRetriesPerRequest: null }), "subscriber");
  void sub.subscribe(CHANNEL).catch(() => {});
  sub.on("message", (_channel, payload) => {
    let raw: unknown;
    try {
      raw = JSON.parse(payload);
    } catch {
      return;
    }
    const parsed = ServerMessage.safeParse(raw);
    if (!parsed.success) {
      console.warn("dropped malformed bus message:", parsed.error.issues[0]?.message);
      return;
    }
    onMessage(parsed.data);
  });
  return sub;
}
