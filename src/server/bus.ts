/**
 * Pub/sub between the ingest worker and the web server, over Postgres
 * LISTEN/NOTIFY.
 *
 * They are separate processes on purpose — a CPU-pinned decode loop inside the
 * server would stall the dashboard — so the worker cannot call broadcast()
 * directly. One channel carries every ServerMessage; the server relays each one
 * to its WebSocket clients unchanged.
 *
 * Postgres rather than Redis because Postgres is already here, already
 * connected, and already the system of record. A second daemon whose only job
 * is to move a few hundred bytes a second between two local processes is a
 * second thing to install, supervise and explain.
 *
 * ponytail: fire-and-forget, no persistence and no replay. A client that is
 * offline during an event misses it and refetches over REST on reconnect,
 * which is what a live map wants anyway. NOTIFY also caps a payload at 8000
 * bytes — see MAX_PAYLOAD. If replay or large payloads ever become
 * requirements, this is the file to replace, and its three exported functions
 * are the whole interface.
 */
import type postgres from "postgres";
import { ServerMessage } from "@/contract";
import { sql } from "@/server/db";

/**
 * Unquoted-identifier safe. LISTEN takes an identifier, not a string, so a
 * colon in the name would have to be quoted at every call site.
 */
export const CHANNEL = "argus_events";

/**
 * Postgres refuses a NOTIFY payload over 8000 bytes with an error, which would
 * surface as a failed handler rather than a missing dashboard tick. Ours run a
 * few hundred bytes; the cap is here so an unusually long trajectory degrades
 * to "the client refetches" instead of "the worker logs an error".
 */
const MAX_PAYLOAD = 7500;

export type Bus = postgres.Sql;

export function publisher(): Bus {
  return sql;
}

/**
 * Publishing must never take the process down. Live updates are a convenience;
 * the database write that preceded them is the thing that mattered.
 */
export async function publish(bus: Bus, msg: ServerMessage): Promise<void> {
  const payload = JSON.stringify(msg);
  if (payload.length > MAX_PAYLOAD) {
    console.warn(
      `bus: dropped a ${payload.length}-byte ${msg.type} (limit ${MAX_PAYLOAD}); ` +
      `clients will pick it up on their next poll`);
    return;
  }
  try {
    await bus.notify(CHANNEL, payload);
  } catch (e) {
    console.warn(`bus publish failed (${(e as Error).message}); live updates are off`);
  }
}

/**
 * Subscribe and hand each valid message to `onMessage`.
 *
 * Messages are validated against the contract on the way IN, not just on the
 * way out: a malformed publish would otherwise reach every connected browser
 * and break the dashboard's switch, with the bug looking like a frontend fault.
 *
 * postgres.js reconnects a dropped listener and re-issues LISTEN itself, so a
 * database restart costs the missed events and nothing else.
 */
export function subscribe(onMessage: (m: ServerMessage) => void): void {
  void sql.listen(CHANNEL, (payload) => {
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
  }).catch((e: Error) => {
    console.warn(`bus subscribe failed (${e.message}); live updates are off`);
  });
}
