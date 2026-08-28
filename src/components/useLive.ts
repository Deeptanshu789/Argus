"use client";
/**
 * The WebSocket, as hooks. Every dashboard view subscribes through here.
 *
 * The switch IGNORES unknown message types on purpose — that rule is what lets
 * the server add a message type without breaking a dashboard built earlier, and
 * it is the reason Dev B could build this before the pipeline emitted anything.
 */
import { useEffect, useRef, useState } from "react";
import { connect } from "@/lib/api";
import type { Alert, Hop, ServerMessage } from "@/contract";

export interface LiveDetection {
  camera_id: string;
  track_id: string;
  bbox: [number, number, number, number];
  vehicle_type: string;
  /** null when OCR read nothing on this track yet. The wall draws a read box
   *  differently from an unread one, which is the whole point of carrying it. */
  plate_text: string | null;
  conf: number;
  at: number;
}

export interface LiveState {
  connected: boolean;
  /** Newest first, capped. An uncapped array grows without limit over a long
   *  demo and the page eventually stutters — the cap IS the fix. */
  detections: LiveDetection[];
  matches: (Hop & { trajectory_id: string; plate_text: string | null; at: number })[];
  alerts: Alert[];
  perCamera: Record<string, { vehicle_count: number; congestion_score: number }>;
  city: { vehicle_count: number; avg_speed_kmh: number } | null;
  /** Detections seen per camera since mount — what makes a tile look alive. */
  counts: Record<string, number>;
}

const CAP = 60;

export function useLive(): LiveState {
  const [state, setState] = useState<LiveState>({
    connected: false, detections: [], matches: [], alerts: [],
    perCamera: {}, city: null, counts: {},
  });

  useEffect(() => {
    // Batch through a ref and flush on a timer. A detection arrives 20x a
    // second across four cameras; setState per message re-renders the whole
    // tree 20x a second and the map drops frames.
    const pending: ServerMessage[] = [];
    const disconnect = connect((m) => pending.push(m));

    const flush = setInterval(() => {
      if (!pending.length) return;
      const batch = pending.splice(0, pending.length);
      setState((s) => {
        const next: LiveState = { ...s, connected: true };
        for (const m of batch) {
          switch (m.type) {
            case "detection": {
              const d = { ...m.data, at: Date.now() };
              next.detections = [d, ...next.detections].slice(0, CAP);
              next.counts = {
                ...next.counts,
                [d.camera_id]: (next.counts[d.camera_id] ?? 0) + 1,
              };
              break;
            }
            case "match":
              next.matches = [{ ...m.data, at: Date.now() }, ...next.matches].slice(0, CAP);
              break;
            case "alert":
              // Same alert can arrive twice if the worker restarts mid-demo.
              next.alerts = [m.data, ...next.alerts.filter((a) => a.id !== m.data.id)]
                .slice(0, CAP);
              break;
            case "analytics":
              next.perCamera = m.data.per_camera;
              next.city = m.data.city;
              break;
            case "trajectory_update":
              break;   // the map refetches; nothing to hold here
            // NO default that throws. An unknown type is forward compatibility,
            // not an error.
          }
        }
        return next;
      });
    }, 250);

    return () => { clearInterval(flush); disconnect(); };
  }, []);

  return state;
}

/** Poll a REST endpoint. The WebSocket carries events; this carries state. */
export function usePoll<T>(fetcher: () => Promise<T>, ms = 10_000): T | null {
  const [data, setData] = useState<T | null>(null);
  const fn = useRef(fetcher);
  fn.current = fetcher;

  useEffect(() => {
    let alive = true;
    const tick = () => {
      fn.current()
        .then((d) => { if (alive) setData(d); })
        // A failed poll must not blank the view. Keeping the last good data on
        // screen through a blip is the difference between "the demo froze" and
        // "the demo is fine".
        .catch(() => {});
    };
    tick();
    const t = setInterval(tick, ms);
    return () => { alive = false; clearInterval(t); };
  }, [ms]);

  return data;
}
