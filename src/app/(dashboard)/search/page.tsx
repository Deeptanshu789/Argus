"use client";
/**
 * Vehicle search: type a plate, get every sighting and every stitched journey.
 *
 * A miss is a 200 with empty arrays — never a 404 — so "we searched and found
 * nothing" renders differently from "the request failed". Those are different
 * facts for an operator and must not look alike.
 */
import { useCallback, useEffect, useState } from "react";
import { search as searchApi, getCameras } from "@/lib/api";
import { Empty, Panel, T, Tag, ago, methodColour } from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { SearchResult } from "@/contract";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: SearchResult }
  | { kind: "error"; message: string };

export default function SearchView() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });
  const cameras = usePoll(getCameras, 60_000);
  const names = new Map((cameras ?? []).map((c) => [c.id, c.name]));

  const run = useCallback(async (raw: string) => {
    const plate = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
    if (!plate) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    try {
      setState({ kind: "done", result: await searchApi(plate) });
    } catch (e) {
      setState({ kind: "error", message: (e as Error).message });
    }
  }, []);

  // Debounce. A search per keystroke is a query per keystroke, and the operator
  // is typing ten characters.
  useEffect(() => {
    const t = setTimeout(() => void run(query), 350);
    return () => clearTimeout(t);
  }, [query, run]);

  return (
    <div style={{ display: "grid", gap: "1rem", maxWidth: 900, margin: "0 auto" }}>
      <Panel title="Vehicle search">
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="KA05MR7821"
          spellCheck={false}
          style={{
            width: "100%", padding: ".75rem 1rem", fontSize: 20, letterSpacing: ".12em",
            fontFamily: "ui-monospace, monospace", textTransform: "uppercase",
            background: T.raised, border: `1px solid ${T.line}`, borderRadius: 8,
            color: T.text, outline: "none",
          }}
        />
        <p style={{ fontSize: 11, color: T.faint, margin: ".5rem 0 0" }}>
          Spaces and dashes are ignored. Plates are stored as read by OCR after
          positional correction, so search matches the corrected form.
        </p>
      </Panel>

      {state.kind === "loading" && <Panel><Empty>Searching…</Empty></Panel>}

      {state.kind === "error" && (
        <Panel title="Search failed">
          <p style={{ color: T.bad, fontSize: 13, margin: 0 }}>{state.message}</p>
          <Empty>This is a request failure, not an empty result.</Empty>
        </Panel>
      )}

      {state.kind === "done" && (() => {
        const r = state.result;
        const nothing = !r.sightings.length && !r.trajectories.length;
        return (
          <>
            <Panel title={`Result — ${r.plate_text}`}>
              {nothing ? (
                <Empty>
                  Not seen. No camera has recorded this plate — the search
                  succeeded and returned nothing.
                </Empty>
              ) : (
                <div style={{ display: "flex", gap: "2rem", fontSize: 13, flexWrap: "wrap" }}>
                  <div>
                    <div style={{ color: T.dim, fontSize: 11 }}>SIGHTINGS</div>
                    <strong style={{ fontSize: 22 }}>{r.sightings.length}</strong>
                  </div>
                  <div>
                    <div style={{ color: T.dim, fontSize: 11 }}>JOURNEYS</div>
                    <strong style={{ fontSize: 22 }}>{r.trajectories.length}</strong>
                  </div>
                  {r.last_seen && (
                    <div>
                      <div style={{ color: T.dim, fontSize: 11 }}>LAST SEEN</div>
                      <strong style={{ fontSize: 15 }}>
                        {r.last_seen.camera_id} · {ago(r.last_seen.ts)}
                      </strong>
                    </div>
                  )}
                </div>
              )}
            </Panel>

            {r.trajectories.length > 0 && (
              <Panel title="Journeys">
                {r.trajectories.map((t) => (
                  <div key={t.id} style={{ padding: ".6rem 0",
                                           borderBottom: `1px solid ${T.line}` }}>
                    <div style={{ display: "flex", alignItems: "center",
                                  gap: ".5rem", flexWrap: "wrap" }}>
                      <strong style={{ fontSize: 13 }}>{t.vehicle_type}</strong>
                      <span style={{ fontSize: 11, color: T.dim }}>
                        {ago(t.started_at)} · {t.path.length} cameras
                      </span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: ".4rem",
                                  marginTop: ".4rem", flexWrap: "wrap" }}>
                      {t.hops.length ? t.hops.map((h, i) => (
                        <span key={i} style={{ display: "flex", alignItems: "center",
                                               gap: ".35rem", fontSize: 12 }}>
                          {i === 0 && <code>{h.from_camera}</code>}
                          <Tag colour={methodColour(h.method)}>
                            {h.method} {h.confidence.toFixed(2)} · {h.travel_time_s}s
                          </Tag>
                          <code>{h.to_camera}</code>
                        </span>
                      )) : <Empty>Single camera — no cross-camera hop.</Empty>}
                    </div>
                  </div>
                ))}
              </Panel>
            )}

            {r.sightings.length > 0 && (
              <Panel title="Sightings">
                <div style={{ maxHeight: 340, overflowY: "auto" }}>
                  {r.sightings.map((s, i) => (
                    <div key={i} style={{ display: "flex", gap: ".75rem", fontSize: 12,
                                          padding: ".35rem 0",
                                          borderBottom: `1px solid ${T.line}` }}>
                      <code style={{ color: T.accent, width: 56 }}>{s.camera_id}</code>
                      <span style={{ flex: 1, color: T.dim }}>
                        {names.get(s.camera_id) ?? ""}
                      </span>
                      <span style={{ color: T.dim }}>
                        {/* null confidence means the plate was never OCR'd on
                            this track — Re-ID put the vehicle here. Blank, not 0. */}
                        {s.confidence === null ? "via Re-ID" : s.confidence.toFixed(2)}
                      </span>
                      <span style={{ color: T.faint, width: 74, textAlign: "right" }}>
                        {ago(s.ts)}
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </>
        );
      })()}
    </div>
  );
}
