"use client";
/**
 * Vehicle search: type a plate, get every sighting and the stitched journey.
 *
 * A miss is a 200 with empty arrays — never a 404 — so "we searched and found
 * nothing" renders differently from "the request failed". Those are different
 * facts for an operator and must not look alike.
 *
 * The journey is drawn as a chain rather than a table because the answer to
 * "where did it go" is an ordering, and a table makes the reader reconstruct
 * one. Between two cameras sits the evidence for that hop — which layers
 * agreed, at what confidence, over how long — so the claim and its support are
 * never more than a centimetre apart.
 */
import { useCallback, useEffect, useState } from "react";
import { search as searchApi, getCameras, getLinks } from "@/lib/api";
import {
  Bar, Conf, Cols, Dot, Empty, Figure, HATCH, LABEL, META, MONO, Panel, Pips,
  Plate, SANS, T, ago, clock, hopTime, layerCount, methodColour, rowStyle,
} from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { SearchResult, Trajectory } from "@/contract";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: SearchResult }
  | { kind: "error"; message: string };

/** Longest first: the journey worth drawing is the one with the most hops. */
const longest = (ts: Trajectory[]) =>
  [...ts].sort((a, b) => b.hops.length - a.hops.length)[0] ?? null;

export default function SearchView() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });
  const [recent, setRecent] = useState<string[]>([]);
  const cameras = usePoll(getCameras, 60_000);
  const links = usePoll(getLinks, 300_000);
  const names = new Map((cameras ?? []).map((c) => [c.id, c.name]));
  const statusOf = new Map((cameras ?? []).map((c) => [c.id, c.status]));

  const run = useCallback(async (raw: string) => {
    const plate = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
    if (!plate) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    try {
      const result = await searchApi(plate);
      setState({ kind: "done", result });
      // Only plates that actually matched something go in the chip row. A row
      // of typos is worse than no row.
      if (result.sightings.length) {
        setRecent((r) => [plate, ...r.filter((p) => p !== plate)].slice(0, 6));
      }
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

  const r = state.kind === "done" ? state.result : null;
  const chain = r ? longest(r.trajectories) : null;

  // Distance is the road graph's, not a straight line. Layer 3 already knows
  // how far apart two cameras are; asking it again here keeps one answer.
  const linkDist = new Map((links ?? []).map((l) => [`${l.from}:${l.to}`, l.distance_m]));
  const metres = chain?.hops.reduce((m, h) =>
    m + (linkDist.get(`${h.from_camera}:${h.to_camera}`)
      ?? linkDist.get(`${h.to_camera}:${h.from_camera}`) ?? 0), 0) ?? 0;

  const spanSeconds = r?.sightings.length
    ? (Date.parse(r.sightings[r.sightings.length - 1]!.ts) - Date.parse(r.sightings[0]!.ts)) / 1000
    : 0;
  const seenCameras = new Set(r?.sightings.map((s) => s.camera_id) ?? []).size;
  const best = chain?.hops.reduce<typeof chain.hops[number] | null>(
    (b, h) => (!b || layerCount(h.method) > layerCount(b.method) ? h : b), null) ?? null;

  // Sightings arrive newest-first from the API; a journey reads earliest-first.
  const ledger = [...(r?.sightings ?? [])].sort((a, b) => (a.ts < b.ts ? -1 : 1));
  const nodes = chain
    ? [chain.hops[0]!.from_camera, ...chain.hops.map((h) => h.to_camera)]
    : [...new Set(ledger.map((s) => s.camera_id))];
  const firstAt = (cam: string) => ledger.find((s) => s.camera_id === cam) ?? null;
  const reidOnly = (r?.trajectories ?? []).flatMap((t) =>
    t.hops.filter((h) => h.method !== "plate").map((h) => ({ ...h, id: t.id })));

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div style={{ border: `1px solid ${T.line}`, background: T.panel, borderRadius: 3,
                    boxShadow: T.shadow }}>
        <div style={{ padding: "16px 16px 14px", display: "flex", alignItems: "flex-end",
                      gap: 14, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <label style={{ ...LABEL, display: "block", fontSize: 9.5 }}>Plate number</label>
            <input
              autoFocus value={query} spellCheck={false}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="OD02OJ9428"
              style={{
                marginTop: 7, width: "100%", padding: "11px 12px",
                border: `1px solid ${T.line2}`, background: T.raised, borderRadius: 2,
                font: `600 19px/1 ${MONO}`, letterSpacing: ".11em",
                textTransform: "uppercase", color: T.text, outline: "none",
              }}
            />
          </div>
          {recent.length > 0 && (
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap", paddingBottom: 2 }}>
              {recent.map((p) => (
                <button key={p} onClick={() => setQuery(p)} style={{
                  padding: "5px 9px", borderRadius: 2, font: `500 11px ${MONO}`,
                  letterSpacing: ".06em",
                  background: query.toUpperCase() === p ? T.text : "transparent",
                  color: query.toUpperCase() === p ? T.panel : T.dim,
                  border: `1px solid ${query.toUpperCase() === p ? T.text : T.line2}`,
                }}>{p}</button>
              ))}
            </div>
          )}
        </div>

        {r && (r.sightings.length > 0 || r.trajectories.length > 0) && (
          <div style={{ padding: "13px 16px 15px", display: "flex", gap: 26,
                        flexWrap: "wrap", borderTop: `1px solid ${T.line}` }}>
            <Figure label="Sightings" value={r.sightings.length} />
            <Figure label="Cameras" value={seenCameras} />
            <Figure label="Window" value={spanSeconds >= 60
              ? `${Math.round(spanSeconds / 60)}m` : `${Math.round(spanSeconds)}s`} />
            <Figure label="Distance" value={metres ? (metres / 1000).toFixed(1) : "—"}
                    unit={metres ? "km" : undefined} />
            <div style={{ flex: 1 }} />
            {best && (
              <div style={{ alignSelf: "flex-end", display: "flex", alignItems: "center",
                            gap: 9 }}>
                <div style={{ font: `400 11px ${SANS}`, color: T.dim, maxWidth: 320 }}>
                  {layerCount(best.method) === 3
                    ? "All three layers agree on the strongest hop."
                    : layerCount(best.method) === 2
                      ? "Two of three layers agree — the confirmation threshold."
                      : "Single-layer evidence only. Not a confirmed journey."}
                </div>
                <Pips method={best.method} rule={false} />
              </div>
            )}
          </div>
        )}
      </div>

      {state.kind === "loading" && <Panel title="Search"><Empty>Searching…</Empty></Panel>}

      {state.kind === "error" && (
        <Panel title="Search failed">
          <p style={{ color: T.bad, fontSize: 13, margin: 0 }}>{state.message}</p>
          <Empty>This is a request failure, not an empty result.</Empty>
        </Panel>
      )}

      {r && !r.sightings.length && !r.trajectories.length && (
        <Panel title={`Result · ${r.plate_text}`}>
          <Empty>
            Not seen. No camera has recorded this plate — the search succeeded
            and returned nothing.
          </Empty>
        </Panel>
      )}

      {r && nodes.length > 0 && (
        <Panel flush title="Stitched trajectory"
               right={<span style={META}>earliest first</span>}>
          <div style={{ overflowX: "auto", padding: "22px 16px 18px" }}>
            <div style={{ display: "flex", alignItems: "stretch", minWidth: 760 }}>
              {nodes.map((cam, i) => {
                const hop = i > 0 ? chain?.hops[i - 1] ?? null : null;
                const at = firstAt(cam);
                return (
                  <div key={`${cam}:${i}`} style={{ display: "flex", alignItems: "stretch" }}>
                    {hop && (
                      <div style={{ width: 150, flex: "0 0 150px", display: "flex",
                                    flexDirection: "column", justifyContent: "center",
                                    alignItems: "center", padding: "0 6px",
                                    position: "relative" }}>
                        <div style={{ position: "absolute", left: 0, right: 0, top: 37,
                                      height: 0,
                                      borderTop: layerCount(hop.method) === 3
                                        ? `2px solid ${T.text}`
                                        : `2px dashed ${T.line2}` }} />
                        <div style={{ position: "relative", background: T.panel,
                                      padding: "0 5px", marginTop: -2 }}>
                          <Pips method={hop.method} />
                        </div>
                        <div style={{ marginTop: 9, textAlign: "center" }}>
                          <div className="tnum" style={{ font: `500 11px ${MONO}` }}>
                            {hop.confidence.toFixed(2)}
                          </div>
                          <div className="tnum" style={{ marginTop: 4, ...META }}>
                            {hopTime(hop.travel_time_s)}
                          </div>
                        </div>
                      </div>
                    )}
                    <div style={{
                      width: 148, flex: "0 0 148px", padding: "12px 13px",
                      border: `1px solid ${T.line}`, background: T.panel, borderRadius: 3,
                      display: "flex", flexDirection: "column", alignItems: "flex-start",
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                        <Dot colour={statusOf.get(cam) === "online" ? T.ok : T.line2} />
                        <span style={{ font: `500 12px ${MONO}`, letterSpacing: ".04em" }}>
                          {cam}
                        </span>
                      </div>
                      <div style={{ marginTop: 5, font: `400 11px/1.35 ${SANS}`,
                                    color: T.dim, maxWidth: 118 }}>
                        {names.get(cam) ?? "unmapped camera"}
                      </div>
                      <div className="tnum" style={{ marginTop: 9, font: `500 15px ${MONO}` }}>
                        {at ? clock(at.ts) : "—"}
                      </div>
                      <div style={{ marginTop: 8, width: "100%", display: "flex",
                                    alignItems: "center", gap: 6 }}>
                        <Conf value={at?.confidence ?? null} width="100%"
                              missing="PLATE NOT READ" />
                      </div>
                      <div style={{ marginTop: 8, width: 56, height: 38, background: HATCH,
                                    border: `1px solid ${T.line}` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </Panel>
      )}

      {r && ledger.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 14,
                      alignItems: "start" }}>
          <Panel flush title="Sighting ledger" style={{ minWidth: 0 }}>
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: 560 }}>
                <Cols>
                  <div style={{ width: 96, flex: "0 0 96px" }}>Time</div>
                  <div style={{ width: 74, flex: "0 0 74px" }}>Camera</div>
                  <div style={{ flex: 1 }}>Location</div>
                  <div style={{ width: 126, flex: "0 0 126px" }}>Plate conf</div>
                  <div style={{ width: 90, flex: "0 0 90px", textAlign: "right" }}>Seen</div>
                </Cols>
                {ledger.map((s, i) => (
                  <div key={`${s.camera_id}:${s.ts}`} style={rowStyle(i === ledger.length - 1)}>
                    <div className="tnum" style={{ width: 96, flex: "0 0 96px",
                                                   font: `500 12px ${MONO}` }}>
                      {clock(s.ts)}
                    </div>
                    <div style={{ width: 74, flex: "0 0 74px", font: `400 11.5px ${MONO}`,
                                  color: T.dim }}>{s.camera_id}</div>
                    <div style={{ flex: 1, minWidth: 0, font: `400 12px ${SANS}`,
                                  overflow: "hidden", textOverflow: "ellipsis",
                                  whiteSpace: "nowrap" }}>
                      {names.get(s.camera_id) ?? "unmapped camera"}
                    </div>
                    <div style={{ width: 126, flex: "0 0 126px" }}>
                      {/* null confidence means the plate was never read on this
                          track — Re-ID put the vehicle here. Named, not blank. */}
                      <Conf value={s.confidence} missing="VIA RE-ID" />
                    </div>
                    <div className="tnum" style={{ width: 90, flex: "0 0 90px",
                                                   textAlign: "right", ...META }}>
                      {ago(s.ts)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Panel>

          <Panel flush style={{ minWidth: 0 }} title="Held without a plate read">
            <div style={{ padding: "11px 14px", borderBottom: `1px solid ${T.line}`,
                          font: `400 11.5px/1.5 ${SANS}`, color: T.dim }}>
              About a third of tracked vehicles never yield a plate at a given camera.
              These hops were carried on appearance and travel time instead, and they
              are the reason the journey above is unbroken.
            </div>
            {reidOnly.length ? reidOnly.map((h, i) => (
              <div key={`${h.id}:${i}`} style={{ ...rowStyle(i === reidOnly.length - 1),
                                                 alignItems: "flex-start" }}>
                <div style={{ width: 40, height: 32, flex: "0 0 40px", background: HATCH,
                              border: `1px solid ${T.line}` }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6,
                                flexWrap: "wrap" }}>
                    <Plate text={null} />
                    <span style={META}>{h.id}</span>
                  </div>
                  <div style={{ marginTop: 5, font: `400 11px ${SANS}`, color: T.dim }}>
                    {h.from_camera} → {h.to_camera} in {hopTime(h.travel_time_s)}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <Pips method={h.method} rule={false} />
                  <div className="tnum" style={{ marginTop: 5, font: `500 11px ${MONO}`,
                                                 color: T.dim }}>
                    {h.confidence.toFixed(2)}
                  </div>
                  <div style={{ marginTop: 4, display: "flex", justifyContent: "flex-end" }}>
                    <Bar value={h.confidence} width={44} colour={methodColour(h.method)} />
                  </div>
                </div>
              </div>
            )) : (
              <div style={{ padding: "12px 14px" }}>
                <Empty>Every hop for this plate was confirmed on the plate text itself.</Empty>
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
