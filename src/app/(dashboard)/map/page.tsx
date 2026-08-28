"use client";
/**
 * The map: camera positions, the road graph, and confirmed cross-camera
 * journeys as animated trips — with the hop ledger beside it.
 *
 * deck.gl WITHOUT a basemap library. A MapLibre style is fetched from a remote
 * host at load, so a flaky venue network turns the centrepiece view into a
 * blank page — and the demo is graded live. The OSM raster tiles below are a
 * bonus: TileLayer failing renders nothing and every deck.gl layer still draws.
 * The cameras, links and trajectories ARE the content.
 *
 * The ledger is the other half of the argument. An arc on a map says two
 * cameras saw the same vehicle; only the ledger says WHY the system believes
 * it, hop by hop, and that is the claim Module C has to defend.
 */
import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ArcLayer, PathLayer, ScatterplotLayer, TextLayer, BitmapLayer } from "@deck.gl/layers";
import { TileLayer, TripsLayer } from "@deck.gl/geo-layers";
import type { MapViewState, PickingInfo } from "@deck.gl/core";
import { getCameras, getLinks, getTrajectories } from "@/lib/api";
import {
  Conf, Empty, LABEL, META, MONO, Panel, Pips, Plate, SANS, T,
  clock, hopTime, methodColour,
} from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Camera, CameraLink, Trajectory } from "@/contract";

const INITIAL: MapViewState = {
  longitude: 77.607, latitude: 12.973, zoom: 13.4, pitch: 45, bearing: 0,
};

const RGB = (hex: string): [number, number, number] => [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16),
];

/** deck.gl wants literal channels, and the palette is CSS custom properties.
 *  Resolve them once against the live document so the map re-colours with the
 *  theme instead of freezing whichever palette was active at first paint. */
function useInk(): Record<string, [number, number, number]> {
  const [ink, setInk] = useState<Record<string, [number, number, number]>>({});
  useEffect(() => {
    const read = () => {
      const cs = getComputedStyle(document.documentElement);
      const of = (name: string): [number, number, number] => {
        const v = cs.getPropertyValue(name).trim();
        if (v.startsWith("#")) return RGB(v);
        const m = v.match(/\d+/g);
        return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [130, 130, 130];
      };
      setInk({ accent: of("--accent"), info: of("--info"), ok: of("--ok"),
               warn: of("--warn"), line2: of("--line2"), ink: of("--ink") });
    };
    read();
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);
  return ink;
}

export default function MapView() {
  const cameras = usePoll(getCameras, 15_000);
  const links = usePoll(getLinks, 60_000);
  const trajectories = usePoll(() => getTrajectories({ limit: 40 }), 10_000);
  const [time, setTime] = useState(0);
  const [selected, setSelected] = useState<Trajectory | null>(null);
  const [basemap, setBasemap] = useState(true);
  const ink = useInk();

  // Every trajectory replays on one shared clock, so journeys that really
  // overlapped in time overlap on screen.
  const span = useMemo(() => Math.max(
    60, ...(trajectories ?? []).map((t) => t.path.at(-1)?.[2] ?? 0),
  ), [trajectories]);

  useEffect(() => {
    let raf = 0;
    const tick = () => { setTime((t) => (t + span / 400) % span); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [span]);

  const byId = useMemo(
    () => new Map((cameras ?? []).map((c) => [c.id, c])), [cameras]);

  const all = trajectories ?? [];
  const shown = selected ? [selected] : all;
  const colourOf = (m: string): [number, number, number] =>
    (m.startsWith("plate") ? ink.accent : m.startsWith("reid") ? ink.info : ink.ok)
    ?? [130, 130, 130];

  const layers = [
    basemap && new TileLayer<ImageBitmap>({
      id: "basemap",
      data: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      minZoom: 0, maxZoom: 19, tileSize: 256, opacity: 0.35,
      // No error handler needed: a failed tile simply never renders, which is
      // exactly the offline behaviour we want.
      renderSubLayers: (props) => {
        const { boundingBox } = props.tile;
        // deck.gl's own recipe; its TileLayer/BitmapLayer prop types do not
        // line up in strict mode, and the cast is the documented shape.
        return new BitmapLayer({
          id: props.id,
          image: props.data as unknown as ImageBitmap,
          bounds: [boundingBox[0][0]!, boundingBox[0][1]!,
                   boundingBox[1][0]!, boundingBox[1][1]!] as [number, number, number, number],
          opacity: props.opacity,
        });
      },
    }),

    // The road graph. This IS layer 3 of the association engine, drawn.
    new PathLayer<CameraLink>({
      id: "links",
      data: (links ?? []).filter((l) => l.from < l.to),   // one line per pair
      getPath: (l) => {
        const a = byId.get(l.from), b = byId.get(l.to);
        return a && b ? [[a.lon, a.lat], [b.lon, b.lat]] : [];
      },
      getColor: [...(ink.line2 ?? [120, 120, 120]), 150] as [number, number, number, number],
      getWidth: 3, widthMinPixels: 1.5,
      pickable: true,
    }),

    new ArcLayer<Trajectory["hops"][number] & { t: Trajectory }>({
      id: "hops",
      data: shown.flatMap((t) => t.hops.map((h) => ({ ...h, t }))),
      getSourcePosition: (h) => {
        const c = byId.get(h.from_camera);
        return c ? [c.lon, c.lat] : [0, 0];
      },
      getTargetPosition: (h) => {
        const c = byId.get(h.to_camera);
        return c ? [c.lon, c.lat] : [0, 0];
      },
      // Colour by the LAYER that confirmed it. A plate match and a Re-ID match
      // are different claims with different confidence, and one colour hides
      // the distinction the whole project is built on.
      getSourceColor: (h) => colourOf(h.method),
      getTargetColor: (h) => colourOf(h.method),
      getWidth: (h) => 1 + h.confidence * 4,
      getHeight: 0.4,
      pickable: true,
      updateTriggers: { getSourceColor: [ink], getTargetColor: [ink] },
    }),

    new TripsLayer<Trajectory>({
      id: "trips",
      data: shown,
      getPath: (t) => t.path.map((p) => [p[0], p[1]]) as [number, number][],
      getTimestamps: (t) => t.path.map((p) => p[2]),
      getColor: (t) => (t.plate_text ? ink.accent : ink.info) ?? [130, 130, 130],
      widthMinPixels: 3,
      trailLength: Math.max(30, span * 0.25),
      currentTime: time,
      updateTriggers: { getColor: [ink] },
    }),

    new ScatterplotLayer<Camera>({
      id: "cameras",
      data: cameras ?? [],
      getPosition: (c) => [c.lon, c.lat],
      getRadius: 40, radiusMinPixels: 7, radiusMaxPixels: 22,
      getFillColor: (c) => (c.status === "online" ? ink.ok
                          : c.status === "degraded" ? ink.warn : ink.line2)
                          ?? [130, 130, 130],
      getLineColor: [10, 15, 20], lineWidthMinPixels: 2, stroked: true,
      pickable: true,
      updateTriggers: { getFillColor: [ink] },
    }),

    new TextLayer<Camera>({
      id: "camera-labels",
      data: cameras ?? [],
      getPosition: (c) => [c.lon, c.lat],
      getText: (c) => c.id,
      getSize: 12, getColor: ink.ink ?? [230, 237, 243],
      getPixelOffset: [0, -20],
      fontFamily: "ui-monospace, monospace",
      updateTriggers: { getColor: [ink] },
    }),
  ].filter(Boolean);

  const tooltip = ({ object }: PickingInfo) => {
    if (!object) return null;
    const o = object as Record<string, unknown>;
    if (typeof o.name === "string") return { text: `${o.id}\n${o.name}\n${o.status}` };
    if (typeof o.method === "string") {
      return { text: `${o.from_camera} to ${o.to_camera}\n${o.method} · ` +
                     `${o.confidence} · ${o.travel_time_s}s` };
    }
    if (typeof o.travel_time_s === "number") {
      return { text: `${o.from} to ${o.to}\n${o.distance_m} m · ${o.travel_time_s}s expected` };
    }
    return null;
  };

  const sel = selected ?? all[0] ?? null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 372px", gap: 14,
                  alignItems: "start" }}>
      <div style={{ minWidth: 0, border: `1px solid ${T.line}`, background: T.panel,
                    borderRadius: 3, overflow: "hidden", boxShadow: T.shadow }}>
        <div style={{ padding: "10px 12px", borderBottom: `1px solid ${T.line}`,
                      display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <div style={LABEL}>
            {selected ? "One journey" : `${all.length} journeys · road graph`}
          </div>
          <div style={{ flex: 1 }} />
          <button onClick={() => setBasemap(!basemap)} style={{
            border: `1px solid ${basemap ? T.text : T.line2}`,
            background: basemap ? T.text : "transparent",
            color: basemap ? T.panel : T.dim, borderRadius: 2, padding: "3px 8px",
            font: `500 9.5px ${MONO}`, letterSpacing: ".09em",
          }}>BASEMAP</button>
          {selected && (
            <button onClick={() => setSelected(null)} style={{
              border: `1px solid ${T.line2}`, background: "transparent", color: T.dim,
              borderRadius: 2, padding: "3px 8px", font: `500 9.5px ${MONO}`,
              letterSpacing: ".09em",
            }}>SHOW ALL</button>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 5, ...META }}>
            <span style={{ display: "inline-block", width: 20,
                           borderTop: `2.5px solid ${T.text}` }} />3 of 3 agree
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5, ...META }}>
            <span style={{ display: "inline-block", width: 20,
                           borderTop: `2.5px dashed ${T.faint}` }} />2 of 3
          </div>
        </div>
        <div style={{ position: "relative", height: "calc(100vh - 11rem)" }}>
          <DeckGL initialViewState={INITIAL} controller layers={layers}
                  getTooltip={tooltip}
                  onClick={(i) => {
                    const o = i.object as { t?: Trajectory } | null;
                    if (o?.t) setSelected(o.t);
                  }}
                  style={{ position: "absolute", top: "0", left: "0",
                           width: "100%", height: "100%" }} />
        </div>
      </div>

      <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        <Panel flush title="Trajectories"
               right={<span className="tnum" style={META}>{all.length} shown</span>}>
          <div style={{ maxHeight: 246, overflowY: "auto" }}>
            {all.length ? all.map((t) => {
              const on = sel?.id === t.id;
              return (
                <button key={t.id} onClick={() => setSelected(t)} style={{
                  display: "block", width: "100%", textAlign: "left",
                  background: on ? T.accentSoft : "transparent", border: "none",
                  borderBottom: `1px solid ${T.line}`, color: T.text,
                  padding: "var(--rowpad) 12px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ flex: 1, minWidth: 0 }}><Plate text={t.plate_text} /></div>
                    <div className="tnum" style={META}>{clock(t.started_at)}</div>
                  </div>
                  <div style={{ marginTop: 7, display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ font: `400 10.5px ${MONO}`, color: T.dim,
                                  letterSpacing: ".03em", overflow: "hidden",
                                  textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {[t.hops[0]?.from_camera, ...t.hops.map((h) => h.to_camera)]
                        .filter(Boolean).join(" → ") || `${t.path.length} points`}
                    </div>
                    <div style={{ flex: 1 }} />
                    <div className="tnum" style={{ ...META, width: 44, textAlign: "right" }}>
                      {t.hops.length} hop{t.hops.length === 1 ? "" : "s"}
                    </div>
                  </div>
                </button>
              );
            }) : (
              <div style={{ padding: "12px 14px" }}>
                <Empty>
                  No journeys yet. One appears when Module C confirms the same
                  vehicle at two cameras.
                </Empty>
              </div>
            )}
          </div>
        </Panel>

        <Panel flush title="Selected · hop ledger"
               right={sel ? <Plate text={sel.plate_text} /> : undefined}>
          {sel?.hops.length ? sel.hops.map((h, i, hs) => (
            <div key={i} style={{ padding: "var(--rowpad) 12px",
                                  borderBottom: i === hs.length - 1
                                    ? "none" : `1px solid ${T.line}` }}>
              <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                <div style={{ flex: 1, minWidth: 0, font: `400 11px ${MONO}`,
                              letterSpacing: ".03em" }}>
                  {h.from_camera} → {h.to_camera}
                </div>
                <div className="tnum" style={META}>{hopTime(h.travel_time_s)}</div>
              </div>
              <div style={{ marginTop: 7, display: "flex", alignItems: "center", gap: 10 }}>
                <Pips method={h.method} />
                <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center",
                              gap: 6 }}>
                  <Conf value={h.confidence} width="100%" colour={methodColour(h.method)} />
                  <span style={{ font: `400 9.5px ${MONO}`, letterSpacing: ".08em",
                                 color: T.faint, textTransform: "uppercase",
                                 whiteSpace: "nowrap" }}>
                    {h.method === "spatial_temporal" ? "spatio-temporal" : h.method}
                  </span>
                </div>
              </div>
              <div style={{ marginTop: 6, font: `400 10.5px/1.5 ${SANS}`, color: T.faint }}>
                {byName(h.from_camera, byId)} to {byName(h.to_camera, byId)}.
                {" "}Confirmed by {h.method === "spatial_temporal" ? "travel-time feasibility"
                  : h.method === "reid" ? "appearance similarity" : "plate text"} at{" "}
                {h.confidence.toFixed(2)}.
              </div>
            </div>
          )) : (
            <div style={{ padding: "12px 14px" }}>
              <Empty>
                {sel ? "Single camera — no cross-camera hop to explain."
                     : "Pick a trajectory to see why each hop was believed."}
              </Empty>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

const byName = (id: string, byId: Map<string, Camera>) => byId.get(id)?.name ?? id;
