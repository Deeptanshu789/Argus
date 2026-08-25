"use client";
/**
 * The map: camera positions, the road graph, and confirmed cross-camera
 * journeys as animated trips.
 *
 * deck.gl WITHOUT a basemap library. A MapLibre style is fetched from a remote
 * host at load, so a flaky venue network turns the centrepiece view into a
 * blank page — and the demo is graded live. The OSM raster tiles below are a
 * bonus: TileLayer failing renders nothing and every deck.gl layer still draws.
 * The cameras, links and trajectories ARE the content.
 */
import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ArcLayer, PathLayer, ScatterplotLayer, TextLayer, BitmapLayer } from "@deck.gl/layers";
import { TileLayer, TripsLayer } from "@deck.gl/geo-layers";
import type { MapViewState, PickingInfo } from "@deck.gl/core";
import { getCameras, getLinks, getTrajectories } from "@/lib/api";
import { Empty, Panel, T, Tag, methodColour, statusColour } from "@/components/ui";
import { usePoll } from "@/components/useLive";
import type { Camera, CameraLink, Trajectory } from "@/contract";

const INITIAL: MapViewState = {
  longitude: 77.607, latitude: 12.973, zoom: 13.4, pitch: 45, bearing: 0,
};

const RGB = (hex: string): [number, number, number] => [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16),
];

export default function MapView() {
  const cameras = usePoll(getCameras, 15_000);
  const links = usePoll(getLinks, 60_000);
  const trajectories = usePoll(() => getTrajectories({ limit: 40 }), 10_000);
  const [time, setTime] = useState(0);
  const [selected, setSelected] = useState<Trajectory | null>(null);
  const [basemap, setBasemap] = useState(true);

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

  const shown = selected ? [selected] : (trajectories ?? []);

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
      getColor: [80, 110, 140, 150],
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
      getSourceColor: (h) => RGB(methodColour(h.method)),
      getTargetColor: (h) => RGB(methodColour(h.method)),
      getWidth: (h) => 1 + h.confidence * 4,
      getHeight: 0.4,
      pickable: true,
    }),

    new TripsLayer<Trajectory>({
      id: "trips",
      data: shown,
      getPath: (t) => t.path.map((p) => [p[0], p[1]]) as [number, number][],
      getTimestamps: (t) => t.path.map((p) => p[2]),
      getColor: (t) => (t.plate_text ? RGB(T.accent) : RGB(T.reid)),
      widthMinPixels: 3,
      trailLength: Math.max(30, span * 0.25),
      currentTime: time,
    }),

    new ScatterplotLayer<Camera>({
      id: "cameras",
      data: cameras ?? [],
      getPosition: (c) => [c.lon, c.lat],
      getRadius: 40, radiusMinPixels: 7, radiusMaxPixels: 22,
      getFillColor: (c) => RGB(statusColour(c.status)),
      getLineColor: [10, 15, 20], lineWidthMinPixels: 2, stroked: true,
      pickable: true,
    }),

    new TextLayer<Camera>({
      id: "camera-labels",
      data: cameras ?? [],
      getPosition: (c) => [c.lon, c.lat],
      getText: (c) => c.id,
      getSize: 12, getColor: [230, 237, 243],
      getPixelOffset: [0, -20],
      fontFamily: "ui-monospace, monospace",
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

  return (
    <div style={{ display: "grid", gap: "1rem",
                  gridTemplateColumns: "minmax(0, 1fr) minmax(260px, 320px)" }}>
      <Panel style={{ padding: 0, overflow: "hidden", height: "calc(100vh - 8rem)",
                      position: "relative" }}>
        <DeckGL initialViewState={INITIAL} controller layers={layers}
                getTooltip={tooltip}
                onClick={(i) => {
                  const o = i.object as { t?: Trajectory } | null;
                  if (o?.t) setSelected(o.t);
                }}
                style={{ position: "absolute", top: "0", left: "0",
                         width: "100%", height: "100%" }} />
        <div style={{ position: "absolute", left: "12px", bottom: "12px", display: "flex",
                      gap: ".5rem", alignItems: "center", background: `${T.panel}dd`,
                      padding: ".4rem .7rem", borderRadius: 6, fontSize: 11,
                      border: `1px solid ${T.line}` }}>
          <Tag colour={T.accent}>plate</Tag>
          <Tag colour={T.reid}>reid</Tag>
          <Tag colour={T.ok}>spatial_temporal</Tag>
          <label style={{ color: T.dim, marginLeft: ".5rem", cursor: "pointer" }}>
            <input type="checkbox" checked={basemap}
                   onChange={(e) => setBasemap(e.target.checked)} /> basemap
          </label>
        </div>
      </Panel>

      <div style={{ display: "grid", gap: "1rem", alignContent: "start" }}>
        <Panel title={selected ? "Selected journey" : `Journeys — ${(trajectories ?? []).length}`}
               right={selected ? (
                 <button onClick={() => setSelected(null)} style={{
                   background: "transparent", border: `1px solid ${T.line}`,
                   color: T.dim, borderRadius: 4, fontSize: 11,
                   padding: "2px 7px", cursor: "pointer" }}>show all</button>
               ) : undefined}>
          <div style={{ maxHeight: "calc(100vh - 14rem)", overflowY: "auto" }}>
            {(trajectories ?? []).length ? (trajectories ?? []).map((t) => (
              <button key={t.id} onClick={() => setSelected(t)} style={{
                display: "block", width: "100%", textAlign: "left", cursor: "pointer",
                background: selected?.id === t.id ? T.raised : "transparent",
                border: "none", borderBottom: `1px solid ${T.line}`,
                color: T.text, padding: ".5rem .3rem",
              }}>
                <div style={{ fontSize: 13 }}>
                  {t.plate_text ?? <span style={{ color: T.reid }}>unidentified</span>}
                </div>
                <div style={{ fontSize: 11, color: T.dim, marginTop: 2 }}>
                  {t.path.length} cameras · {t.hops.length} hop
                  {t.hops.length === 1 ? "" : "s"}
                </div>
                <div style={{ display: "flex", gap: ".3rem", marginTop: 4, flexWrap: "wrap" }}>
                  {t.hops.map((h, i) => (
                    <Tag key={i} colour={methodColour(h.method)}>{h.method}</Tag>
                  ))}
                </div>
              </button>
            )) : (
              <Empty>
                No journeys yet. One appears when Module C confirms the same
                vehicle at two cameras.
              </Empty>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
