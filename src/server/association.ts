/**
 * Module C — cross-camera association. THE DIFFERENTIATOR.
 *
 * Decides whether a vehicle that left one camera is the same vehicle that
 * arrived at another. Any team can ship single-camera ANPR; this is what wins.
 *
 * Three layers, and a match is confirmed only when TWO OF THREE agree:
 *
 *   1 HARD  — exact plate text, both sides confident.       ~60-70% of cases
 *   2 SOFT  — cosine similarity of 512-dim OSNet embeddings. ~20-25% (occluded
 *             or unreadable plates), with colour histogram as a second signal
 *   3 PHYS  — road-graph travel time feasibility. Rejects the impossible and
 *             disambiguates layer 2
 *
 * The 2-of-3 rule is the whole defence against false trajectories, which are
 * the most damaging thing a judge can see. A plate match ALONE is not enough.
 *
 * Layer 3 is special: when it has PROVED impossibility (a known route the
 * vehicle could not have covered in the observed time, or an arrival before the
 * exit) it VETOES, rather than being outvoted by the other two. OCR misreads
 * and cloned plates are real, and two white sedans look alike; a car covering
 * 1.4 km in 30 seconds is not real. When the camera graph simply has no route
 * between the two cameras, layer 3 abstains instead — that is ignorance, not
 * proof, and vetoing on it would lose genuine matches to missing topology.
 *
 * Pure functions over plain objects. No database, no ML library — this is
 * business logic, which is why it lives on the TypeScript side.
 */
import type { CameraLink, Hop, MatchMethod } from "@/contract";

export interface TrackRecord {
  id: string;
  camera_id: string;
  track_id: string;
  plate_text: string | null;
  plate_conf: number | null;
  color_hist: number[];
  /** 512-dim OSNet vector. Computed on track exit only — see CLAUDE.md. */
  embedding: number[];
  entry_time: string;
  exit_time: string | null;
}

export interface AssociationConfig {
  /** Cosine similarity above which layer 2 fires. */
  reidThreshold: number;
  /** Both plates must be at least this confident for layer 1 to fire. */
  plateConfMin: number;
  /** Layer 3 accepts actual travel time in [lo, hi] x expected. */
  timeWindowLo: number;
  timeWindowHi: number;
  /** Layers that must agree. Lower this and you start showing false routes. */
  minLayers: number;
}

/**
 * ponytail: tuned by hand on demo footage, not learned. These are the knobs
 * worth touching first if matching is too eager or too shy — see WORKFLOW.md
 * Stage 5. Do not lower minLayers to "get more matches"; that is how you end up
 * demoing a wrong trajectory.
 */
export const DEFAULTS: AssociationConfig = {
  reidThreshold: 0.75,
  plateConfMin: 0.8,
  timeWindowLo: 0.5,
  timeWindowHi: 2.0,
  minLayers: 2,
};

export interface LayerVerdict {
  passed: boolean;
  score: number;
  detail: string;
}

interface TemporalVerdict extends LayerVerdict {
  actual: number;
  expected: number | null;
  /**
   * Physics says this pair CANNOT be the same vehicle — not merely that the
   * layer abstained. Distinct from `!passed`, because "no route in the graph"
   * is ignorance, while "arrived before it left" is proof.
   */
  impossible: boolean;
}

export interface MatchResult {
  from: TrackRecord;
  to: TrackRecord;
  method: MatchMethod;
  confidence: number;
  travel_time_s: number;
  layers: { plate: LayerVerdict; reid: LayerVerdict; temporal: LayerVerdict };
  /** Which layers agreed, for the alert banner and for debugging a bad match. */
  agreed: MatchMethod[];
}

// ------------------------------------------------------------- similarity --

export function cosine(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!, y = b[i]!;
    dot += x * y; na += x * x; nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  // Clamp: floating point can push a self-comparison a hair over 1.0.
  return Math.max(-1, Math.min(1, dot / Math.sqrt(na * nb)));
}

/**
 * Histogram intersection, normalized. Preferred over cosine for colour because
 * it is bin-wise and insensitive to overall brightness — two photos of the same
 * white car under different exposure still intersect well.
 */
export function histogramSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let inter = 0, total = 0;
  for (let i = 0; i < a.length; i++) {
    inter += Math.min(a[i]!, b[i]!);
    total += a[i]!;
  }
  return total === 0 ? 0 : Math.max(0, Math.min(1, inter / total));
}

// ------------------------------------------------------------- road graph --

/**
 * Shortest travel time between two cameras over the link graph.
 *
 * Not just a direct-link lookup: with four cameras and six links, CAM2 and CAM3
 * have no direct edge, and a direct-only check would make every CAM2->CAM3
 * match physically "impossible" and silently unmatchable. Dijkstra over a
 * handful of nodes costs nothing and removes that whole class of missed match.
 */
export function travelTime(links: readonly CameraLink[], from: string, to: string): number | null {
  if (from === to) return 0;
  const adj = new Map<string, { to: string; t: number }[]>();
  for (const l of links) {
    if (!adj.has(l.from)) adj.set(l.from, []);
    adj.get(l.from)!.push({ to: l.to, t: l.travel_time_s });
  }
  const dist = new Map<string, number>([[from, 0]]);
  // ponytail: linear scan for the next node. n is single-digit; a heap here
  // would be more code than the whole function.
  const unvisited = new Set<string>([from, ...links.flatMap((l) => [l.from, l.to])]);
  while (unvisited.size) {
    let cur: string | null = null, best = Infinity;
    for (const n of unvisited) {
      const d = dist.get(n) ?? Infinity;
      if (d < best) { best = d; cur = n; }
    }
    if (cur === null || best === Infinity) break;
    if (cur === to) return best;
    unvisited.delete(cur);
    for (const e of adj.get(cur) ?? []) {
      const next = best + e.t;
      if (next < (dist.get(e.to) ?? Infinity)) dist.set(e.to, next);
    }
  }
  return dist.get(to) ?? null;
}

// ----------------------------------------------------------------- layers --

function layerPlate(a: TrackRecord, b: TrackRecord, cfg: AssociationConfig): LayerVerdict {
  if (!a.plate_text || !b.plate_text) {
    return { passed: false, score: 0, detail: "plate unreadable on one side" };
  }
  if (a.plate_text !== b.plate_text) {
    return { passed: false, score: 0, detail: "plates differ" };
  }
  const conf = Math.min(a.plate_conf ?? 0, b.plate_conf ?? 0);
  if (conf < cfg.plateConfMin) {
    return { passed: false, score: 0, detail: `plate match but conf ${conf.toFixed(2)} too low` };
  }
  return { passed: true, score: 0.99, detail: `plate ${a.plate_text}` };
}

function layerReid(a: TrackRecord, b: TrackRecord, cfg: AssociationConfig): LayerVerdict {
  const sim = cosine(a.embedding, b.embedding);
  return {
    passed: sim > cfg.reidThreshold,
    score: sim,
    detail: `reid cos ${sim.toFixed(3)}`,
  };
}

function layerTemporal(
  a: TrackRecord, b: TrackRecord, links: readonly CameraLink[], cfg: AssociationConfig,
): TemporalVerdict {
  const expected = travelTime(links, a.camera_id, b.camera_id);
  const exit = Date.parse(a.exit_time ?? a.entry_time);
  const actual = (Date.parse(b.entry_time) - exit) / 1000;

  // Arrived before it left. Not a near miss — the pair is in the wrong order,
  // and confirming it draws a backwards route on the map.
  if (actual <= 0) {
    return { passed: false, impossible: true, score: 0, actual, expected,
             detail: "arrival precedes exit" };
  }
  if (expected === null) {
    // Ignorance, not proof: the camera graph simply has no route between these
    // two. Abstain rather than veto, so a genuine match is not lost to missing
    // topology data.
    return { passed: false, impossible: false, score: 0, actual, expected,
             detail: `no route ${a.camera_id}->${b.camera_id}` };
  }
  const lo = cfg.timeWindowLo * expected;
  const hi = cfg.timeWindowHi * expected;
  const passed = actual >= lo && actual <= hi;
  // 1 at the expected time, falling off linearly in either direction.
  const score = Math.max(0, 1 - Math.abs(actual - expected) / Math.max(expected, 1));
  return {
    passed,
    // A known route the vehicle could not have covered in the observed time.
    impossible: !passed,
    score, actual, expected,
    detail: `${Math.round(actual)}s vs expected ${expected}s`,
  };
}

// ------------------------------------------------------------- the engine --

/**
 * Score one candidate pair. Returns null when fewer than `minLayers` agree.
 *
 * `method` records which layer carried the match, because the dashboard colours
 * each trajectory leg by it — that is how a judge SEES the three-layer engine
 * rather than being told about it.
 */
export function scorePair(
  from: TrackRecord,
  to: TrackRecord,
  links: readonly CameraLink[],
  cfg: AssociationConfig = DEFAULTS,
): MatchResult | null {
  if (from.camera_id === to.camera_id) return null;

  const plate = layerPlate(from, to, cfg);
  const reid = layerReid(from, to, cfg);
  const temporal = layerTemporal(from, to, links, cfg);

  // Layer 3 vetoes rather than votes when it has PROVED impossibility. Plain
  // 2-of-3 lets plate + appearance outvote physics, which is backwards: OCR
  // misreads and cloned plates are both real, and two photos of the same model
  // of white car look alike. A vehicle covering 1.4 km in 30 seconds is not.
  // Missing topology still only abstains — see layerTemporal.
  if (temporal.impossible) return null;

  const agreed = [plate, reid, temporal].filter((l) => l.passed).length;
  if (agreed < cfg.minLayers) return null;

  const colour = histogramSimilarity(from.color_hist, to.color_hist);
  const method: MatchMethod = plate.passed ? "plate" : reid.passed ? "reid" : "spatial_temporal";
  // Appearance evidence is capped below an exact plate read. Without the cap, a
  // near-perfect embedding match with ideal timing scores ~1.0 and displays as
  // MORE certain than reading the plate off the car — which it is not: two
  // white sedans of the same model genuinely look alike, and the number plate
  // is the only thing that identifies the individual vehicle.
  const confidence = plate.passed
    ? 0.99
    : Math.min(0.95,
        Math.round((0.6 * reid.score + 0.2 * colour + 0.2 * temporal.score) * 100) / 100);

  const agreedNames: MatchMethod[] = [];
  if (plate.passed) agreedNames.push("plate");
  if (reid.passed) agreedNames.push("reid");
  if (temporal.passed) agreedNames.push("spatial_temporal");

  return {
    from, to, method, confidence,
    travel_time_s: Math.round(temporal.actual),
    layers: { plate, reid, temporal },
    agreed: agreedNames,
  };
}

/**
 * Best match for a track that has just exited, among tracks that entered
 * elsewhere. Highest confidence wins; ties break toward the more plausible
 * travel time, since two equally-confident candidates are separated by physics
 * and nothing else.
 */
export function associate(
  exited: TrackRecord,
  candidates: readonly TrackRecord[],
  links: readonly CameraLink[],
  cfg: AssociationConfig = DEFAULTS,
): MatchResult | null {
  const scored = candidates
    .map((c) => scorePair(exited, c, links, cfg))
    .filter((m): m is MatchResult => m !== null);
  if (!scored.length) return null;
  scored.sort((a, b) =>
    b.confidence - a.confidence || b.layers.temporal.score - a.layers.temporal.score);
  return scored[0]!;
}

/**
 * Best PREDECESSOR for a track that has just been seen, among tracks that
 * already finished elsewhere.
 *
 * This is the direction real event ordering gives you. A vehicle leaves CAM1 at
 * 10:00 and reaches CAM3 at 10:03, so CAM3's track closes LAST — by the time
 * you hold it, CAM1's track is already history. Searching forward from CAM1
 * would find nothing, because CAM3's arrival has not happened yet.
 */
export function associateArrival(
  arrived: TrackRecord,
  priorTracks: readonly TrackRecord[],
  links: readonly CameraLink[],
  cfg: AssociationConfig = DEFAULTS,
): MatchResult | null {
  const scored = priorTracks
    .map((p) => scorePair(p, arrived, links, cfg))
    .filter((m): m is MatchResult => m !== null);
  if (!scored.length) return null;
  scored.sort((a, b) =>
    b.confidence - a.confidence || b.layers.temporal.score - a.layers.temporal.score);
  return scored[0]!;
}

export const toHop = (m: MatchResult): Hop => ({
  from_camera: m.from.camera_id,
  to_camera: m.to.camera_id,
  method: m.method,
  confidence: m.confidence,
  travel_time_s: m.travel_time_s,
});

/**
 * Stitch confirmed pairs into journeys. Each match is an edge from one track to
 * the next; a trajectory is a maximal chain. Cycles are impossible because
 * every edge moves forward in time, which is why a plain walk is safe here.
 */
export function stitch(matches: readonly MatchResult[]): TrackRecord[][] {
  const next = new Map<string, MatchResult>();
  const hasIncoming = new Set<string>();
  for (const m of matches) {
    // One outgoing edge per track: keep the most confident.
    const prev = next.get(m.from.id);
    if (!prev || m.confidence > prev.confidence) next.set(m.from.id, m);
  }
  for (const m of next.values()) hasIncoming.add(m.to.id);

  const chains: TrackRecord[][] = [];
  for (const m of next.values()) {
    if (hasIncoming.has(m.from.id)) continue; // not a chain head
    const chain = [m.from];
    const seen = new Set([m.from.id]);
    let cur: MatchResult | undefined = m;
    while (cur && !seen.has(cur.to.id)) {
      chain.push(cur.to);
      seen.add(cur.to.id);
      cur = next.get(cur.to.id);
    }
    chains.push(chain);
  }
  return chains;
}
