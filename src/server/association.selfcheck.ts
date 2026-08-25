/**
 * One runnable check for Module C. If the matching logic breaks, this fails.
 *
 *   npx tsx src/server/association.selfcheck.ts
 */
import assert from "node:assert/strict";
import type { CameraLink } from "@/contract";
import {
  DEFAULTS, associate, associateArrival, cosine, histogramSimilarity, scorePair, stitch,
  travelTime, type TrackRecord,
} from "@/server/association";

const LINKS: CameraLink[] = [
  { from: "CAM1", to: "CAM3", distance_m: 1400, travel_time_s: 168 },
  { from: "CAM3", to: "CAM1", distance_m: 1400, travel_time_s: 168 },
  { from: "CAM3", to: "CAM2", distance_m: 900,  travel_time_s: 120 },
];

const T0 = Date.parse("2026-08-25T10:00:00Z");
const at = (s: number) => new Date(T0 + s * 1000).toISOString().replace(/\.\d{3}Z$/, "Z");

/** Deterministic unit-ish vector; `drift` rotates it away from the original. */
function embed(seed: number, drift = 0): number[] {
  return Array.from({ length: 512 }, (_, i) =>
    Math.sin((i + 1) * 0.017 + seed) + drift * Math.cos((i + 1) * 0.031 + seed));
}

function track(o: Partial<TrackRecord> & { id: string; camera_id: string }): TrackRecord {
  return {
    track_id: `T${o.id}`, plate_text: null, plate_conf: null,
    embedding: embed(1), color_hist: [10, 20, 30, 40],
    entry_time: at(0), exit_time: at(10), ...o,
  };
}

// -------------------------------------------------------------- primitives --
assert.equal(cosine(embed(1), embed(1)), 1, "identical vectors are 1.0");
assert.ok(cosine(embed(1), embed(9)) < 0.75, "different vehicles fall below threshold");
assert.equal(cosine([1, 2], [1, 2, 3]), 0, "length mismatch is 0, not a crash");
assert.equal(cosine([0, 0], [0, 0]), 0, "zero vector is 0, not NaN");
assert.equal(histogramSimilarity([1, 2, 3], [1, 2, 3]), 1);
assert.equal(histogramSimilarity([1, 2, 3], [0, 0, 0]), 0);

// ------------------------------------------------------------- road graph --
assert.equal(travelTime(LINKS, "CAM1", "CAM3"), 168, "direct link");
assert.equal(travelTime(LINKS, "CAM1", "CAM1"), 0);
// CAM1->CAM2 has no direct edge. A direct-only lookup would call every such
// match physically impossible; the graph walk finds CAM1->CAM3->CAM2.
assert.equal(travelTime(LINKS, "CAM1", "CAM2"), 288, "multi-hop route");
assert.equal(travelTime(LINKS, "CAM1", "CAM9"), null, "unknown camera");

// ------------------------------------------- layer 1 + 3: the common case --
{
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821", plate_conf: 0.97 });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.95,
                     entry_time: at(180), exit_time: at(190), embedding: embed(9) });
  const m = scorePair(from, to, LINKS);
  assert.ok(m, "plate + feasible time must confirm");
  assert.equal(m.method, "plate");
  assert.equal(m.confidence, 0.99);
  assert.equal(m.travel_time_s, 170);
}

// ------------------------- THE REQUIRED TEST: physics overrides everything --
{
  // Same plate, same confidence — but 30 s for a 168 s journey.
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821", plate_conf: 0.97 });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.97,
                     entry_time: at(40), exit_time: at(50), embedding: embed(9) });
  assert.ok(scorePair(from, to, LINKS) === null,
    "a teleporting plate match must be rejected: OCR misreads and cloned plates are real");
}
{
  // Arrived before it left.
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA01AA1111", plate_conf: 0.9,
                       embedding: embed(2), exit_time: at(500) });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA01AA1111", plate_conf: 0.9,
                     embedding: embed(2), entry_time: at(100) });
  assert.ok(scorePair(from, to, LINKS) === null, "negative travel time must be rejected");
}

// ------------------------------- layer 2 + 3: unreadable plate still works --
{
  const from = track({ id: "a", camera_id: "CAM1", embedding: embed(4) });
  const to = track({ id: "b", camera_id: "CAM3", embedding: embed(4, 0.05),
                     entry_time: at(180), exit_time: at(190) });
  const m = scorePair(from, to, LINKS);
  assert.ok(m, "no plate on either side, but appearance + physics agree");
  assert.equal(m.method, "reid");
  // Appearance evidence never outranks an exact plate read, however good the
  // embedding and timing agreement happens to be.
  assert.ok(m.confidence > 0.7 && m.confidence <= 0.95,
    `reid confidence must stay below a plate match, got ${m.confidence}`);
  assert.equal(m.layers.plate.passed, false);
}

// ----------------------------------------- one layer alone is never enough --
{
  // Appearance agrees; travel time does not.
  const from = track({ id: "a", camera_id: "CAM1", embedding: embed(4) });
  const to = track({ id: "b", camera_id: "CAM3", embedding: embed(4, 0.05),
                     entry_time: at(20), exit_time: at(30) });
  assert.ok(scorePair(from, to, LINKS) === null, "reid alone must not confirm");
}
{
  // Travel time is plausible; nothing else agrees. This is the false-positive
  // trap: without the 2-of-3 rule, every vehicle passing at the right moment
  // would match every other one.
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA01AA1111", plate_conf: 0.95,
                       embedding: embed(1) });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "MH02BB2222", plate_conf: 0.95,
                     embedding: embed(50), entry_time: at(180), exit_time: at(190) });
  assert.ok(scorePair(from, to, LINKS) === null, "timing alone must not confirm");
}

// --------------------------------------------------------- low-confidence --
{
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821", plate_conf: 0.55,
                       embedding: embed(4) });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.55,
                     embedding: embed(90), entry_time: at(180), exit_time: at(190) });
  assert.ok(scorePair(from, to, LINKS) === null,
    "matching text at 0.55 confidence is two guesses agreeing, not evidence");
}

// ------------------------------------------------------------- same camera --
assert.ok(
  scorePair(track({ id: "a", camera_id: "CAM1" }), track({ id: "b", camera_id: "CAM1" }), LINKS)
    === null, "a track cannot match another on its own camera");

// ------------------------------------------------- best-of among candidates --
{
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821", plate_conf: 0.97 });
  const decoy = track({ id: "x", camera_id: "CAM3", embedding: from.embedding,
                        entry_time: at(300), exit_time: at(310) });
  const right = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.96,
                        embedding: embed(9), entry_time: at(180), exit_time: at(190) });
  const m = associate(from, [decoy, right], LINKS);
  assert.equal(m?.to.id, "b", "plate match outranks an appearance-only candidate");
  assert.ok(associate(from, [], LINKS) === null, "no candidates is null, not a throw");
}

// ---------------------------------------------------------------- stitching --
{
  const a = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821", plate_conf: 0.97 });
  const b = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.97,
                    entry_time: at(180), exit_time: at(190) });
  const c = track({ id: "c", camera_id: "CAM2", plate_text: "KA05MR7821", plate_conf: 0.97,
                    entry_time: at(320), exit_time: at(330) });
  const m1 = scorePair(a, b, LINKS), m2 = scorePair(b, c, LINKS);
  assert.ok(m1 && m2, "both legs must confirm");
  const chains = stitch([m1, m2]);
  assert.equal(chains.length, 1, "two edges sharing a node are one journey, not two");
  assert.deepEqual(chains[0]!.map((t) => t.camera_id), ["CAM1", "CAM3", "CAM2"]);
  assert.deepEqual(stitch([]), []);
}

// --------------------------------------------------------- config is a knob --
{
  // Appearance agrees; no plate; CAM4 has no route so layer 3 abstains. One
  // layer, so the default rejects. Lowering minLayers to 1 accepts it — which
  // is exactly how you end up demoing a wrong route, hence the default of 2.
  const from = track({ id: "a", camera_id: "CAM1", embedding: embed(4) });
  const to = track({ id: "b", camera_id: "CAM4", embedding: embed(4, 0.05),
                     entry_time: at(400), exit_time: at(410) });
  assert.ok(scorePair(from, to, LINKS) === null, "one layer is not enough by default");
  assert.ok(scorePair(from, to, LINKS, { ...DEFAULTS, minLayers: 1 }),
    "minLayers is the knob that rejected it");
}

// ----------------------- the veto is NOT a knob: minLayers cannot unlock it --
{
  const e = embed(31);
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821",
                       plate_conf: 0.97, embedding: e });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.97,
                     embedding: e, entry_time: at(40), exit_time: at(50) });
  assert.ok(scorePair(from, to, LINKS, { ...DEFAULTS, minLayers: 1 }) === null,
    "no configuration should let a physically impossible match through");
}

// -------------------------- layer 3 VETOES, it does not merely fail to vote --
{
  // Plate AND appearance both agree — a plain 2-of-3 tally would confirm this.
  // The vehicle would have had to cover 1.4 km in 30 s.
  const e = embed(7);
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821",
                       plate_conf: 0.97, embedding: e });
  const to = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821", plate_conf: 0.97,
                     embedding: e, entry_time: at(40), exit_time: at(50) });
  assert.ok(scorePair(from, to, LINKS) === null,
    "physics must veto, not be outvoted: OCR misreads and cloned plates are real, teleporting is not");
}

// ------------------ ...but abstains when the graph simply has no route data --
{
  // CAM4 is in no link. That is ignorance, not proof of impossibility, so a
  // strong plate + appearance match should still stand rather than be lost to
  // incomplete topology.
  const e = embed(11);
  const from = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821",
                       plate_conf: 0.97, embedding: e });
  const to = track({ id: "b", camera_id: "CAM4", plate_text: "KA05MR7821", plate_conf: 0.97,
                     embedding: e, entry_time: at(400), exit_time: at(410) });
  const m = scorePair(from, to, LINKS);
  assert.ok(m, "missing topology must abstain, not veto");
  assert.equal(m.layers.temporal.passed, false);
  assert.deepEqual(m.agreed, ["plate", "reid"]);
}

// --------------- appearance confidence is capped below an exact plate read --
{
  // Identical embedding, identical colour, textbook travel time: the raw
  // formula scores ~1.0 here. It must still not present as plate-certain.
  const e = embed(21);
  const from = track({ id: "a", camera_id: "CAM1", embedding: e });
  const to = track({ id: "b", camera_id: "CAM3", embedding: e,
                     entry_time: at(178), exit_time: at(188) });
  const m = scorePair(from, to, LINKS);
  assert.ok(m && m.method === "reid");
  assert.equal(m.confidence, 0.95, "capped, not 1.0");
}

// ------------------------- arrivals search backwards, which is how events land --
{
  // Real ordering: CAM1's track closes at 10:00:10, CAM3's at 10:03:20. When
  // the CAM3 track arrives, CAM1's is already history — so the search runs
  // backwards, not forwards.
  const earlier = track({ id: "a", camera_id: "CAM1", plate_text: "KA05MR7821",
                          plate_conf: 0.97, embedding: embed(3) });
  const arrived = track({ id: "b", camera_id: "CAM3", plate_text: "KA05MR7821",
                          plate_conf: 0.96, embedding: embed(8),
                          entry_time: at(180), exit_time: at(200) });
  const m = associateArrival(arrived, [earlier], LINKS);
  assert.ok(m, "arrival must find its predecessor");
  assert.equal(m.from.id, "a");
  assert.equal(m.to.id, "b");
  assert.equal(m.method, "plate");

  // The forward search finds nothing here, which is precisely the bug this
  // function exists to prevent.
  assert.ok(associate(arrived, [earlier], LINKS) === null,
    "searching forward from the later track cannot work: the earlier one is in its past");
  assert.ok(associateArrival(arrived, [], LINKS) === null);
}

console.log("association selfcheck ok");
