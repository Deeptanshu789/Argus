/**
 * The shared chrome's pure logic, checked.
 *
 * Everything here is used by every view, so a silent change is a silent change
 * in six places at once. `Pips` in particular renders Module C's 2-of-3 rule --
 * the claim the whole project rests on -- and it derives entirely from
 * `layerCount`, so that one is worth an assert even though it is three lines.
 *
 *   npx tsx src/components/ui.selfcheck.ts
 */
import assert from "node:assert/strict";
import { RAMP, ago, hopTime, layerCount, methodColour, rampFor, statusColour, T }
  from "./ui";

// --- the 2-of-3 rule -------------------------------------------------------
assert.equal(layerCount("plate"), 1);
assert.equal(layerCount("plate+reid"), 2);
assert.equal(layerCount("plate+reid+spatial_temporal"), 3);
assert.equal(layerCount("reid+spatial_temporal"), 2);
// An unknown layer name must not inflate the count. Module C could grow a
// fourth layer, and until Pips grows a fourth slot the glyph must stay honest.
assert.equal(layerCount("plate+colour_histogram"), 1);
assert.equal(layerCount(""), 0);

// --- colour by deciding layer ---------------------------------------------
// A plate match and a Re-ID match must never share a colour; that distinction
// is the whole argument of the map and the ledger.
assert.notEqual(methodColour("plate"), methodColour("reid"));
assert.notEqual(methodColour("reid"), methodColour("spatial_temporal"));
assert.equal(methodColour("plate+reid"), methodColour("plate"),
             "a compound method takes the colour of its strongest layer");

// --- the congestion ramp ---------------------------------------------------
assert.equal(rampFor(0), RAMP[0]);
assert.equal(rampFor(1), RAMP[RAMP.length - 1]);
assert.equal(rampFor(0.5), RAMP[3]);
// Out of range and NaN must still pick a colour: a cell that renders nothing
// reads as "no data", which is a different fact from "score unavailable".
assert.equal(rampFor(-3), RAMP[0]);
assert.equal(rampFor(9), RAMP[RAMP.length - 1]);
assert.equal(rampFor(Number.NaN), RAMP[0]);

// --- status colours --------------------------------------------------------
assert.equal(statusColour("online"), T.ok);
assert.equal(statusColour("degraded"), T.warn);
assert.equal(statusColour("error"), T.bad);
assert.equal(statusColour("nonsense"), T.line2, "an unknown status is not an alarm");

// --- travel time -----------------------------------------------------------
assert.equal(hopTime(45), "45 s");
assert.equal(hopTime(60), "1 m 00 s");
assert.equal(hopTime(212), "3 m 32 s");
assert.equal(hopTime(605), "10 m 05 s", "seconds are zero padded or the column jitters");

// --- relative time ---------------------------------------------------------
const now = Date.now();
assert.equal(ago(new Date(now - 5_000).toISOString()), "5s ago");
assert.equal(ago(new Date(now - 300_000).toISOString()), "5m ago");
assert.equal(ago(new Date(now - 7_200_000).toISOString()), "2h ago");
// A clock skew between the worker and the browser must not print "-3s ago".
assert.equal(ago(new Date(now + 60_000).toISOString()), "0s ago");

console.log("ui selfcheck OK");
