/**
 * Delete trajectories that are not valid paths.
 *
 *     npx tsx db/repair.ts            # report only
 *     npx tsx db/repair.ts --apply    # delete them
 *
 * A trajectory records one vehicle's route as an ordered list of track ids, so
 * the same track cannot appear twice. Sidecars used to restart their track
 * numbering from 1, which let two different vehicles share one row and produced
 * cycles like [251, 255, 252, 251] with timestamps running backwards. The
 * sidecar now prefixes every track id with a per-run id, and
 * extendTrajectory() refuses to close a cycle -- but rows written before those
 * fixes are still in the table, and every read of them fails the contract.
 *
 * Two ways a row can be invalid, both from the same cause:
 *
 *   CYCLE      the same track appears twice in the chain.
 *   BACKWARDS  a later track in the chain entered its camera BEFORE an earlier
 *              one did. A merged track kept one row's id and another run's
 *              timestamps, so the journey appears to travel back in time and
 *              the contract's ordering check rejects it on every read.
 *   MIXED      the chain spans both uploaded footage and live cameras. Module C
 *              no longer compares across that boundary, but journeys stitched
 *              before it did are still stored, and they put cameras the
 *              operator never uploaded on their results page.
 *
 * Deletes trajectories, and the small number of TRACKS that are themselves
 * impossible. Detections are untouched: an observation is real even when the
 * track it was filed under is not.
 *
 * An impossible track is one whose exit time is more than an hour after its
 * entry. A track closes after MISS_LIMIT frames without a sighting -- two
 * seconds at 5 fps -- so no genuine track lasts minutes, let alone hours. One
 * arises when two runs share a pinned ARGUS_RUN_ID hours apart: the upsert
 * keeps the first run's entry_time and takes the last run's exit_time, and the
 * row spans the gap between them. Its recent exit time then makes it look live
 * to the association window, it matches a fresh vehicle by plate, and the
 * trajectory built from it has timestamps running backwards. Deleting the bad
 * trajectory alone does not help: the poisoned track is still there to build
 * another one.
 */
import { sql } from "@/server/db";

const apply = process.argv.includes("--apply");

const bad = await sql<{ id: string; reason: string; detail: string }[]>`
  WITH ordered AS (
    SELECT tr.id,
           array_length(tr.track_ids, 1) AS n,
           (SELECT count(DISTINCT t) FROM unnest(tr.track_ids) AS t) AS distinct_n,
           (SELECT bool_or(t.entry_time < prev.entry_time)
              FROM unnest(tr.track_ids) WITH ORDINALITY AS u(tid, pos)
              JOIN tracks t    ON t.id = u.tid
              JOIN unnest(tr.track_ids) WITH ORDINALITY AS p(ptid, ppos)
                ON p.ppos = u.pos - 1
              JOIN tracks prev ON prev.id = p.ptid) AS backwards,
           (SELECT count(DISTINCT c.is_upload) > 1
              FROM unnest(tr.track_ids) AS u(tid)
              JOIN tracks t  ON t.id = u.tid
              JOIN cameras c ON c.id = t.camera_id) AS mixed,
           tr.track_ids
      FROM trajectories tr)
  SELECT id,
         CASE WHEN n <> distinct_n THEN 'CYCLE'
              WHEN mixed           THEN 'MIXED'
              ELSE 'BACKWARDS' END AS reason,
         array_to_string(track_ids, ', ') AS detail
    FROM ordered
   WHERE n <> distinct_n OR backwards OR mixed
   ORDER BY id`;

/** A single tracked vehicle cannot be in one camera's view for an hour. */
const impossible = await sql<
  { id: string; camera_id: string; track_id: string; hours: number }[]
>`
  SELECT id, camera_id, track_id,
         EXTRACT(EPOCH FROM (exit_time - entry_time)) / 3600 AS hours
    FROM tracks
   WHERE exit_time IS NOT NULL
     AND exit_time - entry_time > INTERVAL '1 hour'
   ORDER BY id`;

if (bad.length === 0) {
  console.log("no invalid trajectories");
} else {
  for (const t of bad) console.log(`  ${t.reason.padEnd(9)} trajectory ${t.id}: [${t.detail}]`);
}

for (const t of impossible) {
  console.log(`  IMPOSSIBLE track ${t.id}: ${t.camera_id}/${t.track_id} ` +
              `spans ${Number(t.hours).toFixed(1)} h`);
}

const total = bad.length + impossible.length;
if (total === 0) {
  console.log("nothing to repair");
} else if (apply) {
  if (bad.length) {
    await sql`DELETE FROM trajectories WHERE id = ANY(${bad.map((t) => t.id)}::bigint[])`;
  }
  if (impossible.length) {
    const ids = impossible.map((t) => t.id);
    // Anything stitched FROM a poisoned track is poisoned too, whether or not
    // its own ordering happened to come out valid.
    await sql`DELETE FROM trajectories
               WHERE track_ids && ${ids}::bigint[]`;
    // matches has a foreign key onto tracks, so it goes first.
    await sql`DELETE FROM matches
               WHERE from_track = ANY(${ids}::bigint[])
                  OR to_track   = ANY(${ids}::bigint[])`;
    await sql`DELETE FROM tracks WHERE id = ANY(${ids}::bigint[])`;
  }
  console.log(`\ndeleted ${bad.length} trajectory(ies) and ${impossible.length} track(s)`);
} else {
  console.log(`\n${total} invalid. Re-run with --apply to delete them.`);
}

await sql.end();
