/**
 * Apply db/schema.sql and seed the camera topology.
 *
 *   npm run db:setup
 *
 * Safe to run repeatedly: every statement in schema.sql is idempotent and the
 * seeds upsert. This exists because the Postgres container entrypoint only runs
 * its init scripts on an EMPTY volume — after the first `up`, a schema change
 * has no other way in short of destroying the data.
 *
 * The camera topology is real configuration, not fixtures: the coordinates and
 * link travel times ARE layer 3 of the association engine. Seeding the four
 * demo cameras here means the pipeline has a graph to reason about from the
 * first run, and `--seed-none` skips it for a real deployment.
 */
import { readFileSync } from "node:fs";
import { CAMERAS, LINKS } from "@/server/mock";
import { sql, DATABASE_URL } from "@/server/db";

async function main() {
  console.log(`applying db/schema.sql to ${DATABASE_URL.replace(/:[^:@]*@/, ":***@")}`);
  // .simple() because schema.sql is many statements in one string, which the
  // extended (prepared) protocol refuses to send.
  await sql.unsafe(readFileSync("db/schema.sql", "utf8")).simple();
  console.log("schema ok");

  if (process.argv.includes("--seed-none")) {
    console.log("skipping seed (--seed-none)");
  } else {
    for (const c of CAMERAS) {
      await sql`
        INSERT INTO cameras (id, name, lat, lon, heading_deg, source_uri)
        VALUES (${c.id}, ${c.name}, ${c.lat}, ${c.lon}, ${c.heading_deg}, ${c.stream_url})
        ON CONFLICT (id) DO UPDATE SET
          name = EXCLUDED.name, lat = EXCLUDED.lat, lon = EXCLUDED.lon,
          heading_deg = EXCLUDED.heading_deg`;
    }
    for (const l of LINKS) {
      await sql`
        INSERT INTO camera_links (from_camera, to_camera, distance_m, travel_time_s)
        VALUES (${l.from}, ${l.to}, ${l.distance_m}, ${l.travel_time_s})
        ON CONFLICT (from_camera, to_camera) DO UPDATE SET
          distance_m = EXCLUDED.distance_m, travel_time_s = EXCLUDED.travel_time_s`;
    }
    console.log(`seeded ${CAMERAS.length} cameras, ${LINKS.length} links`);
  }

  const counts = await sql<{ table_name: string; n: number }[]>`
    SELECT 'cameras' AS table_name, count(*)::int AS n FROM cameras
    UNION ALL SELECT 'camera_links', count(*)::int FROM camera_links
    UNION ALL SELECT 'tracks',       count(*)::int FROM tracks
    UNION ALL SELECT 'detections',   count(*)::int FROM detections
    UNION ALL SELECT 'matches',      count(*)::int FROM matches
    UNION ALL SELECT 'trajectories', count(*)::int FROM trajectories
    UNION ALL SELECT 'alerts',       count(*)::int FROM alerts`;
  for (const c of counts) console.log(`  ${c.table_name.padEnd(13)} ${c.n}`);

  await sql.end();
}

main().catch((e) => {
  console.error("db:setup failed:", e.message);
  console.error("\nIs the database up?  docker compose up -d db redis");
  process.exit(1);
});
