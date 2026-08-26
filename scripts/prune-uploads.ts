/**
 * Delete the VIDEO FILES of old uploads.
 *
 *     npx tsx scripts/prune-uploads.ts --days 7            # report only
 *     npx tsx scripts/prune-uploads.ts --days 7 --apply    # delete them
 *
 * Uploaded video is the only thing in Argus that grows without bound. A few
 * minutes of 1080p is hundreds of megabytes, the worker decodes it once and
 * never reads it again, and nothing removes it — on a VPS the disk fills
 * silently and the first symptom is Postgres refusing to write.
 *
 * WHAT IS DELETED: the source video, and only for uploads that have finished.
 * A pending or running upload is skipped however old it is, because the worker
 * may be reading it right now.
 *
 * WHAT IS KEPT: everything the system produced from it — the plates, tracks,
 * detections and journeys, and the upload row itself. Those are rows, they are
 * small, and they are the point. The results page keeps working; only the
 * original footage is gone.
 */
import { readdir, stat, unlink } from "node:fs/promises";
import { join } from "node:path";
import { sql, uploadsOlderThan } from "@/server/db";

const arg = (name: string, fallback: number) => {
  const i = process.argv.indexOf(`--${name}`);
  const v = i >= 0 ? Number(process.argv[i + 1]) : NaN;
  return Number.isFinite(v) ? v : fallback;
};

const days = arg("days", 7);
const apply = process.argv.includes("--apply");
const dir = process.env.ARGUS_UPLOAD_DIR ?? "uploads";

const size = async (p: string) => {
  try {
    return (await stat(p)).size;
  } catch {
    return 0;   // already gone; pruning is idempotent
  }
};

const human = (n: number) =>
  n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${(n / 1e6).toFixed(0)} MB`;

const rows = await uploadsOlderThan(days);
let total = 0;
let files = 0;

for (const u of rows) {
  const present: string[] = [];
  for (const f of u.files) {
    const bytes = await size(f);
    if (!bytes) continue;
    present.push(f);
    total += bytes;
    files++;
  }
  if (!present.length) continue;
  const age = Math.round((Date.now() - u.created_at.getTime()) / 86_400_000);
  console.log(`  upload ${u.id}  ${age}d  ${u.label ?? "untitled"}  ` +
              `${present.length} file(s)`);
  if (apply) {
    for (const f of present) await unlink(f).catch(() => {});
  }
}

// Files with no upload row at all: a failed request that saved before it
// inserted, or a row deleted by hand. Nothing will ever read these.
const known = new Set(rows.flatMap((u) => u.files.map((f) => f.split("/").pop())));
const all = await readdir(dir).catch(() => [] as string[]);
const live = await sql<{ path: string }[]>`SELECT path FROM upload_sources`;
const referenced = new Set(live.map((r) => r.path.split("/").pop()));
let orphans = 0;
let orphanBytes = 0;
for (const name of all) {
  if (referenced.has(name) || known.has(name)) continue;
  const p = join(dir, name);
  const bytes = await size(p);
  if (!bytes) continue;
  orphans++;
  orphanBytes += bytes;
  console.log(`  orphan  ${name}  ${human(bytes)}`);
  if (apply) await unlink(p).catch(() => {});
}

console.log(
  `\n${files} file(s) from uploads older than ${days} days, ${human(total)}` +
  (orphans ? `, plus ${orphans} orphan(s), ${human(orphanBytes)}` : ""));
console.log(apply ? "deleted." : "Re-run with --apply to delete them.");

await sql.end();
