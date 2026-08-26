/**
 * The live API. Same dispatch-table shape as the mock, same response shapes —
 * that is the point: the frontend switches between them with one env var and
 * changes nothing else.
 *
 * Every response is validated against `src/contract.ts` before it leaves.
 * A zod parse on the way out costs microseconds and converts "the dashboard
 * renders blank and nobody knows why" into a named error in the server log
 * pointing at the exact field. That trade is worth making on a 36-hour build.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import {
  Alert, AnalyticsResponse, Camera, CameraLink, Device, SearchResult, Track,
  Trajectory, Upload, UploadResult,
} from "@/contract";
import * as db from "@/server/db";

/**
 * Where uploaded video lands. Outside `public/`, deliberately: these are files
 * an operator chose off their own machine, and serving them back at a guessable
 * URL would publish them to anyone who can reach the dashboard.
 */
const UPLOAD_DIR = process.env.ARGUS_UPLOAD_DIR ?? "uploads";

/** Containers OpenCV can actually decode. Anything else wastes a sidecar boot. */
const VIDEO_EXT = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]);

/** Per file. A 36-hour build has no chunked-upload story, and Node buffers the
 *  whole request body in memory, so this is a real ceiling and not a policy. */
const MAX_BYTES = 512 * 1024 * 1024;

/** Strip every path component and anything that is not plainly a filename.
 *  A name like "../../etc/passwd" must not survive into a path join. */
const safeName = (name: string) =>
  (name.split(/[\\/]/).pop() ?? "video")
    .replace(/[^A-Za-z0-9._-]/g, "_")
    .slice(-80) || "video";

export const dynamic = "force-dynamic";

/** Validate on the way out, and say which route broke if it does not hold. */
function ok<T>(schema: z.ZodType<T>, data: unknown, route: string) {
  const parsed = schema.safeParse(data);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    console.error(`/api/${route} violates the contract:`,
      `${issue?.path.join(".")}: ${issue?.message}`);
    return NextResponse.json(
      { error: "response failed contract validation", route, detail: issue?.message },
      { status: 500 },
    );
  }
  return NextResponse.json(parsed.data);
}

/** A database that is down is a 503, not a 500: it is a dependency outage the
 *  caller can retry, and the dashboard shows "reconnecting" rather than "bug". */
function down(route: string, err: unknown) {
  const message = err instanceof Error ? err.message : String(err);
  console.error(`/api/${route} failed:`, message);
  const isConn = /ECONNREFUSED|ENOTFOUND|CONNECT_TIMEOUT|terminating connection/i.test(message);
  return NextResponse.json(
    { error: isConn ? "database unavailable" : "query failed", route, detail: message },
    { status: isConn ? 503 : 500 },
  );
}

const num = (v: string | null, fallback: number) => {
  const n = Number(v);
  return v !== null && Number.isFinite(n) ? n : fallback;
};

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const q = req.nextUrl.searchParams;
  const route = path.join("/");

  try {
    switch (route) {
      case "health": {
        const alive = await db.ping();
        return NextResponse.json({ ok: alive, database: alive ? "up" : "down" },
          { status: alive ? 200 : 503 });
      }
      case "cameras":
        return ok(z.array(Camera), await db.getCameras(), route);
      case "cameras/links":
        return ok(z.array(CameraLink), await db.getLinks(), route);
      case "tracks":
        return ok(z.array(Track), await db.getTracks({
          camera: q.get("camera") ?? undefined,
          since: q.get("since") ?? undefined,
          limit: num(q.get("limit"), 100),
        }), route);
      case "trajectories":
        return ok(z.array(Trajectory), await db.getTrajectories({
          since: q.get("since") ?? undefined,
          limit: num(q.get("limit"), 50),
        }), route);
      case "search":
        // A miss is a 200 with empty arrays, never a 404.
        return ok(SearchResult, await db.search(q.get("plate") ?? ""), route);
      case "analytics":
        return ok(AnalyticsResponse,
          await db.getAnalytics(q.get("camera") ?? undefined, num(q.get("hours"), 6)), route);
      case "devices":
        return ok(z.array(Device), await db.getDevices(), route);
      case "uploads":
        return ok(z.array(Upload), await db.getUploads(num(q.get("limit"), 20)), route);
      case "alerts": {
        const acked = q.get("acked");
        return ok(z.array(Alert),
          await db.getAlerts(acked === null ? undefined : acked === "true",
                             num(q.get("limit"), 200)), route);
      }
    }

    // devices/<code> — what the phone page reads to confirm a code is real
    if (path[0] === "devices" && path[1] && path.length === 2) {
      const device = await db.getDeviceByCode(path[1]);
      if (!device) return NextResponse.json({ error: "unknown code" }, { status: 404 });
      return ok(Device, device, route);
    }

    // uploads/<id>
    if (path[0] === "uploads" && path[1] && path.length === 2) {
      const result = await db.getUploadResult(path[1]);
      if (!result) return NextResponse.json({ error: "no such upload" }, { status: 404 });
      return ok(UploadResult, result, route);
    }
  } catch (err) {
    return down(route, err);
  }
  return NextResponse.json({ error: `no route: ${route}` }, { status: 404 });
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const route = path.join("/");

  // devices — issue a pairing code
  if (route === "devices") {
    try {
      const body = await req.json().catch(() => ({}));
      const label = typeof (body as { label?: unknown }).label === "string"
        ? String((body as { label: string }).label).slice(0, 60) : null;
      return ok(Device, await db.createDevice(label || null), route);
    } catch (err) {
      return down(route, err);
    }
  }

  // devices/<code>/url — pair by pointing at an IP-camera app's stream instead
  if (path[0] === "devices" && path[2] === "url" && path[1]) {
    try {
      const body = await req.json().catch(() => ({}));
      const url = String((body as { url?: unknown }).url ?? "").trim();
      // Only what the sidecar can actually open, and nothing that would make it
      // read a local file through a crafted "source".
      if (!/^(rtsp|rtmps?|https?):\/\/[^\s]+$/i.test(url)) {
        return NextResponse.json(
          { error: "expected an rtsp:// or http:// stream url" }, { status: 400 });
      }
      const device = await db.pairDevice(path[1], "url", url);
      if (!device) return NextResponse.json({ error: "unknown code" }, { status: 404 });
      return ok(Device, device, route);
    } catch (err) {
      return down(route, err);
    }
  }

  // devices/<id>/revoke
  if (path[0] === "devices" && path[2] === "revoke" && path[1]) {
    try {
      const gone = await db.revokeDevice(path[1]);
      if (!gone) return NextResponse.json({ error: "no such device" }, { status: 404 });
      return NextResponse.json({ ok: true });
    } catch (err) {
      return down(route, err);
    }
  }

  if (route === "uploads") {
    try {
      const form = await req.formData();
      const files = form.getAll("files").filter((f): f is File => f instanceof File);
      if (!files.length) {
        return NextResponse.json({ error: "no files" }, { status: 400 });
      }
      for (const f of files) {
        if (!VIDEO_EXT.has(extname(f.name).toLowerCase())) {
          return NextResponse.json(
            { error: `${f.name}: not a video (${[...VIDEO_EXT].join(", ")})` },
            { status: 415 });
        }
        if (f.size > MAX_BYTES) {
          return NextResponse.json(
            { error: `${f.name}: ${(f.size / 1e6).toFixed(0)} MB exceeds the ` +
                     `${MAX_BYTES / 1e6} MB limit` },
            { status: 413 });
        }
      }

      const gapRaw = form.get("gap_seconds");
      const gap = Number(gapRaw);
      const gapSeconds =
        typeof gapRaw === "string" && gapRaw.trim() !== "" && Number.isFinite(gap) && gap > 0
          ? Math.round(gap) : null;

      await mkdir(UPLOAD_DIR, { recursive: true });
      const stamp = Date.now();
      const saved: { filename: string; path: string }[] = [];
      for (const [i, f] of files.entries()) {
        const name = safeName(f.name);
        const dest = join(UPLOAD_DIR, `${stamp}-${i}-${name}`);
        await writeFile(dest, Buffer.from(await f.arrayBuffer()));
        saved.push({ filename: name, path: dest });
      }

      // The worker picks it up from here. Decoding video inside the web server
      // would pin a core and stall every other request, which is the whole
      // reason the worker is a separate process.
      const id = await db.createUpload({
        label: typeof form.get("label") === "string" && form.get("label")
          ? String(form.get("label")) : null,
        gapSeconds,
        files: saved,
      });
      const upload = await db.getUpload(id);
      return ok(Upload, upload, route);
    } catch (err) {
      return down(route, err);
    }
  }

  if (path[0] === "alerts" && path[2] === "ack" && path[1]) {
    try {
      const alert = await db.ackAlert(path[1]);
      if (!alert) return NextResponse.json({ error: "no such alert" }, { status: 404 });
      return ok(Alert, alert, route);
    } catch (err) {
      return down(route, err);
    }
  }
  return NextResponse.json({ error: `no route: ${route}` }, { status: 404 });
}
