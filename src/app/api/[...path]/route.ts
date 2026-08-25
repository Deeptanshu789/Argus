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
import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import {
  Alert, AnalyticsResponse, Camera, CameraLink, SearchResult, Track, Trajectory,
} from "@/contract";
import * as db from "@/server/db";

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
      case "alerts": {
        const acked = q.get("acked");
        return ok(z.array(Alert),
          await db.getAlerts(acked === null ? undefined : acked === "true"), route);
      }
    }
  } catch (err) {
    return down(route, err);
  }
  return NextResponse.json({ error: `no route: ${route}` }, { status: 404 });
}

export async function POST(_req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const route = path.join("/");

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
