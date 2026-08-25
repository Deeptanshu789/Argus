/**
 * Catch-all mock API. One file instead of eight route folders — the routes are
 * a flat dispatch table, and Next's file-per-route convention buys nothing here.
 *
 * Real routes go under /api/* and replace these one at a time behind identical
 * shapes. The frontend flips VITE-style env `NEXT_PUBLIC_MOCK` and changes
 * nothing else.
 */
import { NextResponse, type NextRequest } from "next/server";
import * as mock from "@/server/mock";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const q = req.nextUrl.searchParams;
  const route = path.join("/");

  switch (route) {
    case "cameras":
      return NextResponse.json(mock.getCameras());
    case "cameras/links":
      return NextResponse.json(mock.getLinks());
    case "tracks":
      return NextResponse.json(mock.getTracks({
        camera: q.get("camera") ?? undefined,
        since: q.get("since") ?? undefined,
        limit: Number(q.get("limit") ?? 100),
      }));
    case "trajectories":
      return NextResponse.json(mock.getTrajectories({
        since: q.get("since") ?? undefined,
        limit: Number(q.get("limit") ?? 50),
      }));
    case "search":
      return NextResponse.json(mock.search(q.get("plate") ?? ""));
    case "analytics":
      return NextResponse.json(mock.getAnalytics(q.get("camera") ?? undefined));
    case "alerts": {
      const acked = q.get("acked");
      return NextResponse.json(mock.getAlerts(acked === null ? undefined : acked === "true"));
    }
  }
  return NextResponse.json({ error: `no mock route: ${route}` }, { status: 404 });
}

export async function POST(_req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  // alerts/{id}/ack
  if (path[0] === "alerts" && path[2] === "ack" && path[1]) {
    return NextResponse.json(mock.ackAlert(path[1]));
  }
  return NextResponse.json({ error: `no mock route: ${path.join("/")}` }, { status: 404 });
}
