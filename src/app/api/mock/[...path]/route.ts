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
    case "devices":
      return NextResponse.json(mock.getDevices());
    case "uploads": {
      const limit = Number(q.get("limit"));
      return NextResponse.json(mock.getUploads(
        Number.isFinite(limit) && limit > 0 ? limit : 20));
    }
    case "alerts": {
      const acked = q.get("acked");
      const limit = Number(q.get("limit"));
      return NextResponse.json(mock.getAlerts(
        acked === null ? undefined : acked === "true",
        Number.isFinite(limit) && limit > 0 ? limit : 200));
    }
  }

  // devices/<code>
  if (path[0] === "devices" && path[1] && path.length === 2) {
    const device = mock.getDeviceByCode(path[1]);
    if (!device) return NextResponse.json({ error: "unknown code" }, { status: 404 });
    return NextResponse.json(device);
  }

  // uploads/<id>
  if (path[0] === "uploads" && path[1] && path.length === 2) {
    const result = mock.getUpload(path[1]);
    if (!result) return NextResponse.json({ error: "no such upload" }, { status: 404 });
    return NextResponse.json(result);
  }
  return NextResponse.json({ error: `no mock route: ${route}` }, { status: 404 });
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  // alerts/{id}/ack
  if (path[0] === "alerts" && path[2] === "ack" && path[1]) {
    return NextResponse.json(mock.ackAlert(path[1]));
  }
  if (path.join("/") === "devices") {
    const body = await req.json().catch(() => ({}));
    const label = typeof (body as { label?: unknown }).label === "string"
      ? String((body as { label: string }).label) : null;
    return NextResponse.json(mock.createDevice(label || null));
  }
  if (path[0] === "devices" && path[2] === "url" && path[1]) {
    const body = await req.json().catch(() => ({}));
    const url = String((body as { url?: unknown }).url ?? "").trim();
    if (!/^(rtsp|rtmps?|https?):\/\/[^\s]+$/i.test(url)) {
      return NextResponse.json(
        { error: "expected an rtsp:// or http:// stream url" }, { status: 400 });
    }
    const device = mock.pairDeviceUrl(path[1], url);
    if (!device) return NextResponse.json({ error: "unknown code" }, { status: 404 });
    return NextResponse.json(device);
  }
  if (path[0] === "devices" && path[2] === "revoke" && path[1]) {
    if (!mock.revokeDevice(path[1])) {
      return NextResponse.json({ error: "no such device" }, { status: 404 });
    }
    return NextResponse.json({ ok: true });
  }

  // uploads — accepted and queued, then nothing. The mock has no worker, and a
  // fake "done" would hide the one failure worth designing for: the real worker
  // not running.
  if (path.join("/") === "uploads") {
    const form = await req.formData();
    const names = form.getAll("files")
      .filter((f): f is File => f instanceof File)
      .map((f) => f.name);
    if (!names.length) return NextResponse.json({ error: "no files" }, { status: 400 });
    const gap = Number(form.get("gap_seconds"));
    const label = form.get("label");
    return NextResponse.json(mock.createUpload(
      names,
      Number.isFinite(gap) && gap > 0 ? Math.round(gap) : null,
      typeof label === "string" && label ? label : null));
  }
  return NextResponse.json({ error: `no mock route: ${path.join("/")}` }, { status: 404 });
}
