# Deploying Argus to a VPS

Four containers, one command. Caddy terminates TLS and holds the only password;
the app, the worker and the database talk over a private network and are not
reachable from the internet.

```
internet ──▶ caddy :80 :443 ──▶ app :3000 ──┐
             TLS + password                  ├──▶ db :5432   (not published)
                              worker ────────┘
                              (YOLO + PaddleOCR, CPU)
```

---

## Before you start

**Read this part.** Two things about this deployment are easy to get wrong and
expensive to get wrong.

**Argus authenticates nobody.** There is no login in the application. Caddy's
basic auth is the whole of it, and it covers the dashboard, the API, the event
WebSocket and uploads. `docker-compose.prod.yml` deliberately does **not**
publish port 3000 — publishing it would route straight around the password.

**The pair links are exempt on purpose.** `/cam/<code>` and `/ws/cam` are open
so a phone can join by link alone without typing an operator password on a
handset. The six-character code is then the only thing guarding frame upload.
That lets someone who guesses a code push video *into* a camera slot; it never
lets them read a plate, a track or a journey back out. If that trade is wrong
for you, delete the `@pair` handle block in `Caddyfile` and the phone will
prompt for the password like everything else.

**This repository is public.** Never commit `.env.production`. It is gitignored,
and anything pushed to a public repo stays in the history even after deletion.

### Sizing

| | minimum | comfortable |
|---|---|---|
| vCPU | 2 | 4 |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB | 40 GB+ if operators upload video |

Inference is the constraint and it is CPU-only — there is no GPU path in this
build. Measured on the build machine: YOLOv8n with BoT-SORT and ReID runs about
17 ms a frame at `imgsz=480`. At 5 fps that is one stream per core with room to
spare, and roughly four streams on four cores. `WORKER_CPUS` caps the worker so
it cannot starve the web tier.

---

## 1. Get the code onto the box

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER"   # then log out and back in
git clone https://github.com/Deeptanshu789/Argus.git
cd Argus
```

**Check:**

```bash
docker compose version && docker info >/dev/null && echo ok
```

`permission denied` on the docker socket means the group change has not taken
effect yet — log out and back in.

## 2. Copy the trained weights up

`runs/` is gitignored: weights are large and a `.pt` is never committed. A fresh
clone has none, and the worker mounts that directory read-only.

Without the plate detector the sidecar still tracks vehicles, but OCR then runs
on the whole vehicle crop and reads far fewer plates. Two directories, because
`ml/sidecar.py` now defaults to a trained reader as well as a trained detector.
From the build machine:

```bash
rsync -av runs/detect/plate-k12/ USER@VPS:/home/USER/Argus/runs/detect/plate-k12/
rsync -av runs/reader-ft/        USER@VPS:/home/USER/Argus/runs/reader-ft/
```

The reader directory is whichever one `_READER_DEFAULT` in `ml/sidecar.py`
names — it has moved between trained readers already, and
`./scripts/deploy-prepare.sh` reads the path from that file rather than
keeping a second copy of it.

**Check**, on the VPS:

```bash
ls runs/detect/plate-k12/weights/best_openvino_model runs/reader-ft/best.pt
```

The OpenVINO directory is the one the sidecar loads. A bare `best.pt` also
works but is slower.

Copy neither and the box still runs — an untrained detector, and PaddleOCR
instead of the CRNN. Nothing in any log says the pipeline differs from the one
you measured, which is why `./scripts/deploy-prepare.sh` checks both paths by
name. To run PaddleOCR deliberately rather than by accident, set
`ARGUS_READER=paddle` on the worker service.

## 3. Generate the secrets

```bash
./scripts/deploy-prepare.sh                            # IP address only
./scripts/deploy-prepare.sh argus.example.com you@example.com   # with a domain
```

It writes `.env.production`, prints an operator password **once**, and checks
the machine has the CPU, memory and disk to run this.

Idempotent: re-running it fills only what is still blank, so adding a domain
later does not roll your database password and lock you out of your own data.

**Check:**

```bash
grep -c . .env.production                 # non-empty
docker compose -f docker-compose.prod.yml --env-file .env.production config >/dev/null && echo valid
```

### About the certificate

| you have | `ARGUS_SITE` | `ARGUS_TLS` | phone camera |
|---|---|---|---|
| a domain pointing here | `argus.example.com` | `you@example.com` | works, no warning |
| an IP address only | `203.0.113.10` | `internal` | works after accepting a warning once |

`ARGUS_SITE` must be a **hostname or an IP**, never a bare `:443`. A site
address with no name gives Caddy nothing to put in a certificate, and every
connection then dies in the TLS handshake with `tlsv1 alert internal error` —
an error that mentions neither the certificate nor the line that caused it.
The script fills this in with the address it detects; correct it by hand if the
box is behind NAT and its interface address is not what the outside world
reaches.

Let's Encrypt will not issue a certificate for a bare IP, so the second row is
not a shortcut — it is the only option until a domain points here. HTTPS is
required either way: a phone will not hand its camera to a plain-http page, and
the pair flow simply does not work without it.

## 4. Open the firewall

```bash
sudo ufw allow 22/tcp && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp
sudo ufw --force enable
```

Port 80 is needed even with a domain: Let's Encrypt validates over it.

**Check:**

```bash
sudo ufw status | grep -E '80|443'
```

## 5. Bring it up

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

The first build downloads PyTorch, OpenVINO and PaddleOCR. Ten to twenty
minutes on a small box is normal, and the worker image is the slow half. The
finished images are roughly 2 GB for the app and several GB for the worker, so
this is where the 20 GB disk minimum goes.

**Check:**

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production ps
```

All four `running`, with `db` and `app` healthy. If `app` is restarting, read
its log first — it exits deliberately when it cannot reach the database.

## 6. Seed the camera topology

The database container applies `db/schema.sql` only on an empty volume, and
that creates tables but no rows. The camera graph comes from `db/setup.ts`:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production \
  exec app npm run db:setup
```

**Check** — it prints what it found:

```
seeded 4 cameras, 6 links
  cameras       4
  camera_links  6
```

Every statement in `db/schema.sql` is idempotent, so this is always safe to
re-run, and it is how a schema change is applied later.

The worker re-reads the camera graph every fifteen seconds, so it picks this up
on its own — there is no requirement to seed before starting it.

## 7. Verify it actually works

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production \
  exec app npm run smoke -- --no-pipeline
```

Runs inside the container against `localhost:3000`, so it bypasses Caddy and
tests the application rather than the proxy. Expect every check to pass.
`--no-pipeline` skips the part that spawns sidecars, which belongs to the
worker container.

Then from your laptop:

```bash
curl -sk https://YOUR_HOST/api/health          # expect 401
curl -sk -u argus:YOUR_PASSWORD https://YOUR_HOST/api/health
```

The first must be **401**. If it returns `{"ok":true}` the password is not
being applied and the deployment is open — stop and fix `Caddyfile` before
going further.

## 8. Pair a phone

Open `https://YOUR_HOST/devices`, sign in, and issue a code. Open the pair link
on the phone — no password there, by design — and allow the camera.

**Check:** the device turns `live` within a few seconds, and `PHONE1` appears on
the dashboard beside the other cameras.

If it stays `waiting`, the phone reached the page but the camera was refused.
With `ARGUS_TLS=internal` you must accept the certificate warning first; that
warning is exactly what makes the origin secure enough for the browser to hand
over the camera.

---

## Running it

```bash
# a shorthand worth keeping
alias argus='docker compose -f docker-compose.prod.yml --env-file .env.production'

argus logs -f worker      # matches, alerts, sidecar output
argus logs -f app         # requests and the event bus
argus restart worker      # after changing weights
argus down                # stop, keeping the data volumes
```

### Updating

```bash
git pull
argus up -d --build
argus exec app npm run db:setup      # picks up any schema change
```

### Backups

The database holds registration numbers and camera positions. The volume is
`argus_pgdata`.

```bash
argus exec -T db pg_dump -U argus argus | gzip > argus-$(date +%F).sql.gz
```

Restore into an empty database:

```bash
gunzip -c argus-2026-08-26.sql.gz | argus exec -T db psql -U argus argus
```

Uploaded video lives in the `argus_uploads` volume and is not in that dump.
Nothing prunes it, either — see below.

---

## When something is wrong

**`app` restarts in a loop.** It exits on purpose when the database is
unreachable. `argus logs app` names the reason; `argus logs db` usually has it.

**The dashboard is empty and no vehicles appear.** Nothing is feeding it. On a
VPS with no camera files that is correct — upload a video at `/upload` or pair
a phone at `/devices`. `ARGUS_CAMERAS` is only for RTSP sources the VPS can
reach.

**Uploads sit at `pending` forever.** The worker claims them, so it is not
running. `argus ps` and `argus logs worker`.

**A paired phone shows `stale` and the worker keeps restarting a sidecar.** The
stream ended. The backoff runs 5s, 10s, 20s to a minute, so this is loud in the
log but harmless; it recovers on its own when the phone comes back.

**Plates are rarely read.** Check step 2. `argus logs worker | grep "plate
detector"` — if it says no trained weights were found, the mount is empty.

**The disk fills.** Uploaded video is the only thing here that grows without
bound. Prune it:

```bash
argus exec app npx tsx scripts/prune-uploads.ts --days 7           # report
argus exec app npx tsx scripts/prune-uploads.ts --days 7 --apply   # delete
```

It removes the source video of finished uploads older than `--days`, plus any
file with no upload row at all. It KEEPS the plates, tracks, detections and
journeys — those are rows, they are small, and they are what the system is for.
The results page keeps working; only the original footage goes.

An upload still `pending` or `running` is never touched, however old, because
the worker may be reading it.

Weekly, on the host:

```bash
(crontab -l 2>/dev/null; echo "0 4 * * 0 cd $PWD && docker compose -f docker-compose.prod.yml --env-file .env.production exec -T app npx tsx scripts/prune-uploads.ts --days 7 --apply") | crontab -
```

**Every connection fails and `curl` reports `tlsv1 alert internal error`.**
`ARGUS_SITE` is a bare `:443` or empty, so Caddy has no name to issue a
certificate for. Set it to the hostname or IP you reach the box at, then
`argus up -d caddy`.

**The build fails installing torch**, with `No matching distribution found for
flit_core`. Debian's pip is too old and rejects a wheel it should accept, then
cannot fetch the build dependency to compile the fallback. `worker/Dockerfile`
upgrades pip before installing torch for exactly this reason — if you have
edited that file, put the upgrade back.

**Everything is slow and the dashboard lags.** Inference is starving the web
tier. Lower `WORKER_CPUS`, or cut `imgsz` before cutting features — see
CLAUDE.md, "CPU inference budget".

---

## What is exposed

| path | reachable without the password | why |
|---|---|---|
| `/cam/<code>`, `/ws/cam` | yes | a phone pairs by link; the code is the credential |
| `GET /api/devices/<code>` | yes | the one lookup the pair page makes |
| `/_next/*`, `/favicon.ico` | yes | static assets the pair page needs |
| everything else | no | dashboard, API, event WebSocket, uploads |

Port 3000 is never published. Port 5432 is never published. Only Caddy binds a
public port at all.
