#!/usr/bin/env bash
#
# Fill in .env.production with real secrets, and check the machine can actually
# run the stack.
#
#     ./scripts/deploy-prepare.sh                 # IP address only
#     ./scripts/deploy-prepare.sh argus.example.com you@example.com
#
# Idempotent: an existing .env.production is left alone except for values that
# are still blank, so re-running it after adding a domain does not roll your
# database password and lock you out of your own data.
set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-}"
ENV_FILE=".env.production"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\n\033[31mstopped: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "Checking the machine"

command -v docker >/dev/null || die "docker not installed"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not available"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (sudo, or add yourself to the docker group)"

mem_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
cpus=$(nproc)
echo "cpus $cpus, memory ${mem_gb} GB"
# Measured on the build machine: YOLOv8n plus BoT-SORT with ReID is ~17 ms a
# frame at imgsz=480 on 16 threads. Two cores will keep up with one or two
# streams at 5 fps and no more.
[ "$cpus" -ge 2 ] || echo "  WARNING: 1 core. Inference will not keep up with even one stream."
[ "$mem_gb" -ge 4 ] || echo "  WARNING: under 4 GB. YOLO, PaddleOCR and Postgres together will be tight."

free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk ${free_gb} GB"
[ "$free_gb" -ge 10 ] || echo "  WARNING: under 10 GB. The images alone are several GB, before any uploads."

# ---------------------------------------------------------------------------
step "Writing $ENV_FILE"

[ -f "$ENV_FILE" ] || { cp .env.production.example "$ENV_FILE"; echo "created from the example"; }

# Reads a key, and fills it only when it is still empty. Never overwrites.
fill() {
  local key="$1" value="$2"
  local current
  current=$(grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d= -f2-)
  if [ -n "$current" ]; then
    echo "$key already set, left alone"
    return
  fi
  # The value can contain / and $, so use a different sed delimiter and escape.
  local escaped=${value//\\/\\\\}
  escaped=${escaped//|/\\|}
  sed -i "s|^${key}=.*|${key}=${escaped}|" "$ENV_FILE"
  echo "$key set"
}

fill POSTGRES_PASSWORD "$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)"

if [ -z "$(grep -E '^ARGUS_PASSWORD_HASH=' "$ENV_FILE" | cut -d= -f2-)" ]; then
  OPERATOR_PASSWORD=$(openssl rand -base64 18 | tr -d '/+=' | head -c 20)
  HASH=$(docker run --rm caddy caddy hash-password --plaintext "$OPERATOR_PASSWORD")
  # A bcrypt hash is full of dollar signs ($2a$14$...) and docker compose
  # interpolates the value it reads from an env file. Unescaped, everything
  # after the third $ is read as a variable name, substituted with nothing, and
  # the hash silently arrives at Caddy corrupted -- basic auth then rejects the
  # correct password with no clue why. Doubling each $ makes compose emit one.
  fill ARGUS_PASSWORD_HASH "${HASH//\$/\$\$}"
  echo
  echo "  ┌──────────────────────────────────────────────────────┐"
  printf  "  │ operator password: %-33s │\n" "$OPERATOR_PASSWORD"
  echo "  └──────────────────────────────────────────────────────┘"
  echo "  WRITE THIS DOWN. Only the hash is stored; it cannot be recovered."
else
  echo "ARGUS_PASSWORD_HASH already set, left alone"
fi

if [ -n "$DOMAIN" ]; then
  [ -n "$EMAIL" ] || die "a domain needs an email for Let's Encrypt: $0 $DOMAIN you@example.com"
  sed -i "s|^ARGUS_SITE=.*|ARGUS_SITE=${DOMAIN}|" "$ENV_FILE"
  sed -i "s|^ARGUS_TLS=.*|ARGUS_TLS=${EMAIL}|" "$ENV_FILE"
  echo "ARGUS_SITE=$DOMAIN with a Let's Encrypt certificate"
else
  # Caddy needs a NAME to issue a certificate for. A bare `:443` fails every
  # TLS handshake with an error that mentions neither certificates nor config.
  ip_addr=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)
  [ -n "$ip_addr" ] || ip_addr=localhost
  sed -i "s|^ARGUS_SITE=.*|ARGUS_SITE=${ip_addr}|" "$ENV_FILE"
  sed -i "s|^ARGUS_TLS=.*|ARGUS_TLS=internal|" "$ENV_FILE"
  echo "no domain given: ARGUS_SITE=$ip_addr with a self-signed certificate"
  echo "  Browsers will warn once. Behind NAT this is the INTERNAL address —"
  echo "  set ARGUS_SITE in $ENV_FILE to whatever the outside world reaches."
fi

# ---------------------------------------------------------------------------
step "Checking the trained weights"

# runs/ is gitignored — weights are large and never committed — so a fresh
# clone has none, and the worker mounts the directory read-only. Without them
# the sidecar still tracks vehicles but reads far fewer plates, because OCR
# then runs on the whole vehicle crop instead of a detected plate.
# Both models are checked, and by the paths ml/sidecar.py actually defaults to.
# Checking only the old detector path reported a healthy deploy while the box
# ran a different pipeline from the build machine: an older detector, and
# PaddleOCR instead of the trained reader, with nothing in any log to say so.
weights_ok=1
if [ -d runs/detect/plate-k12/weights/best_openvino_model ]; then
  echo "runs/detect/plate-k12/weights/best_openvino_model present"
elif [ -d runs/detect/plate/weights/best_openvino_model ]; then
  echo "  NOTE: only the older runs/detect/plate detector is here."
  echo "  It works. It is not what this build measures against."
  weights_ok=0
else
  echo "  WARNING: no plate detector found."
  echo "  Without one the sidecar runs OCR on the whole vehicle crop and"
  echo "  reads far fewer plates."
  weights_ok=0
fi

if [ -f runs/reader-k12/best.pt ]; then
  echo "runs/reader-k12/best.pt present"
else
  echo "  NOTE: no trained reader; the sidecar will fall back to PaddleOCR."
  echo "  That is the higher-precision reader, so this is safe -- but it is"
  echo "  not the pipeline the build machine is running."
  weights_ok=0
fi

if [ "$weights_ok" = 0 ]; then
  mkdir -p runs
  echo
  echo "  Copy them up from the build machine:"
  echo
  echo "      rsync -av runs/detect/plate-k12/ USER@THIS_HOST:$(pwd)/runs/detect/plate-k12/"
  echo "      rsync -av runs/reader-k12/      USER@THIS_HOST:$(pwd)/runs/reader-k12/"
fi

# ---------------------------------------------------------------------------
step "Validating the stack"
docker compose -f docker-compose.prod.yml --env-file "$ENV_FILE" config >/dev/null
echo "docker-compose.prod.yml is valid"

cat <<EOF

Ready. Bring it up:

    docker compose -f docker-compose.prod.yml --env-file $ENV_FILE up -d --build

Then seed the camera topology, once:

    docker compose -f docker-compose.prod.yml --env-file $ENV_FILE exec app npm run db:setup

Open the site, and expect a browser password prompt. DEPLOY.md has the rest,
including the firewall and what to check when something does not come up.
EOF
