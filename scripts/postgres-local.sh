#!/usr/bin/env bash
#
# Install and configure Postgres 16 + TimescaleDB natively on Fedora, so Argus
# needs no container and no message broker.
#
#     sudo ./scripts/postgres-local.sh
#
# Idempotent: every step checks whether it has already been done. Re-running it
# after a partial failure is safe and is the intended way to recover.
#
# What it does NOT do: touch any existing database's contents, or stop a
# container that is holding port 5432. Both are your call, and the script says
# so and stops rather than deciding for you.
set -euo pipefail

DB_NAME=argus
DB_USER=argus
DB_PASS=argus
PGDATA=/var/lib/pgsql/data
CONF="$PGDATA/postgresql.conf"
HBA="$PGDATA/pg_hba.conf"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\n\033[31mstopped: %s\033[0m\n' "$1" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run me with sudo"

# ---------------------------------------------------------------------------
step "Checking port 5432"

# A running container on 5432 will make postgresql.service fail to bind, and
# the error it prints does not mention containers at all.
if ss -ltn | grep -q ':5432 '; then
  if systemctl is-active --quiet postgresql; then
    echo "postgresql.service already owns 5432 — continuing"
  else
    echo "Something else is listening on 5432. If it is the Argus container:"
    echo
    echo "    docker context use default"
    echo "    docker compose down          # stops it; the pgdata volume survives"
    echo
    die "free port 5432, then run this again"
  fi
fi

# ---------------------------------------------------------------------------
step "Installing packages"
dnf install -y postgresql-server postgresql-contrib timescaledb

# ---------------------------------------------------------------------------
step "Initialising the data directory"
if [ -f "$PGDATA/PG_VERSION" ]; then
  echo "$PGDATA already initialised (PG_VERSION $(cat "$PGDATA/PG_VERSION")) — left alone"
else
  postgresql-setup --initdb
fi

# ---------------------------------------------------------------------------
step "Enabling TimescaleDB"

# TimescaleDB must be loaded at server start; CREATE EXTENSION alone is not
# enough and fails with a message telling you exactly this.
if grep -qE "^shared_preload_libraries *=.*timescaledb" "$CONF"; then
  echo "already in shared_preload_libraries"
else
  cp "$CONF" "$CONF.argus-backup.$(date +%s)"
  sed -i "/^#*shared_preload_libraries/d" "$CONF"
  echo "shared_preload_libraries = 'timescaledb'   # added by scripts/postgres-local.sh" >> "$CONF"
  echo "added"
fi

# ---------------------------------------------------------------------------
step "Allowing password logins from localhost"

# Fedora ships `ident` for host connections, which rejects the password in
# DATABASE_URL. postgres.js connects over TCP, so this has to change or every
# query fails with "no pg_hba.conf entry for host".
if grep -qE "^host +all +all +127\.0\.0\.1/32 +scram-sha-256" "$HBA"; then
  echo "already scram-sha-256"
else
  cp "$HBA" "$HBA.argus-backup.$(date +%s)"
  sed -i -E "s|^(host +all +all +(127\.0\.0\.1/32\|::1/128) +).*|\1scram-sha-256|" "$HBA"
  echo "set to scram-sha-256 (backup written next to it)"
fi

# ---------------------------------------------------------------------------
step "Starting Postgres"
systemctl enable --now postgresql
systemctl restart postgresql          # picks up shared_preload_libraries
systemctl is-active --quiet postgresql || die "postgresql.service failed to start
  journalctl -u postgresql -n 30 --no-pager"

# Wait for it to accept connections rather than racing the next step.
for _ in $(seq 1 30); do
  sudo -u postgres pg_isready -q && break
  sleep 1
done
sudo -u postgres pg_isready || die "postgres is up but not accepting connections"

# ---------------------------------------------------------------------------
step "Creating the role and database"
psql_su() { sudo -u postgres psql -v ON_ERROR_STOP=1 "$@"; }

if [ "$(psql_su -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'")" = "1" ]; then
  echo "role $DB_USER exists"
else
  psql_su -c "CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS'"
  echo "role $DB_USER created"
fi

if [ "$(psql_su -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")" = "1" ]; then
  echo "database $DB_NAME exists — contents untouched"
else
  psql_su -c "CREATE DATABASE $DB_NAME OWNER $DB_USER"
  echo "database $DB_NAME created"
fi

psql_su -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS timescaledb" >/dev/null
psql_su -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER" >/dev/null

# ---------------------------------------------------------------------------
step "Verifying"
ver=$(psql_su -d "$DB_NAME" -tAc \
      "SELECT extversion FROM pg_extension WHERE extname='timescaledb'")
[ -n "$ver" ] || die "timescaledb extension did not install"
echo "postgres  $(psql_su -tAc 'SHOW server_version')"
echo "timescale $ver"

# The real test: connect the way the application does, over TCP with a
# password. Everything above can pass while this still fails.
PGPASSWORD="$DB_PASS" psql -h 127.0.0.1 -U "$DB_USER" -d "$DB_NAME" \
  -tAc "SELECT 'application login ok'" \
  || die "the app's own connection string does not work; check $HBA"

cat <<EOF

Done. Next, as your normal user:

    npm run db:setup     # applies db/schema.sql and seeds the camera topology
    npm run dev          # http://localhost:3000

DATABASE_URL defaults to postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME,
so nothing needs configuring. There is no Redis any more: worker-to-dashboard
events go over Postgres LISTEN/NOTIFY (src/server/bus.ts).
EOF
