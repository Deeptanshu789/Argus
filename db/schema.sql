-- Argus schema. Applied automatically by docker-entrypoint on first `up`,
-- or by hand: psql "$DATABASE_URL" -f db/schema.sql
--
-- Shared contract between the pipeline (writes), the analytics engine (reads),
-- and the API (reads). Change it only by announcing it to both devs.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------- topology --
-- Authored by Dev B (camera placement + road links), consumed by Dev A's
-- cross-camera association engine as its spatial-temporal layer.

CREATE TABLE cameras (
    id           TEXT PRIMARY KEY,            -- 'CAM1'
    name         TEXT NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    heading_deg  SMALLINT,                    -- direction the camera faces
    source_uri   TEXT                         -- file path or RTSP url
);

-- Directed road links. travel_time_s is the estimate Layer 3 uses to reject
-- physically impossible matches ("can't be 10 km away in 30 s").
CREATE TABLE camera_links (
    from_camera    TEXT REFERENCES cameras(id),
    to_camera      TEXT REFERENCES cameras(id),
    distance_m     INTEGER NOT NULL,
    travel_time_s  INTEGER NOT NULL,
    PRIMARY KEY (from_camera, to_camera)
);

-- ------------------------------------------------------------- per-camera --

-- One row per vehicle track within one camera. ByteTrack owns track_id.
CREATE TABLE tracks (
    id             BIGSERIAL PRIMARY KEY,
    camera_id      TEXT NOT NULL REFERENCES cameras(id),
    track_id       TEXT NOT NULL,             -- 'CAM1_T42'
    vehicle_type   TEXT,                      -- car|bus|truck|motorcycle|auto
    color          TEXT,
    plate_text     TEXT,                      -- best OCR result for the track
    plate_conf     REAL,
    embedding      REAL[],                    -- 512-dim OSNet Re-ID vector
    color_hist     REAL[],
    entry_time     TIMESTAMPTZ NOT NULL,
    exit_time      TIMESTAMPTZ,
    UNIQUE (camera_id, track_id)
);
CREATE INDEX ON tracks (plate_text) WHERE plate_text IS NOT NULL;
CREATE INDEX ON tracks (camera_id, exit_time DESC);

-- Per-frame detections. High volume, time-series -> hypertable.
CREATE TABLE detections (
    ts           TIMESTAMPTZ NOT NULL,
    camera_id    TEXT NOT NULL,
    track_id     TEXT NOT NULL,
    bbox         INTEGER[4] NOT NULL,         -- x1,y1,x2,y2
    conf         REAL,
    speed_kmh    REAL
);
SELECT create_hypertable('detections', 'ts', chunk_time_interval => INTERVAL '1 hour');
CREATE INDEX ON detections (camera_id, ts DESC);
CREATE INDEX ON detections (track_id, ts);

-- ------------------------------------------------------------ cross-camera --

-- The differentiator. One row per confirmed same-vehicle match between two
-- tracks on different cameras. `method` records which of the 3 layers fired.
CREATE TABLE matches (
    id            BIGSERIAL PRIMARY KEY,
    from_track    BIGINT NOT NULL REFERENCES tracks(id),
    to_track      BIGINT NOT NULL REFERENCES tracks(id),
    method        TEXT NOT NULL,              -- plate|reid|spatial_temporal
    confidence    REAL NOT NULL,
    travel_time_s INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_track, to_track)
);

-- A stitched multi-camera journey: an ordered chain of track ids.
CREATE TABLE trajectories (
    id           BIGSERIAL PRIMARY KEY,
    plate_text   TEXT,
    track_ids    BIGINT[] NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON trajectories (plate_text);

-- ---------------------------------------------------------------- analytics --

CREATE TABLE analytics (
    ts               TIMESTAMPTZ NOT NULL,
    camera_id        TEXT NOT NULL,
    vehicle_count    INTEGER NOT NULL DEFAULT 0,
    avg_speed_kmh    REAL,
    congestion_score REAL,                    -- 0-100
    by_type          JSONB                    -- {"car": 12, "bus": 1, ...}
);
SELECT create_hypertable('analytics', 'ts', chunk_time_interval => INTERVAL '1 day');
CREATE INDEX ON analytics (camera_id, ts DESC);

CREATE TABLE alerts (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    camera_id   TEXT REFERENCES cameras(id),
    kind        TEXT NOT NULL,                -- stationary|wrong_way|volume_spike|watchlist
    severity    TEXT NOT NULL DEFAULT 'warn', -- info|warn|critical
    track_id    TEXT,
    plate_text  TEXT,
    detail      TEXT,
    acked       BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX ON alerts (ts DESC) WHERE NOT acked;
