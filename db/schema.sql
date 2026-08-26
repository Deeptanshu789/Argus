-- Argus schema. Applied automatically by docker-entrypoint on first `up`,
-- or at any time by `npm run db:setup`, which also seeds the camera topology.
--
-- EVERY statement here is idempotent (IF NOT EXISTS, named indexes,
-- if_not_exists on the hypertables) so re-running it on a live database is
-- safe. The container entrypoint only fires on an EMPTY volume, so without
-- that property a schema change would be unappliable without dropping data.
--
-- Shared contract between the pipeline (writes), the analytics engine (reads),
-- and the API (reads). Change it only by announcing it to both devs.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------- topology --
-- Authored by Dev B (camera placement + road links), consumed by Dev A's
-- cross-camera association engine as its spatial-temporal layer.

CREATE TABLE IF NOT EXISTS cameras (
    id           TEXT PRIMARY KEY,            -- 'CAM1'
    name         TEXT NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    heading_deg  SMALLINT,                    -- direction the camera faces
    source_uri   TEXT                         -- file path or RTSP url
);
-- Speed and wrong-way detection are CALIBRATION, not code: a pixel is not a
-- metre and the ratio differs per camera. Measure metres_per_pixel once against
-- a known landmark (a lane width, a zebra crossing). NULL means uncalibrated,
-- and the analytics layer then declines to claim a speed rather than guessing.
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS metres_per_pixel  REAL;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS expected_flow_deg SMALLINT;

-- Directed road links. travel_time_s is the estimate Layer 3 uses to reject
-- physically impossible matches ("can't be 10 km away in 30 s").
CREATE TABLE IF NOT EXISTS camera_links (
    from_camera    TEXT REFERENCES cameras(id),
    to_camera      TEXT REFERENCES cameras(id),
    distance_m     INTEGER NOT NULL,
    travel_time_s  INTEGER NOT NULL,
    PRIMARY KEY (from_camera, to_camera)
);

-- ------------------------------------------------------------- per-camera --

-- One row per vehicle track within one camera. ByteTrack owns track_id.
CREATE TABLE IF NOT EXISTS tracks (
    id             BIGSERIAL PRIMARY KEY,
    camera_id      TEXT NOT NULL REFERENCES cameras(id),
    track_id       TEXT NOT NULL,             -- 'CAM1_T42'
    vehicle_type   TEXT,                      -- car|bus|truck|motorcycle|auto
    color          TEXT,
    plate_text     TEXT,                      -- best OCR result for the track
    plate_conf     REAL,
    embedding      REAL[],                    -- ReID vector; dimension is whatever
                                              -- the tracker's model emits, not fixed
    color_hist     REAL[],
    entry_time     TIMESTAMPTZ NOT NULL,
    exit_time      TIMESTAMPTZ,
    UNIQUE (camera_id, track_id)
);
CREATE INDEX IF NOT EXISTS tracks_plate ON tracks (plate_text) WHERE plate_text IS NOT NULL;
CREATE INDEX IF NOT EXISTS tracks_cam_exit ON tracks (camera_id, exit_time DESC);

-- Written once, when the track closes and its whole detection history is
-- known. Null when the track is too short to time, or when the camera has no
-- metres-per-pixel survey -- a blank speed is honest, a zero is not.
ALTER TABLE tracks ADD COLUMN IF NOT EXISTS speed_kmh REAL;

-- Per-frame detections. High volume, time-series -> hypertable.
CREATE TABLE IF NOT EXISTS detections (
    ts           TIMESTAMPTZ NOT NULL,
    camera_id    TEXT NOT NULL,
    track_id     TEXT NOT NULL,
    bbox         INTEGER[4] NOT NULL,         -- x1,y1,x2,y2
    conf         REAL,
    speed_kmh    REAL
);
SELECT create_hypertable('detections', 'ts', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS det_cam_ts ON detections (camera_id, ts DESC);
CREATE INDEX IF NOT EXISTS det_track_ts ON detections (track_id, ts);

-- ------------------------------------------------------------ cross-camera --

-- The differentiator. One row per confirmed same-vehicle match between two
-- tracks on different cameras. `method` records which of the 3 layers fired.
CREATE TABLE IF NOT EXISTS matches (
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
CREATE TABLE IF NOT EXISTS trajectories (
    id           BIGSERIAL PRIMARY KEY,
    plate_text   TEXT,
    track_ids    BIGINT[] NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS traj_plate ON trajectories (plate_text);

-- ---------------------------------------------------------------- analytics --

CREATE TABLE IF NOT EXISTS analytics (
    ts               TIMESTAMPTZ NOT NULL,
    camera_id        TEXT NOT NULL,
    vehicle_count    INTEGER NOT NULL DEFAULT 0,
    avg_speed_kmh    REAL,
    congestion_score REAL,                    -- 0-100
    by_type          JSONB                    -- {"car": 12, "bus": 1, ...}
);
SELECT create_hypertable('analytics', 'ts', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS analytics_cam_ts ON analytics (camera_id, ts DESC);

CREATE TABLE IF NOT EXISTS alerts (
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
CREATE INDEX IF NOT EXISTS alerts_open ON alerts (ts DESC) WHERE NOT acked;

-- ------------------------------------------------------------------ uploads --

-- Video uploaded from the operator's machine, analysed by the same sidecar
-- pipeline as a live camera.
--
-- Each file in an upload becomes its own row in `cameras`, so every existing
-- query -- tracks, trajectories, analytics, Module C -- works on uploaded
-- footage with no special case anywhere. The upload tables only record which
-- cameras belong to which upload, which is what lets a results page show one
-- upload's footage and nothing else.
CREATE TABLE IF NOT EXISTS uploads (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    label       TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|error
    -- Expected seconds for a vehicle to travel between the uploaded cameras.
    -- NULL means unknown, and layer 3 of the association engine then abstains
    -- rather than inventing a road graph: see layerTemporal in
    -- src/server/association.ts. Supplying it lets a plate match plus plausible
    -- timing confirm a journey on its own.
    gap_seconds INTEGER,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS uploads_recent ON uploads (created_at DESC);

CREATE TABLE IF NOT EXISTS upload_sources (
    id        BIGSERIAL PRIMARY KEY,
    upload_id BIGINT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL UNIQUE REFERENCES cameras(id),
    filename  TEXT NOT NULL,          -- what the operator called it
    path      TEXT NOT NULL,          -- where it landed on disk
    status    TEXT NOT NULL DEFAULT 'pending',
    error     TEXT
);
CREATE INDEX IF NOT EXISTS upload_sources_upload ON upload_sources (upload_id);

-- Cameras that came from an upload, so the dashboard can leave them out of the
-- live city view without knowing anything about uploads.
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS is_upload BOOLEAN NOT NULL DEFAULT false;

-- ------------------------------------------------------------------ devices --

-- A phone or laptop acting as a camera, paired with a short code.
--
-- Like an upload, a device becomes a row in `cameras` -- but unlike an upload it
-- is a REAL live camera and stays in the city views, because that is what it
-- is. It carries no `camera_links`, so layer 3 of the association engine
-- abstains on it rather than inventing a road graph to footage nobody surveyed.
--
-- Two ways to satisfy one code, recorded in `kind`:
--   browser  the phone's own browser captures and pushes JPEG frames, which the
--            server re-serves as MJPEG for the sidecar to read.
--   url      an IP-camera app on the phone serves RTSP or MJPEG directly, and
--            the sidecar reads that URL. No secure context needed, which is why
--            it exists: getUserMedia is blocked on plain http from a phone.
CREATE TABLE IF NOT EXISTS devices (
    id             BIGSERIAL PRIMARY KEY,
    code           TEXT NOT NULL UNIQUE,      -- what the operator types on the phone
    camera_id      TEXT NOT NULL UNIQUE REFERENCES cameras(id),
    label          TEXT,
    kind           TEXT,                      -- browser|url, NULL until something connects
    source_url     TEXT,                      -- kind=url only
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    paired_at      TIMESTAMPTZ,
    last_frame_at  TIMESTAMPTZ,
    revoked        BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS devices_live ON devices (created_at DESC) WHERE NOT revoked;
