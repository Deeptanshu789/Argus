# 🎯 SIH26127 — City-Wide ANPR & Trajectory Tracking
## Complete Implementation Plan

> **PS Code:** SIH26127 | **Org:** Bharat Electronics Limited (BEL) | **Theme:** Smart Automation | **Track:** Software | **Prize:** ₹1,00,000

---

## 1. Problem Statement Breakdown

### What BEL Wants (Decoded)
BEL needs a unified AI engine that does **3 things simultaneously**:

| Capability | What It Means | Difficulty |
|---|---|---|
| **ANPR** | Detect license plates from camera feeds → OCR to extract plate text (e.g., KA 01 MB 1234) | ⭐⭐⭐ |
| **Trajectory Tracking** | Track the SAME vehicle across DIFFERENT cameras placed across a city (cross-camera re-identification) | ⭐⭐⭐⭐⭐ |
| **Traffic Analytics** | Real-time congestion heatmaps, vehicle counts, speed estimation, flow patterns, anomaly detection | ⭐⭐⭐⭐ |

> [!IMPORTANT]
> **The differentiator is #2 — Trajectory Tracking.** Any team can do basic ANPR. The cross-camera re-identification + trajectory stitching is what separates winners from also-rans. This is where you invest 40% of your effort.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + deck.gl)                │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Live Feed │  │  City Map    │  │   Analytics Dashboard  │ │
│  │ Grid View │  │  (Trajectories│  │  (Charts, Heatmaps,   │ │
│  │ (4 cams)  │  │   + Heatmap) │  │   Alerts, Reports)    │ │
│  └─────┬─────┘  └──────┬───────┘  └───────────┬────────────┘ │
│        └────────────────┼──────────────────────┘              │
│                         │ WebSocket + REST                    │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│                  BACKEND (Python FastAPI)                     │
│  ┌──────────────────────┼──────────────────────────────┐     │
│  │            API Gateway + WebSocket Hub               │     │
│  └──────────────────────┼──────────────────────────────┘     │
│         ┌───────────────┼───────────────┐                    │
│         ▼               ▼               ▼                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐           │
│  │  ANPR      │  │  Tracker   │  │  Analytics   │           │
│  │  Service   │  │  Service   │  │  Service     │           │
│  │            │  │            │  │              │           │
│  │ YOLOv8    │  │ ByteTrack  │  │ Aggregation  │           │
│  │ + PaddleOCR│  │ + Re-ID    │  │ + Anomaly    │           │
│  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘           │
│        └────────────────┼───────────────┘                    │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │  Cross-Camera       │                         │
│              │  Association Engine │                         │
│              │  (Re-ID + Spatial-  │                         │
│              │   Temporal Graph)   │                         │
│              └─────────┬───────────┘                         │
│                        ▼                                     │
│         ┌──────────────────────────────┐                     │
│         │   PostgreSQL + TimescaleDB   │                     │
│         │   (Detections, Trajectories, │                     │
│         │    Analytics Time-Series)    │                     │
│         └──────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. The 5 Core Modules (Detailed)

### Module A: ANPR Engine
**Goal:** Camera frame → detected plate → extracted text in <100ms

| Component | Technology | Details |
|---|---|---|
| Vehicle Detection | YOLOv8-L | Detect vehicles + classify type (car/bus/truck/bike) |
| Plate Detection | YOLOv8-S (custom trained) | Detect plate bounding box within vehicle crop |
| OCR | PaddleOCR v2 | Extract text from plate crop. PaddleOCR beats EasyOCR on Indian plates |
| Post-Processing | Regex + State Code Validation | Validate format `XX 00 XX 0000`, correct common OCR errors (O↔0, I↔1, B↔8) |

**Indian Plate Formats to Handle:**
- White plate (private): `KA 01 MB 1234`
- Yellow plate (commercial): `MH 04 AQ 5678`
- Green plate (EV): `DL 01 SG 0001`
- HSRP plates (standardized font + hologram)
- Old vs new format, fancy plates, dirty/occluded plates

**Training Data:**
- [DataCluster Labs Indian Plates](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset) — 15K+ images
- [Kaggle Indian Number Plates](https://www.kaggle.com/datasets/praveen12345/indian-number-plate-detection) — 15K+ annotated for YOLO
- [Roboflow Universe](https://universe.roboflow.com/search?q=indian%20number%20plate) — community datasets, pre-formatted

---

### Module B: Within-Camera Vehicle Tracking
**Goal:** Track each vehicle frame-to-frame within a single camera's feed

| Component | Technology | Why |
|---|---|---|
| Tracker | **ByteTrack** | Fastest MOT tracker, handles occlusions well, associates even low-confidence detections |
| Feature Extractor | **OSNet-x1.0** | Lightweight Re-ID backbone that produces 512-dim appearance embeddings per vehicle |
| Output | Track ID + bounding box + embedding per frame | Fed into cross-camera engine |

**Each tracked vehicle gets:**
```json
{
  "track_id": "CAM1_T42",
  "camera_id": "CAM1",
  "plate_text": "KA01MB1234",
  "vehicle_type": "car",
  "color": "white",
  "embedding": [0.12, -0.45, ...],  // 512-dim Re-ID vector
  "timestamps": ["2026-08-24T10:00:01", ...],
  "bboxes": [[x1,y1,x2,y2], ...],
  "entry_time": "2026-08-24T10:00:01",
  "exit_time": "2026-08-24T10:00:08"
}
```

---

### Module C: Cross-Camera Association Engine ⭐ (THE DIFFERENTIATOR)
**Goal:** Match vehicle from Camera 1 with the same vehicle appearing in Camera 3, even if it looks different due to angle/lighting

**Three-Layer Matching Strategy:**

```
Layer 1: HARD MATCH (Plate Text)
   ├── If plate OCR text matches exactly → confirmed same vehicle
   └── Handles 60-70% of cases

Layer 2: SOFT MATCH (Appearance Re-ID)
   ├── Compare 512-dim embeddings using cosine similarity
   ├── Threshold: similarity > 0.75 → likely same vehicle
   ├── Uses color histogram as secondary feature
   └── Handles 20-25% of cases (occluded/unreadable plates)

Layer 3: SPATIAL-TEMPORAL REASONING
   ├── Road network graph with travel time estimates between cameras
   ├── If vehicle exits CAM1 heading north, it should appear at CAM3
   │   within 5-15 minutes (based on road distance + speed limits)
   ├── Reject matches that violate physics (can't be 10km away in 30s)
   └── Handles edge cases and resolves ambiguous soft matches
```

**Implementation:**
```python
# Pseudocode for cross-camera association
def associate_cross_camera(exit_event, candidate_entries):
    matches = []
    for entry in candidate_entries:
        # Layer 1: Plate match
        if exit_event.plate == entry.plate and both_confident:
            return MatchResult(entry, confidence=0.99, method="plate")
        
        # Layer 2: Appearance similarity
        sim = cosine_similarity(exit_event.embedding, entry.embedding)
        color_sim = histogram_similarity(exit_event.color_hist, entry.color_hist)
        
        # Layer 3: Spatial-temporal feasibility
        travel_time = road_graph.estimate_travel_time(
            exit_event.camera, entry.camera
        )
        actual_time = entry.timestamp - exit_event.timestamp
        time_feasible = 0.5 * travel_time <= actual_time <= 2.0 * travel_time
        
        if sim > 0.75 and time_feasible:
            combined_score = 0.6*sim + 0.2*color_sim + 0.2*time_score
            matches.append(MatchResult(entry, combined_score, "re-id"))
    
    return best_match(matches)
```

---

### Module D: Traffic Analytics Engine
**Goal:** Real-time city-wide traffic intelligence from all camera feeds

| Analytic | How It's Computed | Output |
|---|---|---|
| **Vehicle Count** | Count unique track IDs per camera per time window | Time-series chart |
| **Vehicle Classification** | YOLO class labels (car/bus/truck/bike/auto) | Pie chart per camera |
| **Speed Estimation** | Distance between known landmarks ÷ time between frames | Avg speed per road segment |
| **Congestion Score** | Vehicle density × inverse speed, normalized 0-100 | Color-coded road segments |
| **Flow Direction** | Entry/exit direction per camera (N/S/E/W) | Animated flow arrows on map |
| **Anomaly Detection** | Stationary vehicle > 5min, wrong-way travel, unusual volume spike | Real-time alerts |
| **Origin-Destination Matrix** | Cross-camera trajectories aggregated by zone | Chord diagram |

---

### Module E: Visualization Dashboard
**Goal:** A command-center style dashboard that makes BEL judges say "we want to deploy this"

**Four Main Views:**

#### View 1: Live Camera Grid
- 2×2 or 3×3 grid of camera feeds with real-time ANPR overlay
- Each detected vehicle gets a bounding box + plate text + vehicle type label
- Color-coded by tracking status (green = tracked, yellow = new, red = lost)

#### View 2: City Map (THE HERO VIEW)
- Interactive 3D map using **deck.gl + MapLibre GL JS**
- **TripsLayer**: Animated vehicle trajectories — lines that "draw themselves" along roads
- **HeatmapLayer**: Traffic congestion heatmap updating every 5 seconds
- **IconLayer**: Camera positions with status indicators
- Click on any trajectory → see plate number, timestamps, full route

#### View 3: Analytics Dashboard
- Real-time charts (Recharts/Chart.js): vehicle count, speed distribution, congestion trends
- Vehicle type breakdown (pie), hourly flow patterns (bar), congestion timeline (area)
- Top 10 busiest intersections leaderboard

#### View 4: Vehicle Search
- Search by plate number → see full trajectory history on map
- "Last seen" location + timestamp
- Useful for law enforcement (stolen vehicle tracking)

---

## 4. Complete Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Frontend** | React 18 | Latest | UI framework |
| | deck.gl | 9.x | GPU-accelerated geospatial visualization |
| | MapLibre GL JS | 4.x | Open-source 3D map (no Mapbox API key needed) |
| | Recharts | 2.x | Analytics charts |
| | WaveSurfer-like video | — | Camera feed display |
| **Backend** | Python FastAPI | 0.110+ | REST + WebSocket API |
| | Redis | 7.x | Real-time state cache, pub/sub for live updates |
| | PostgreSQL + TimescaleDB | 16 + 2.x | Persistent storage + time-series analytics |
| | Celery | 5.x | Async task queue for heavy processing |
| **ML/CV** | YOLOv8 (Ultralytics) | Latest | Vehicle + plate detection |
| | PaddleOCR | 2.7+ | License plate text extraction |
| | OSNet (torchreid) | — | Vehicle Re-ID feature extraction |
| | ByteTrack | — | Multi-object tracking |
| | OpenCV | 4.9+ | Video processing, image transforms |
| | PyTorch | 2.3+ | Model inference runtime |
| **Infra** | Docker Compose | — | Local multi-container deployment |
| | NVIDIA TensorRT | — | Model optimization for GPU inference (optional) |

---

## 5. Datasets You Need

| Dataset | What | Size | Link |
|---|---|---|---|
| **Indian Number Plates** | Plate detection training | 15K+ images | [HuggingFace](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset) |
| **VeRi-776** | Vehicle Re-ID training/eval | 50K images, 776 vehicles, 20 cameras | [GitHub](https://github.com/JDAI-CV/VeRidataset) |
| **VehicleID** | Vehicle Re-ID (larger) | 220K images | Academic request |
| **UA-DETRAC** | Multi-vehicle tracking benchmark | 10hrs video, 140K frames | [Website](https://detrac-db.rit.albany.edu/) |
| **AI City Challenge** | Multi-camera vehicle tracking | City-scale video | [CVPR Challenge](https://www.aicitychallenge.org/) |
| **Demo Video** | Indian traffic footage for live demo | Any | YouTube/record yourself |

---

## 6. Granular 36-Hour Build Plan

### Phase 1: Foundation (Hours 0-10)

| Hour | Task | Who | AI Co-pilot |
|---|---|---|---|
| 0-2 | Project scaffold: FastAPI backend, React frontend, Docker Compose, DB schema | Member 1 | Claude Opus |
| 0-2 | Download datasets: Indian plates, VeRi-776, demo traffic videos | Member 2 | Gemini 1 |
| 2-4 | Train YOLOv8 plate detector on Indian dataset (fine-tune from COCO pretrained) | Member 3 | Gemini 2 |
| 2-4 | Set up PaddleOCR pipeline with Indian plate regex validation | Member 4 | Gemini 3 |
| 4-6 | Integrate YOLO + PaddleOCR into single ANPR pipeline, test on sample frames | Member 3+4 | Gemini 2+3 |
| 4-6 | Set up ByteTrack + OSNet Re-ID feature extractor for single-camera tracking | Member 5 | Gemini 4 |
| 6-10 | **Integration checkpoint**: Camera feed → YOLO detection → plate OCR → ByteTrack → tracked vehicle with plate text and embedding. This MUST work before moving on. | All | All |

### Phase 2: Cross-Camera Intelligence (Hours 10-20)

| Hour | Task | Who | AI Co-pilot |
|---|---|---|---|
| 10-14 | Build cross-camera association engine (3-layer matching: plate → Re-ID → spatial-temporal) | Member 5+6 | Claude Opus |
| 10-14 | Define camera network graph (camera locations, road connections, estimated travel times) | Member 2 | Gemini 1 |
| 14-16 | Test cross-camera matching with 2 demo videos simulating 2 cameras on the same route | Member 5 | Gemini 4 |
| 14-18 | Build analytics engine: vehicle count, classification breakdown, speed estimation, congestion scoring | Member 4 | Gemini 3 |
| 18-20 | Build WebSocket pipeline: backend pushes live detections + analytics to frontend | Member 1 | Claude Opus |

### Phase 3: Dashboard (Hours 20-30)

| Hour | Task | Who | AI Co-pilot |
|---|---|---|---|
| 20-24 | **City Map View**: deck.gl TripsLayer (animated trajectories) + HeatmapLayer (congestion) on MapLibre | Member 2 | Gemini 1 |
| 20-24 | **Camera Grid View**: 4-camera live feed grid with ANPR overlay (bounding boxes, plate text) | Member 3 | Gemini 2 |
| 24-27 | **Analytics View**: Real-time charts — vehicle count time-series, type distribution, congestion trends | Member 4 | Gemini 3 |
| 24-27 | **Vehicle Search**: Search by plate → show trajectory on map + timeline of sightings | Member 1 | Claude Opus |
| 27-30 | **Anomaly Alerts**: Stationary vehicle detection, wrong-way alert, stolen vehicle flag panel | Member 6 | Gemini 5 |

### Phase 4: Polish & Demo (Hours 30-36)

| Hour | Task | Who | AI Co-pilot |
|---|---|---|---|
| 30-32 | End-to-end test: 4 demo videos → full pipeline → dashboard displays everything correctly | All | All |
| 32-33 | Performance optimization: batch inference, Redis caching, minimize WebSocket payload | Member 1+5 | Claude + Gemini 4 |
| 33-34 | UI polish: dark theme, loading states, smooth transitions, responsive layout | Member 2+3 | Gemini 1+2 |
| 34-35 | Record backup demo video (in case of live demo failure) | Member 6 | Gemini 5 |
| 35-36 | **Final presentation prep**, Q&A rehearsal, edge case documentation | All | Claude Opus |

---

## 7. The Demo Script (What Judges See)

> [!TIP]
> **This demo sequence is designed to build from simple to mind-blowing over 8 minutes.**

### Act 1: "The Basics" (2 min)
> "Here's our system processing 4 live camera feeds simultaneously."
- Show 4-camera grid, vehicles being detected in real-time with bounding boxes
- Zoom into a detection → show plate text "KA 05 MR 7821" extracted with 97% confidence
- "Our ANPR handles Indian HSRP plates, old plates, dirty plates, and night conditions."

### Act 2: "The Magic" (3 min)
> "Now watch what happens when the same vehicle appears across cameras."
- A white car exits Camera 1 heading north
- 2 minutes later, the SAME car appears in Camera 3
- System highlights: "🔗 MATCH: KA 05 MR 7821 — Camera 1 → Camera 3 — Travel time: 2m 14s"
- Switch to **City Map View** → animated trajectory draws itself along the road connecting both cameras
- "We use a 3-layer matching engine: plate text, visual appearance Re-ID, and spatial-temporal road graph reasoning."

### Act 3: "The Intelligence" (2 min)
> "This isn't just tracking — it's city-wide traffic intelligence."
- Show **congestion heatmap** pulsing on the map
- Show **analytics dashboard**: "Junction A has 340 vehicles/hour, 45% over capacity"
- Show **anomaly alert**: "⚠️ Stationary vehicle at Camera 2 for 7 minutes"
- "City traffic command centers can use this for real-time decision making."

### Act 4: "The Power" (1 min)
> "And for law enforcement..."
- Type a plate number into Vehicle Search
- System shows: full trajectory on map, every camera sighting with timestamp
- "In a stolen vehicle scenario, authorities can trace the exact route in seconds."

---

## 8. How to Deploy Your 6 AI Plans (Project-Specific)

| AI Plan | Team Member Role | Specific Tasks on This Project |
|---|---|---|
| **Claude Opus** | Lead Architect / System Designer | Architecture decisions, cross-camera matching algorithm, WebSocket pipeline, complex debugging, presentation script |
| **Gemini Pro #1** | Frontend / Visualization | deck.gl map implementation, TripsLayer animation, HeatmapLayer, MapLibre integration |
| **Gemini Pro #2** | Computer Vision / ANPR | YOLOv8 training, plate detection pipeline, OpenCV preprocessing, night/blur handling |
| **Gemini Pro #3** | Analytics / Data | Analytics engine, congestion scoring algorithm, TimescaleDB queries, chart components |
| **Gemini Pro #4** | Tracking / Re-ID | ByteTrack integration, OSNet Re-ID, cross-camera association, trajectory stitching |
| **Gemini Pro #5** | Integration / Polish | Docker setup, API integration, demo video recording, PPT preparation, Q&A prep |

---

## 9. Scoring Against SIH Rubric

| Criteria | How You Score High | Score Potential |
|---|---|---|
| **Innovation (20%)** | 3-layer cross-camera matching (plate + Re-ID + spatial-temporal graph) is novel. Most existing systems use plate-only matching. | 18/20 |
| **Technical Feasibility (30%)** | Working prototype with 4 camera streams, real-time inference, proven models (YOLOv8, ByteTrack). All open-source, no vendor lock-in. | 27/30 |
| **Impact (50%)** | BEL deploys smart city infrastructure for Indian government. This directly plugs into their product line. Law enforcement + traffic management + urban planning = triple impact. | 45/50 |
| | **Total Potential** | **90/100** |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ANPR fails on blurry/night plates | Can't do trajectory matching | Pre-process with CLAHE contrast enhancement + super-resolution upscaling. Fallback to Re-ID-only matching |
| Cross-camera matching has false positives | Wrong trajectories shown | Use strict spatial-temporal constraints (can't teleport). Require 2 of 3 layers to agree |
| GPU not available at venue | Can't run real-time inference | Pre-process demo videos, cache results, simulate real-time playback. Bring a laptop with GPU if possible |
| Live demo crashes | Presentation ruined | **Always have a pre-recorded backup video**. This is non-negotiable |
| 4 video streams overwhelm compute | Lag/dropped frames | Process at 5 FPS instead of 30 (sufficient for ANPR), use batch inference, skip every 6th frame |

---

## 11. What Makes You UNIQUE vs Other Teams

Most teams attempting ANPR will build:
❌ Single-camera plate reader + basic vehicle counter

**You are building:**
✅ Multi-camera city-scale trajectory intelligence with a 3-layer matching engine, animated trajectory visualization on a 3D city map, real-time congestion heatmaps, anomaly detection, and vehicle search — essentially a **command center for city traffic operations**.

> [!IMPORTANT]
> **The unfair advantage:** Your 6 AI co-pilots let you build in 36 hours what would normally take a team 3-4 weeks. No other team at SIH will have this velocity.
