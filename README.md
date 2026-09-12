# City-wide ANPR trajectory tracking — working demo

A real, running version of the system: phones act as cameras, a FastAPI
backend runs actual plate detection + OCR on every photo, a trajectory
engine stitches sightings of the same plate into a path, and a live
dashboard shows it on a map. Nothing here is mocked except the camera
*hardware* — the AI pipeline is genuine.

## Feature Update (11 September 2026)

Stolen-vehicle/hotlist management and live popup alerts are now available at
`/hotlist.html`. See [HOTLIST_FEATURE.md](HOTLIST_FEATURE.md) for usage, matching
rules, APIs, tests and deployment limits. This is a local operator-managed list,
not an official stolen-vehicle registry connection.

The 13 partially implemented features from the audit now have connected runtime
changes and regression coverage. See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
for the checklist, configuration, test commands, actual OCR smoke-test results and
remaining validation limits. Offline scan caching, rivet handling and a dedicated
handwriting fallback remain separate work. Edit registered camera locations through the dashboard or camera
API; startup no longer overwrites existing camera coordinates.

## What's inside

```
anpr-system/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app — all API endpoints
│   │   ├── database.py        SQLite/PostgreSQL models (Camera, PlateEvent)
│   │   ├── anpr_pipeline.py   Plate detection (YOLOv8) + OCR (PaddleOCR)
│   │   ├── plate_rules.py     Indian plate rules, colour class, review status
│   │   ├── trajectory.py      Cross-camera trajectory stitching engine
│   │   └── seed_cameras.py    Registers the 4 demo "cameras" (edit GPS here)
│   ├── migrations/             PostgreSQL/PostGIS/pg_trgm migration SQL
│   ├── uploads/                Uploaded scan photos land here
│   └── requirements.txt
└── frontend/
    ├── scan.html               Open this on each phone — the camera page
    └── dashboard.html          Open this on a laptop/projector — the live map
```

## 1. Run the backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
On this Windows machine, use Python 3.13 explicitly:
```powershell
cd C:\Users\Dev\Downloads\Model\backend
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
py -3.13 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Train With Kaggle + DataCluster + Roboflow + Albumentations

The project now prefers the Albumentations-expanded YOLOv8 dataset when it
exists:

```text
data/plate-dataset/anpr-augmented-yolo/data.yaml
```

That dataset is generated from the merged labelled sources:

```text
data/plate-dataset/datacluster-roboflow-yolo/data.yaml
```

Prepare the Kaggle baseline:

```powershell
py -3.13 scripts\prepare_kaggle_yolo.py
```

Prepare the DataCluster public sample from Hugging Face:

```powershell
py -3.13 scripts\prepare_datacluster_yolo.py
```

Add a Roboflow Universe export when you have your API key and selected
project/version:

```powershell
$env:ROBOFLOW_API_KEY = "your_api_key"
$env:ROBOFLOW_WORKSPACE = "workspace_name"
$env:ROBOFLOW_PROJECT = "project_slug"
$env:ROBOFLOW_VERSION = "1"
py -3.13 scripts\download_roboflow_yolo.py
```

Merge the labelled sources, generate fog/rain/mud/blur/folded variants with
Albumentations, then train:

```powershell
py -3.13 scripts\merge_yolo_sources.py
py -3.13 scripts\augment_yolo_albumentations.py --per-kind 1
py -3.13 backend\train_resume.py
```

Check which dataset will be used without starting training:

```powershell
py -3.13 backend\train_resume.py --dry-run
```

The current local augmented dataset has 9907 training images/labels and 416
validation images/labels. Roboflow is optional until you provide the four
`ROBOFLOW_*` environment variables; the merge script skips its placeholder
folder if no `data.yaml` exists there. See
`data/plate-dataset/SOURCES.md` for the exact source status.

The first run auto-creates `anpr_demo.db` (SQLite) and seeds 4 demo cameras
(edit their names/GPS in `app/seed_cameras.py` before your run-through —
ideally real junctions near your demo location, so travel-time checks make
sense).

**Find your laptop's local IP** (so phones on the same WiFi can reach it):
```bash
# macOS/Linux
ipconfig getifaddr en0   # or: hostname -I
# Windows
ipconfig
```
You'll get something like `192.168.1.23`. Phones and the dashboard must be
on the **same WiFi network** as the laptop running the backend.

## 2. Open the phone scanner (on each phone)

Serve the frontend folder so phones can load it over WiFi:
```bash
cd frontend
python3 -m http.server 5500
```
On each phone's browser, go to:
```
http://<your-laptop-ip>:5500/scan.html
```
You can also open the scanner directly from the FastAPI backend:
```
http://<your-laptop-ip>:8000/scan.html
```
Pick which camera identity that phone represents (CAM01, CAM02, ...) —
this choice is remembered on that phone. Tap "Start auto scan" and place
the number plate in front of the phone camera; the page captures frames,
sends them to the backend, and shows the best read automatically. Manual
single-photo and multi-photo upload are still available for testing. On
most phone browsers, automatic camera access requires HTTPS or localhost;
manual photo capture still works over ordinary local HTTP.

For automatic phone-camera scanning over WiFi, run the HTTPS demo server:
```powershell
cd C:\Users\Dev\Downloads\Model
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
py -3.13 scripts\run_https_demo.py
```
Then open:
```
https://<your-laptop-ip>:8443/scan.html
```

## 3. Open the live dashboard (on a laptop/projector)

```
http://<your-laptop-ip>:5500/dashboard.html
```
or:
```
http://<your-laptop-ip>:8000/dashboard.html
```
It polls the backend every 4 seconds — as new phones scan, the vehicle
list, stats, and trajectory map update live. Click any vehicle to see its
stitched path across cameras, with a red segment if a hop looks physically
implausible (arrived faster than the road distance allows). Low-confidence
OCR reads appear in the review queue so an operator can correct them.
The dashboard also includes fuzzy GIS search: search a partial/misread
plate within a radius and time window around the selected camera.

## PostgreSQL + PostGIS + pg_trgm production mode

The local demo still runs on SQLite. For production fuzzy/spatial search,
create a PostgreSQL database, install the backend dependency
`psycopg[binary]`, apply:

```sql
\i backend/migrations/001_postgis_pg_trgm.sql
```

or run:

```powershell
py -3.13 backend\apply_postgres_migration.py
```

Then start the backend with:

```powershell
$env:ANPR_DATABASE_URL = "postgresql+psycopg://user:password@host:5432/anpr"
py -3.13 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`/api/search/plates?q=GJ01HV8768&lat=23.02&lng=72.57&radius_m=5000&minutes=30`
uses PostgreSQL `similarity()` plus PostGIS `ST_DWithin()` when running on
PostgreSQL. On SQLite it falls back to Python string similarity and the
camera GPS values so the feature remains testable on a laptop.

## Demo script (rehearse this)

1. Open the dashboard on a screen everyone can see.
2. Phone "CAM01" scans a car's plate → dashboard shows 1 sighting.
3. Wait ~30 seconds, walk to a different spot, phone "CAM02" scans the
   *same* plate → dashboard draws the connecting line live.
4. Repeat once more from "CAM03" → full 3-hop trajectory visible.
5. Point out the confidence/needs-review flow: scan a blurry or angled
   plate — dashboard/response marks it `PENDING_REVIEW` instead of guessing.
   Corrected review images and labels are saved under
   `data/review_feedback/` for later retraining.

## Notes on what's real vs. simulated

- **Real**: plate detection (YOLOv8), PaddleOCR text recognition,
  ByteTrack-based live tracking and OCR deduplication, blur rejection,
  perspective dewarping, double-line plate flattening, bilateral filtering,
  CLAHE/adaptive-threshold preprocessing, Indian plate post-processing,
  HSV plate-colour classification, majority-vote confidence scoring,
  trajectory stitching, travel-time plausibility check (haversine distance
  ÷ assumed speed), fuzzy GIS search, live dashboard, and review queue.
- **Simulated for the demo only**: the "camera network" itself — phones
  stand in for fixed CCTV/ANPR hardware, and GPS coordinates are hardcoded
  per camera_id rather than read from the phone.
- **Swap-in points for production** (already isolated in the code):
  `anpr_pipeline.locate_plates` → a trained YOLOv8 plate detector;
  `trajectory.estimate_travel_minutes` → a real Google Distance Matrix/OSRM
  call instead of straight-line distance.
- **Two-line native YOLO classes**: the runtime already understands
  `plate_single_line` and `plate_double_line` class names. The current
  bundled dataset is one-class (`license_plate`). Indian_LPR support is wired
  through `scripts/prepare_indian_lpr_two_class.py`; after you place an
  authorized Indian_LPR copy under `data/plate-dataset/Indian_LPR`, run:

  ```powershell
  py -3.13 scripts\prepare_indian_lpr_two_class.py --source data\plate-dataset\Indian_LPR
  ```

  This writes `data/plate-dataset/indian-lpr-two-class/data.yaml`, which
  `backend/train_resume.py` automatically prefers for two-class training. The
  public Indian_LPR GitHub repo states the full road dataset is not public due
  to legal restrictions, so this project does not silently download it.

## Troubleshooting

- **Phone can't reach the backend**: confirm phone and laptop are on the
  same WiFi, and that `API_BASE` in `scan.html`/`dashboard.html` resolves
  to the laptop's IP (it auto-detects via `window.location.hostname`, so
  as long as you load the page via `http://<ip>:5500/...` and not
  `localhost`, this works automatically).
- **OCR reads garbage**: expected for angled/blurry/low-light photos —
  that's what the confidence + "needs review" status is for. Hold the
  phone steady, fill the frame with the plate, good lighting.
- **`ModuleNotFoundError: cv2`**: `pip install opencv-python-headless`.
- **`ModuleNotFoundError: paddleocr` or `paddle`**: reinstall the backend
  dependencies with `pip install -r requirements.txt`.
- **Check active runtime config**: open `/api/system/status` to confirm the
  four camera nodes, PaddleOCR, database path, and YOLO weight path.
