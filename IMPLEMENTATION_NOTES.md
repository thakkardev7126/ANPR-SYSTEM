# ANPR Implementation and Verification

> Subsequent update: stolen-vehicle/hotlist management and alerts are now implemented. See [HOTLIST_FEATURE.md](C:/Users/Dev/Downloads/Model/HOTLIST_FEATURE.md). The 13-feature verification below remains the prior baseline.

Updated: 11 September 2026  
Project: `C:\Users\Dev\Downloads\Model`

## Scope and Status

This update addresses the **13 partially implemented items** identified by the original audit: checklist items **2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16 and 18**.

[x] below means the described software mechanism is implemented and connected to the application. It does **not** mean perfect OCR accuracy or production qualification. Severe occlusion, blur, saturation and missing pixels cannot be reliably reconstructed by these changes. The four originally missing features are listed separately at the end.

The original `FEATURE_AUDIT.md` is retained as the pre-change baseline.

## Implemented Checklist

- [x] **2. HSRP and IND noise stripping** (`backend/app/plate_rules.py:strip_hsrp_noise`; `backend/app/anpr_pipeline.py:_is_hsrp_noise`, `_spatially_join_ocr_blocks`, `read_plate_text`).
  Separate IN/IND/INDIA tokens and recognizable attached prefixes are removed. IND123 is not discarded as a logo. This is text-level handling, not a hologram authenticity detector.

- [x] **3. Consistent positional character fixes** (`backend/app/plate_rules.py:normalize_plate_text`; `backend/app/anpr_pipeline.py:clean_plate_text`, `apply_positional_plate_corrections`, `enforce_strict_indian_plate_regex`).
  Shared full-string rules cover 6J -> GJ, 876B -> 8768, O/D -> 0, S -> 5 and Z -> 2 in numeric positions. Extra trailing characters are not silently truncated. Invalid or ambiguous reads remain reviewable.

- [x] **5. Multiple cars and bounding-box area selection** (`backend/app/anpr_pipeline.py:process_image`, `best_plate_candidate`; `backend/app/main.py:_create_scan_event`; `frontend/scan.html:uploadFiles`, `plateSelection`).
  OCR votes are grouped per detection. Uploads default to all detected plates, with a largest-area mode. Responses contain a detections array plus backward-compatible top-level fields. Live detection continues to handle multiple tracks.

- [x] **6. Standard, BH, diplomatic and military-arrow formats** (`backend/app/plate_rules.py:_classify_state_plate`, `_classify_bh_plate`, `_classify_special_plate`, `sanitize_plate_text`).
  BH uses YYBH####XX. Diplomatic markers include CD, CC, CDP, UN, IOD and IOC. Military notation preserves U+2191 and canonicalizes supported arrow positions. Matching is syntactic, not registration verification. Standard automatic acceptance requires the supported state/district/series grammar and four serial digits; other legacy layouts remain for review.

- [x] **7. Cross-path duplicate scan filtering** (`backend/app/database.py:persist_plate_event`, `AsyncPlateEventWriter`; `backend/app/main.py:_create_scan_event`, `StreamNode._enqueue_live_event`).
  Uploads, phone frames and live streams share a transactional same-camera/same-plate debounce. Default window: 12 seconds. SQLite serializes the check/insert; PostgreSQL uses a transaction-scoped camera lock. A better observation can upgrade the existing event. Different cameras retain separate sightings.

- [x] **8. WebSocket auto-reconnect** (`frontend/reconnecting-socket.js:ANPRSocket`; scanner and dashboard socket callers; `backend/app/main.py:camera_socket`, `events_socket`).
  Shared reconnect logic includes backoff with jitter, connection timeout, heartbeat/pong, online/offline listeners, stale-socket protection and explicit teardown. Camera snapshots refresh independently of the consumed inference queue.

- [x] **9. Missing GPS and camera fallback handling** (`backend/app/location.py:resolve_location`, `camera_location`; camera registration/update APIs; `backend/app/seed_cameras.py:seed`; dashboard location controls).
  Coordinates must be a finite, in-range pair. Missing locations are explicitly unknown, not plotted at 0,0. A deliberately supplied 0,0 remains valid. Optional configured defaults are validated; edited demo locations survive startup. No browser GPS permission is required: the registered camera location is the scan location.

- [x] **12. Skew correction and consistent bounding-box padding** (`backend/app/image_quality.py:padded_plate_crop`; `backend/app/anpr_pipeline.py:perspective_unwarp_plate`, `_order_quad_points`; upload/live crop paths).
  Both upload and tracked live crops preserve margin. Perspective transforms retain detected corner geometry instead of replacing it with a rotated rectangle. Unusable contours fall back to the original image.

- [x] **13. Extreme scale normalization** (`backend/app/anpr_pipeline.py:locate_plates`, `process_image`, `plate_like_regions`; `backend/app/image_quality.py:normalize_size`).
  Crops retain source resolution. Large frames use overlapping 1280-pixel detection tiles with coordinate restoration and non-maximum suppression. OCR sizes crops consistently while preserving aspect ratio. Truly tiny or information-poor crops are flagged for review, not presented as recovered detail.

- [x] **14. Dirt/rain noise and motion-blur processing** (`backend/app/image_quality.py:assess_quality`, `enhanced_variants`, `wiener_motion`; `backend/app/main.py:StreamNode._read_tracked_plate`).
  Runtime denoising, contrast variants and bounded directional Wiener deblurring supplement training augmentation. Moderately blurred crops reach OCR; unreadable crops do not. Tracked reads use image quality and retain a better valid read when a later observation degrades. These filters do not remove opaque mud or guarantee heavy-rain/high-speed accuracy.

- [x] **15. Low-light, glare and split-shadow preprocessing** (`backend/app/image_quality.py:illumination_correct`, `enhanced_variants`; `backend/app/anpr_pipeline.py:prepare_ocr_images`, `read_plate_text`).
  Local illumination normalization, CLAHE, adaptive thresholds, inversion, low-light gamma and highlight compression are connected to OCR. Quality flags are retained. Saturated pixels cannot be recovered, and dedicated IR-camera performance remains unvalidated.

- [x] **16. Time-range and geo-radius map history** (`backend/app/location.py:time_window`; `backend/app/main.py:search_plate_sightings`, `get_trajectory`; `backend/app/trajectory.py:build_trajectory`; `frontend/dashboard.html:searchSightings`, `trajectoryUrl`).
  Absolute start/end datetimes and radius filters reach both search and rendered trajectories and survive refresh. Invalid ranges/coordinates are rejected. Unknown camera locations are excluded from radius searches. SQLite no longer discards candidates after the first 1,000 rows. Responses use UTC timestamp suffixes.

- [x] **18. Partial and frame-edge plate handling** (`backend/app/image_quality.py:bbox_frame_edges`, `PartialPlateHistory`; detector/fallback upload and live paths).
  Edge-touching detections carry explicit partial/edge metadata and require review. Bounded same-track text history can expose overlapping-read hypotheses; inferred characters are never auto-confirmed. A subsequent complete observation replaces a cached partial read. This does not reconstruct characters outside the image.

## Verification Performed

- **32 backend tests passed**, using an isolated temporary SQLite database and temporary upload/review directories. Coverage includes parser cases, stacked rows, crop margins, image variants, original-coordinate tiling/NMS, partial-review behavior, multi-box uploads, concurrent debounce, stream/upload deduplication, GPS validation, filtered history beyond 1,000 rows and heartbeat responses.
- **3 JavaScript socket tests passed** for retry, explicit shutdown, stale callbacks, malformed messages and connection timeout.
- **Browser checks passed at 1366x900 and 390x844** using headless Edge/Playwright: Leaflet path rendering, date/radius propagation, filter reset, multiple upload results, camera-socket reconnection, no page errors and no horizontal overflow. API fixtures isolate browser behavior; these checks do not prove camera hardware or OCR accuracy.
- **Actual YOLO and PaddleOCR smoke inference completed** on the project's validation image `data/plate-dataset/anpr-augmented-yolo/images/val/orig_s1_00000015.jpg`, using a working copy so source data remained unchanged. The single-line and stacked plates both read **JH24C1100**, with pipeline scores **0.974** and **0.976**.
- That real inference took **49.6 seconds** on this CPU run, including model initialization and fallback work. The current YOLO checkpoint produced four small false-positive boxes. Contour fallback recovered both plates after fixing the condition that previously let invalid detector text block recovery. This is **not** a throughput benchmark, calibrated confidence estimate or evidence of robust detector accuracy.
- Existing four demo cameras and scan history were not cleared. Starting the updated app performs additive SQLite schema/index migration and normalizes legacy timestamp separators for comparable ordering.

## Running and Testing

On this machine, use system Python 3.13; the existing project virtual environments were not repaired or replaced.

From the project root:

```powershell
py -3.13 -B -m unittest discover -s tests -p test_features.py -v
node --test tests/test_socket.cjs
```

Browser checks require Playwright and Microsoft Edge:

```powershell
$env:NODE_PATH = 'C:\Users\Dev\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node tests/test_browser.cjs
```

Start the application:

```powershell
cd C:\Users\Dev\Downloads\Model\backend
py -3.13 -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Dashboard: http://127.0.0.1:8000/dashboard.html  
Scanner: http://127.0.0.1:8000/scan.html

The local preview binds to loopback. For physical phones, use an appropriately secured LAN/HTTPS deployment; this preview is not externally exposed.

Optional configuration:
- `ANPR_DEDUP_SECONDS`: default 12; 0 disables time-window deduplication.
- `ANPR_DEFAULT_LAT` and `ANPR_DEFAULT_LNG`: supply both, or leave both unset for unknown new-camera locations.
- `ANPR_MODEL_PATH`: select a validated YOLO checkpoint.
- `ANPR_DATABASE_PATH`, `ANPR_UPLOAD_DIR`, `ANPR_REVIEW_DIR`: isolate development/test data.
- Scan endpoints: `POST /api/scan?selection=all` or `?selection=largest`; equivalent selection is supported on `/api/process-frame`.
- PostgreSQL: set `ANPR_DATABASE_URL`, then run `py -3.13 backend/apply_postgres_migration.py` from the project root before using an existing database. The runner applies all ordered SQL files, including `002_camera_location.sql`. PostgreSQL/PostGIS execution was **not tested locally**.

## Remaining Validation and Separate Features

- [ ] Representative real-camera accuracy and sustained-load testing across sharp angles, scale, rain, mud, night/IR and cropped plates (Pending validation).
- [ ] PostgreSQL/PostGIS migration and integration execution (Pending validation; local automated tests use SQLite).
- [ ] Physical-phone Wi-Fi loss/recovery testing (Pending validation; socket closure/retry and browser integration are tested).
- [x] **10. Hotlist/stolen-vehicle popup alerts** (Implemented in the subsequent update; see `HOTLIST_FEATURE.md`).
- [ ] **19. Durable offline mobile scan queue and replay** (Missing; WebSocket reconnection and retaining failed manual selections are not offline caching).
- [ ] **20. Dedicated rivet/screw occlusion handling** (Missing).
- [ ] **21. Dedicated fancy/handwritten-font recognizer fallback** (Missing).

The previously implemented stacked-line sorting, Leaflet map/polyline rendering, color classification and latest-frame dropping remain in place. Stacked-line grouping additionally uses text-height-dependent row tolerance.

Do not describe the original 21-item checklist as fully production-validated. The software gaps above have been addressed within the 13-item scope; the remaining missing features and empirical validation requirements remain explicit.
