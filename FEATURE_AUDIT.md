# ANPR Feature Audit

> Historical pre-change audit. See [IMPLEMENTATION_NOTES.md](C:/Users/Dev/Downloads/Model/IMPLEMENTATION_NOTES.md) for the 11 September implementation update and test results.

Audited: 10 September 2026  
Source: `C:\Users\Dev\Downloads\Model`

**Result: 4 implemented, 13 partially implemented, 4 missing.**

This report covers the active FastAPI backend, database and trajectory code, YOLO/PaddleOCR pipeline, both HTML frontends (including their embedded Leaflet/WebSocket scripts), migrations, and supporting training/data-preparation code. The earlier SIH/PROJECT_GRAPES project is not part of this audit. Virtual environments, uploaded photos, datasets and saved checkpoints are not treated as application implementations; training configuration is supporting evidence only.

Status reflects source implementation and focused executable probes, not measured accuracy across real plate, weather or camera datasets. No application code or existing scan records were changed.

Legend:

- [x] Implemented: the requested mechanism is present and connected to an active path.
- [ ] Partial: some mechanism exists, but a requested part, processing path or correctness requirement remains incomplete.
- [ ] Missing: no dedicated implementation was found in the audited application.

## Critical Priority

- [x] **1. 2-line / stacked plates: Y-axis spatial line sorting** ([anpr_pipeline.py:647](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:647), `_spatially_join_ocr_blocks`; [anpr_pipeline.py:502](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:502), `flatten_double_line_plate`).
  **Implemented.** OCR blocks are grouped by Y position, rows sorted top-to-bottom, and blocks sorted left-to-right. Layout inference and row flattening are connected through `read_plate_text`. A shuffled two-row probe reconstructed `GJ01AB8768`. Caveat: row grouping uses a fixed 15-pixel tolerance, so scale/rotation robustness is not established.

- [ ] **2. HSRP plates and "IND" logo noise stripping** ([anpr_pipeline.py:642](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:642), `_is_hsrp_noise`, `_spatially_join_ocr_blocks`, `read_plate_text`).
  **Partial.** Separate OCR tokens `IN`, `IND` and `INDIA` are removed before joining and confidence averaging. There is no dedicated visual logo/hologram detector or reliable attached-prefix removal. Verified: `clean_plate_text("INDGJ01AB1234")` returns the incorrect candidate `GJ01A812` at 0.55 rather than the plate. The noise helper also removes digits before comparison, so `IND123` is discarded wholesale.

- [ ] **3. Positional OCR character fixes: 6J -> GJ, 876B -> 8768, O -> 0, S -> 5, Z -> 2, D -> 0** ([plate_rules.py:46](C:/Users/Dev/Downloads/Model/backend/app/plate_rules.py:46), correction dictionaries; [anpr_pipeline.py:304](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:304), `apply_positional_plate_corrections`, `enforce_strict_indian_plate_regex`, `clean_plate_text`).
  **Partial: all requested substitutions exist, but normalization is inconsistent.** Verified full OCR cleanup converts `6J01AB876B` to `GJ01AB8768` and `GJ01ABOSZD` to `GJ01AB0520`. However, direct `normalize_plate_text("GJ01ABOSZD")` produces `GJ01ABO520`, choosing a different series/serial split. The strict path accepts the first ten characters of longer input: `GJ01AB12345` becomes `GJ01AB1234` with a 0.96 pattern hint. Shared parsing and length validation remain pending.

- [x] **4. Leaflet GIS live map and polyline trajectory rendering** ([dashboard.html:195](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:195), `initLeafletMap`, `renderMap`; [dashboard.html:471](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:471), `refresh`; [trajectory.py:30](C:/Users/Dev/Downloads/Model/backend/app/trajectory.py:30), `build_trajectory`).
  **Implemented.** OpenStreetMap tiles, camera markers, timestamped hop popups, `L.polyline` and fitted bounds are connected to the trajectory API. Upload/correction WebSocket events trigger refresh; four-second polling also discovers persisted live-stream events. Lines connect sightings directly, not road-routed vehicle positions. The separate camera preview fetches a single snapshot per selection; it should not be confused with continuously refreshed map data.

- [ ] **5. Multiple cars in one frame: bounding-box area selection** ([anpr_pipeline.py:259](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:259), `locate_plates`; [anpr_pipeline.py:889](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:889), `process_image`; [main.py:156](C:/Users/Dev/Downloads/Model/backend/app/main.py:156), `StreamNode._process_frame`).
  **Partial.** YOLO returns multiple plate boxes and the live path tracks/processes multiple detections. Uploaded photos pool candidates from every box, then return and persist one winning plate by vote count/confidence. There is no largest-box selection or user-selectable box. A controlled two-box probe chose the smaller box's plate because it had more OCR candidates. Contour fallback area scoring does not provide area selection for normal YOLO detections.

- [ ] **6. Standard, BH, diplomatic and military-arrow plate regex support** ([plate_rules.py:84](C:/Users/Dev/Downloads/Model/backend/app/plate_rules.py:84), `_classify_state_plate`; [plate_rules.py:150](C:/Users/Dev/Downloads/Model/backend/app/plate_rules.py:150), `_classify_bh_plate`; [plate_rules.py:69](C:/Users/Dev/Downloads/Model/backend/app/plate_rules.py:69), `sanitize_plate_text`).
  **Partial.** State plates have positional/format rules. BH handling exists but expects `BH22AA1234`; the documented BH order is `YY BH #### XX`, such as `22BH1234AA`. The probe accepted the former and rejected the latter. No diplomatic or military-specific grammar exists, and sanitization removes the military arrow before classification. The BH format is corroborated by the [Ministry of Road Transport and Highways announcement](https://www.pib.gov.in/pressreleaseiframepage.aspx?lang=2&prid=1749764&reg=48).

## High Priority

- [ ] **7. Duplicate scan filter / debouncing** ([main.py:110](C:/Users/Dev/Downloads/Model/backend/app/main.py:110), `_read_tracked_plate`, `_enqueue_live_event`; [scan.html:408](C:/Users/Dev/Downloads/Model/frontend/scan.html:408), `captureAutoFrame`; [main.py:388](C:/Users/Dev/Downloads/Model/backend/app/main.py:388), `_create_scan_event`).
  **Partial.** Live tracks cache OCR for eight seconds unless a better detection arrives, and suppress repeated track/plate events for twelve seconds. The stream fallback has a five-second same-plate gate. Phone auto-scan pauses three seconds after a confident result, but does not compare plate identities. Manual uploads and `/api/process-frame` have no server-side same-camera/plate debounce; repeated frames can create repeated events.

- [ ] **8. WebSocket auto-reconnect on network loss** ([dashboard.html:169](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:169), `connectEventSocket`; [dashboard.html:312](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:312), `renderCameraSelector`; [scan.html:197](C:/Users/Dev/Downloads/Model/frontend/scan.html:197), `connectLiveCamera`).
  **Partial.** The dashboard event socket retries after 1.5 seconds; dashboard refresh recreates a closed camera socket. The phone scanner's camera socket only changes its status to offline on close and never schedules reconnection. Backend `MobileStreamReceiver._receive_loop` separately reconnects HTTP/RTSP capture, which does not repair the phone's WebSocket.

- [ ] **9. Missing GPS / camera location fallback defaults** ([seed_cameras.py:9](C:/Users/Dev/Downloads/Model/backend/app/seed_cameras.py:9), `DEMO_CAMERAS`; [main.py:570](C:/Users/Dev/Downloads/Model/backend/app/main.py:570), `register_camera`; [trajectory.py:30](C:/Users/Dev/Downloads/Model/backend/app/trajectory.py:30), `build_trajectory`; [dashboard.html:206](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:206), `renderMap`).
  **Partial.** Scans use registered camera coordinates, four cameras have seeded locations, the dashboard has demo defaults, and null map coordinates are skipped. Missing coordinates during camera registration silently become `0,0`; explicit null values fail float conversion, and coordinate ranges are not validated. Unknown scan camera IDs are rejected rather than assigned a fallback. Phone GPS is not acquired. Startup seeding also overwrites edited locations of the four demo IDs.

- [ ] **10. Stolen vehicle / hotlist alert popup notifications** (Missing / Pending implementation).
  **Missing.** No hotlist/stolen-vehicle table, match service, API, alert event or corresponding notification UI was found. Existing Leaflet popups show sightings; review badges show OCR uncertainty. Neither implements hotlist alerts.

- [x] **11. Multi-coloured plates: EV green and commercial yellow contrast parsing** ([plate_rules.py:205](C:/Users/Dev/Downloads/Model/backend/app/plate_rules.py:205), `classify_plate_color`; [anpr_pipeline.py:539](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:539), `prepare_ocr_images`; [anpr_pipeline.py:679](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:679), `read_plate_text`).
  **Implemented at a basic level.** HSV masks classify green EV, yellow commercial, white private, black rental and red testing backgrounds. OCR tries original, enlarged, grayscale, CLAHE and adaptive-threshold variants, and category/confidence metadata is propagated. Synthetic green/yellow/white crops classified correctly. Enhancement is shared across colours, not a colour-specific OCR model or proof of real-world reading accuracy.

- [ ] **12. Sharp side angles / skewed plates: bounding-box margin padding** ([anpr_pipeline.py:442](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:442), `perspective_unwarp_plate`; [anpr_pipeline.py:921](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:921), upload crop loop; [main.py:184](C:/Users/Dev/Downloads/Model/backend/app/main.py:184), live crop).
  **Partial.** Uploaded-photo boxes receive horizontal/vertical padding, and OCR attempts contour-based perspective rectification. Live tracked boxes are cropped without that padding. Rectification requires a usable four-corner contour and otherwise returns the original crop; PaddleOCR's document-unwarping and textline-orientation options are disabled.

## Medium Priority

- [ ] **13. Extreme scale variation: scooter / truck box normalization** ([anpr_pipeline.py:889](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:889), `process_image`; [anpr_pipeline.py:836](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:836), `_fit_ocr_region`; [anpr_pipeline.py:539](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:539), `prepare_ocr_images`).
  **Partial.** Uploads are limited to a 1280-pixel maximum dimension; OCR offers 2x enlargement, and fallback regions are capped at 480 pixels. There is no common target character-height normalization for every detection, tiled high-resolution detection, or tiny-plate recovery. Crops come from the downscaled image, so later enlargement cannot recover discarded detail. Live frames do not use the same upload size cap.

- [ ] **14. Mud, dirt, rain blur and high-speed motion blur filters** ([anpr_pipeline.py:291](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:291), `is_frame_sharp`; [anpr_pipeline.py:539](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:539), `prepare_ocr_images`; [augment_yolo_albumentations.py:54](C:/Users/Dev/Downloads/Model/scripts/augment_yolo_albumentations.py:54), `transform_for`).
  **Partial.** Runtime uses a Laplacian sharpness gate and bilateral denoising. Training augmentation generates fog, rain, mud, motion blur and defocus examples. Blurry crops are rejected before enhancement/OCR; there is no dedicated deblurring, deraining, mud removal or reconstruction of obscured characters. Augmentation is training support, not a runtime restoration filter.

- [ ] **15. Night vision, IR glare and harsh split-shadow preprocessing** ([anpr_pipeline.py:539](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:539), `prepare_ocr_images`).
  **Partial.** Grayscale, bilateral filtering, CLAHE and local adaptive thresholding provide useful illumination preprocessing. No IR-specific path, saturated-glare recovery, dedicated shadow correction or night/IR validation suite was found. Shared contrast enhancement alone does not establish support for these extreme cases.

- [ ] **16. Map history search by time range and geo-radius** ([main.py:856](C:/Users/Dev/Downloads/Model/backend/app/main.py:856), `search_plate_sightings`; [dashboard.html:407](C:/Users/Dev/Downloads/Model/frontend/dashboard.html:407), `searchSightings`; [001_postgis_pg_trgm.sql:5](C:/Users/Dev/Downloads/Model/backend/migrations/001_postgis_pg_trgm.sql:5)).
  **Partial.** API/UI support a look-back window in minutes and a radius around the selected camera, with PostGIS or SQLite/Python filtering. Arbitrary start/end datetimes are absent. Selecting a search result calls the unrestricted trajectory API, so the drawn route is not restricted to the search's time/radius. SQLite also considers only the newest 1,000 time-window events before similarity/radius filtering.

- [x] **17. High-volume FPS frame dropping under CPU load** ([anpr_pipeline.py:114](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:114), `LatestFrameBuffer`; [anpr_pipeline.py:199](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:199), `StreamProcessor`; [scan.html:408](C:/Users/Dev/Downloads/Model/frontend/scan.html:408), `captureAutoFrame`).
  **Implemented.** Capture has a single-slot queue that replaces stale frames while inference works; the processor samples every third consumed frame. Phone scanning skips capture while a request is in flight. A buffer probe inserting frames 0 through 9 returned only frame 9. This is backlog prevention, not CPU-utilization-based adaptive FPS control or global multi-client admission control.

## Low Priority

- [ ] **18. Partial / frame-edge cropped plates handling** ([anpr_pipeline.py:921](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:921), crop loop; [anpr_pipeline.py:361](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:361), `clean_plate_text`; [anpr_pipeline.py:679](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:679), `read_plate_text`).
  **Partial.** Crop bounds are clipped to the image, bounded fallback regions exist, and unrecognized text can be retained for review. There is no explicit truncated-plate detector, partial-string representation or multi-frame character reconstruction. Substring matching can instead turn cropped/noisy text into a plausible different plate; review retention is not complete cropped-plate recognition.

- [ ] **19. Offline scan caching on mobile when Wi-Fi drops** (Missing / Pending implementation; inspected [scan.html:283](C:/Users/Dev/Downloads/Model/frontend/scan.html:283), `uploadOneFile`, upload handler and `captureAutoFrame`).
  **Missing.** No IndexedDB scan queue, persisted image cache, service worker, replay worker or online-event retry exists. Local storage remembers only the camera ID. Manual failure clears the selected files, and failed auto-scan images are discarded after error display.

- [ ] **20. Physical rivets / screws drilled into characters** (Missing / Pending implementation).
  **Missing.** No dedicated screw/rivet segmentation, masking, component rejection, inpainting or occlusion-aware character reconstruction was found. Generic denoising, OCR and mud augmentation do not establish this feature.

- [ ] **21. Modern fancy / handwritten fonts fallback** (Missing / Pending implementation; inspected [anpr_pipeline.py:68](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:68), `_create_paddle_ocr`, and [anpr_pipeline.py:679](C:/Users/Dev/Downloads/Model/backend/app/anpr_pipeline.py:679), `read_plate_text`).
  **Missing as an automatic font-specific fallback.** PaddleOCR is the single recognizer; alternate image variants use the same reader. There is no alternate handwriting/stylized-font recognizer or confidence-triggered engine switch. Human correction and feedback-image export exist, but they do not implement automatic font fallback or automatically retrain the OCR model.

## Verification and Limits

Focused probes ran with the installed system Python 3.13 and actual imported pipeline/rules functions. YOLO and PaddleOCR inference were not loaded for these probes. The multi-box test mocked detection/OCR outputs to isolate aggregation behavior and disabled debug-image writes.

| Probe | Observed result |
| --- | --- |
| Shuffled stacked rows with separate IND token | Correct row order; GJ01AB8768 |
| Requested positional substitutions | Correct through clean_plate_text; direct normalizer can choose a different split |
| BH examples | 22BH1234AA rejected; BH22AA1234 accepted |
| Diplomatic sample 123CD4567 | Rejected by both format-cleanup paths |
| Military arrow sanitization | Arrow removed |
| Attached IND prefix | Incorrect GJ01A812 candidate, confidence hint 0.55 |
| Extra trailing character | GJ01AB12345 truncated to GJ01AB1234 |
| Green / yellow / white synthetic crops | Correct HSV categories |
| OCR preprocessing | All five image variants produced |
| Flat blurred crop | Rejected by sharpness gate |
| Fallback crop scaling | 20x80 unchanged; 200x1000 reduced to 96x480 |
| Two detected plates in one uploaded image | One result selected by vote count, not box area |
| Latest-frame queue | Only newest frame retained |

No project-owned automated test suite was found outside dependency/generated directories. No end-to-end camera session, actual network outage, PostgreSQL execution, representative OCR accuracy benchmark or sustained load test was performed. These remain necessary to validate operational performance, rather than implementation presence.

The project `.venv` points to an unavailable Python installation under a different Windows user. `.venv-1` lacks OpenCV. The installed `py -3.13` interpreter successfully ran the probes, so neither virtual environment was modified.

The detector training metadata at [args.yaml:4](C:/Users/Dev/Downloads/Model/runs/detect/indian-plates/args.yaml:4) names the augmented dataset, but records one epoch, fraction 0.02 and image size 320. That is not evidence of comprehensive robustness validation. No model checkpoint was deserialized for this audit.

## Recommended Fix Order

1. Correct BH grammar, preserve/classify military notation, add diplomatic grammar, and unify normalization without silent truncation.
2. Handle merged IND prefixes reliably and add regression cases for row sorting and every requested character confusion.
3. Preserve one result per detected plate or implement explicit area selection; apply padding consistently to uploads and streams.
4. Add server-side duplicate suppression for phone scans, phone WebSocket reconnect, and validated camera coordinates.
5. Implement hotlist alerts and mobile offline replay; make history filters carry through to the displayed route.
6. Validate scale, lighting, weather, occlusion and font behavior with representative labeled examples before expanding support claims.
