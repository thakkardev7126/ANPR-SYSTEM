# Stolen Vehicle / Hotlist Alerts

Implemented: 11 September 2026, in `C:\Users\Dev\Downloads\Model`.

## Included

- [x] Local hotlist management: add, search, edit, deactivate/reactivate, optional expiry, stolen/wanted/watchlist category, reason and case reference (`backend/app/hotlist_api.py`; `frontend/hotlist.html`, `hotlist.js`).
- [x] Exact normalized plate matching on uploads, phone frames, batches and live streams (`backend/app/database.py:persist_plate_event`; `backend/app/hotlist.py:match_event`). No fuzzy hotlist matching.
- [x] Persistent alerts saved atomically with sightings. Same-event matches are unique and inherit the existing same-camera scan debounce (`HotlistAlert`; `HotlistNotification`).
- [x] Scanner and dashboard popup notifications, with live WebSocket delivery, REST catch-up after reconnect/reload, and periodic reconciliation (`frontend/hotlist-alerts.js`; `main.py:publish_hotlist_notifications`).
- [x] Low-confidence/cropped reads labelled **Needs verification**, distinct from an **Exact plate match**. Neither means identity or stolen status has been independently verified.
- [x] Acknowledge alerts across pages/devices; searchable hotlist and paginated alert history. Scanner popups show the selected camera; dashboard popups cover all cameras.
- [x] Human corrections create or retract matches. Better confirmed observations upgrade earlier review alerts and reopen acknowledgement.
- [x] Alert snapshots retain plate, listing reason/reference, camera label/GPS, sighting time and OCR confidence. Clearing scan history preserves these snapshots and the hotlist.
- [x] Inactive/expired entries stop future matches. Previous alerts retain their historical listing details.

## Use

1. Open **Hotlist & alerts** from the dashboard or scanner, or open `/hotlist.html`.
2. Choose **Add plate**, enter the plate, listing category and reason, optionally a case reference/expiry, then save.
3. Scan that plate through an existing camera node. Matching produces a popup and a durable alert record.
4. Verify the actual vehicle and case details before acting. **Acknowledge** records that the alert was seen; it does not mark a vehicle recovered or prove an offence.
5. Edit the entry and turn **Active listing** off when it no longer applies. Plate numbers are immutable; use a new entry for a different plate.

Matches apply to scans processed while a listing is active. There is no automatic retrospective sweep when adding an entry. A repeated scan inside the debounce window may match its existing event if the listing was just activated. A sighting after that window creates a new event and alert; adjust `ANPR_DEDUP_SECONDS` as appropriate.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/hotlist?q=GJ01&offset=0&limit=50` | List/search entries |
| POST | `/api/hotlist` | Add entry |
| PUT | `/api/hotlist/{id}` | Replace editable entry fields |
| DELETE | `/api/hotlist/{id}` | Deactivate, retaining history |
| GET | `/api/hotlist-alerts?unacknowledged=true&camera_id=CAM01` | Retrieve durable inbox |
| POST | `/api/hotlist-alerts/{id}/acknowledge` | Acknowledge alert |

Create/update body: `plate_text`, `category` (`stolen`, `wanted`, `watchlist`), `reason`, optional `reference`, `active`, `expires_at` (ISO timestamp or null).

Scan responses include `hotlist_alerts` for each detection. WebSocket `/ws/events` sends `type: "hotlist_alert"` with an `alert` object, stable ID and revision. Treat revisions as updates to the same alert, not independent sightings.

SQLite creates the three new tables on startup without clearing existing data. PostgreSQL DDL is provided in `backend/migrations/003_hotlist.sql` and is included by the existing ordered migration runner. PostgreSQL execution was not tested locally.

## Verification

- 40 backend regression tests pass, including eight hotlist tests covering CRUD/expiry, concurrent duplicate suppression, exact versus near matches, camera scope, review upgrades, acknowledgement, corrections/retractions, history clearing and live-writer WebSocket delivery.
- Three existing WebSocket lifecycle tests pass.
- Browser workflow checks pass at 1366x900 and 390x844: add/edit, inactive suppression, scan popups, dashboard delivery, reload catch-up, deduplication and cross-page acknowledgement. The browser fixture uses a temporary database and deterministic OCR; API, persistence and WebSockets are real. No test hotlist entries are added to your working database.

```powershell
py -3.13 -B -m unittest discover -s tests -p test_features.py -v
node --test tests/test_socket.cjs
$env:NODE_PATH = 'C:\Users\Dev\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node tests/test_hotlist_browser.cjs
```

## Boundaries

This is an **operator-managed local list**, not an official police/insurance registry integration. No external vehicle data is fetched and no enforcement action is automated. OCR false positives remain possible; a matching plate can also be cloned.

Notifications are in-app, while the page is open. Unacknowledged alerts survive closed tabs and are retrieved on return. This is not an offline image-upload queue, SMS/email delivery, or operating-system push notification service.

The existing application has no authentication/role system. Keep the preview on loopback/trusted development machines; add authenticated operator roles, HTTPS, access auditing and retention rules before a shared production deployment. No database credential or authorization-service integration was invented for this feature.
