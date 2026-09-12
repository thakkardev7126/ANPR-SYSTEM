"""
Seeds the camera registry with mock 'camera' locations for the demo.
Each phone used during the demo is assigned one of these camera_ids.
Edit lat/lng to real junctions near your demo location if you want the
map to reflect an actual route.
"""
from app.database import SessionLocal, Camera, Base, engine

DEMO_CAMERAS = [
    {"camera_id": "CAM01", "label": "College Main Gate",  "lat": 23.0395, "lng": 72.5660},
    {"camera_id": "CAM02", "label": "SG Highway Junction", "lat": 23.0325, "lng": 72.5145},
    {"camera_id": "CAM03", "label": "CG Road Crossing",    "lat": 23.0258, "lng": 72.5644},
    {"camera_id": "CAM04", "label": "Parking Exit",        "lat": 23.0410, "lng": 72.5590},
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for cam in DEMO_CAMERAS:
            existing = db.get(Camera, cam["camera_id"])
            if not existing:
                db.add(Camera(**cam))
        db.commit()
        print(f"Seeded {len(DEMO_CAMERAS)} cameras.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
