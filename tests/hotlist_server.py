"""Browser-only fixture: real API/database/sockets, deterministic OCR output."""
import json
import os
import pathlib
import sys
os.environ.setdefault("ANPR_APPEARANCE_BACKEND", "opencv")
os.environ.setdefault("ANPR_APPEARANCE_DEVICE", "cpu")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from app import main


def fixture_ocr(path):
    plate = "GJ01AB5678" if pathlib.Path(path).name.startswith("CAM02_") else "GJ01AB1234"
    candidate = dict(text=plate, raw_text=plate, confidence=.95, status="ok", valid_format=True,
                     detection_id=1, bbox=dict(x=20,y=20,width=200,height=60))
    return plate, .95, "ok", json.dumps([candidate])


main.process_image = fixture_ocr
app = main.app
