"""Regression coverage for the 13-feature implementation; never opens the demo DB."""
import os
import sys
import tempfile
import pathlib
import json
import datetime as dt
import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

TEMP = tempfile.TemporaryDirectory(prefix="anpr-tests-")
os.environ["ANPR_DATABASE_PATH"] = str(pathlib.Path(TEMP.name) / "tests.db")
os.environ["ANPR_UPLOAD_DIR"] = str(pathlib.Path(TEMP.name) / "uploads")
os.environ["ANPR_REVIEW_DIR"] = str(pathlib.Path(TEMP.name) / "review")
os.environ["ANPR_APPEARANCE_BACKEND"] = "opencv"
os.environ["ANPR_APPEARANCE_DEVICE"] = "cpu"
os.environ.pop("ANPR_DATABASE_URL", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

import cv2
import numpy as np
from fastapi.testclient import TestClient
from app import anpr_pipeline as pipeline
from app.plate_rules import normalize_plate_text, strip_hsrp_noise
from app.image_quality import padded_plate_crop, normalize_size, assess_quality, enhanced_variants, PartialPlateHistory
from app.location import resolve_location
from app.database import SessionLocal, PlateEvent, Camera, CameraRoadConnection, Vehicle, VehicleMatchCandidate, persist_plate_event, engine, HotlistEntry, HotlistAlert, HotlistNotification
from app.main import app
from app.seed_cameras import seed
from app.road_network import camera_transition
from app.travel_time import calculate_travel_segment, vehicle_speed_history
from app.vehicle_appearance import (
    analyze_vehicle_appearance,
    appearance_similarity,
    create_embedding,
    deserialize_embedding,
    normalize_vehicle_crop,
    serialize_embedding,
)
from app.vehicle_matching import compare_observations, process_observation_matches


def tearDownModule():
    engine.dispose()
    TEMP.cleanup()


def candidate(text, box, identity):
    return dict(text=text, raw_text=text, confidence=.94, status="ok",
                valid_format=True, bbox=box, detection_id=identity)


def synthetic_vehicle_image(color=(90, 90, 90), plate="GJ01AB1234"):
    image = np.full((240, 360, 3), (36, 42, 48), np.uint8)
    cv2.rectangle(image, (42, 52), (318, 188), color, -1)
    cv2.rectangle(image, (70, 82), (140, 150), (25, 25, 25), 2)
    cv2.rectangle(image, (216, 82), (286, 150), (25, 25, 25), 2)
    cv2.rectangle(image, (112, 158), (258, 198), (238, 238, 238), -1)
    cv2.putText(image, plate, (118, 185), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 2)
    cv2.line(image, (45, 54), (315, 186), (180, 180, 180), 2)
    return image


def encode_jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode synthetic test image")
    return buffer.tobytes()


def write_temp_image(name, image):
    path = pathlib.Path(TEMP.name) / name
    cv2.imwrite(str(path), image)
    return str(path)


def appearance_values(color=(95, 95, 95)):
    embedding, model, version = create_embedding(synthetic_vehicle_image(color))
    return {
        "vehicle_color": "gray" if color[0] == color[1] == color[2] else None,
        "appearance_embedding": serialize_embedding(embedding),
        "appearance_model": model,
        "appearance_embedding_version": version,
        "appearance_quality": 10.0,
    }


class PlateRulesTests(unittest.TestCase):
    def test_requested_corrections_share_one_parser(self):
        for raw, expected in [("6J01AB876B", "GJ01AB8768"),
                              ("GJ01ABOSZD", "GJ01AB0520"), ("GJ6JAB1234", "GJ06AB1234")]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_plate_text(raw).normalized_text, expected)
                self.assertEqual(pipeline.clean_plate_text(raw)[0], expected)

    def test_no_silent_truncation(self):
        for raw in ("GJ01AB12345", "XXGJ01AB1234", "GJ01AB1234XYZ"):
            self.assertIsNone(normalize_plate_text(raw).normalized_text)
            self.assertIsNone(pipeline.clean_plate_text(raw)[0])

    def test_attached_and_separate_ind(self):
        for raw in ("INDGJ01AB1234", "IND GJ 01 AB 1234", "INDIAGJ01AB1234"):
            self.assertEqual(pipeline.clean_plate_text(raw)[0], "GJ01AB1234")
        self.assertFalse(pipeline._is_hsrp_noise("IND123"))
        self.assertTrue(pipeline._is_hsrp_noise("IND"))
        self.assertEqual(strip_hsrp_noise("IND123"), "IND123")

    def test_bh_and_diplomatic(self):
        for raw in ("22BH1234AA", "123CD4567", "10CC123", "12UN1234", "1CD1A"):
            self.assertTrue(normalize_plate_text(raw).valid_format, raw)
        self.assertFalse(normalize_plate_text("BH22AA1234").valid_format)

    def test_military_arrow_survives(self):
        arrow = chr(8593)
        self.assertEqual(normalize_plate_text(arrow+"22B123456M").normalized_text, arrow+"22B123456M")
        self.assertEqual(normalize_plate_text("22B"+arrow+"123456M").normalized_text, arrow+"22B123456M")
        self.assertFalse(normalize_plate_text("22B123456M").valid_format)

    def test_stacked_blocks(self):
        blocks = [dict(raw_text="876B", x=80, y=60), dict(raw_text="6J01", x=20, y=0),
                  dict(raw_text="IND", x=0, y=25), dict(raw_text="AB", x=20, y=60)]
        self.assertEqual(pipeline._spatially_join_ocr_blocks(blocks)[0], "GJ01AB8768")


class ImageTests(unittest.TestCase):
    def test_padding_and_edge_detection(self):
        image = np.zeros((200, 500, 3), np.uint8)
        crop, edges = padded_plate_crop(image, (100, 60, 200, 40))
        self.assertGreater(crop.shape[1], 200)
        self.assertGreater(crop.shape[0], 40)
        self.assertEqual(edges, [])
        crop, edges = padded_plate_crop(image, (0, 0, 100, 40))
        self.assertIn("left", edges)
        self.assertIn("top", edges)

    def test_size_normalization(self):
        for shape in ((20, 100, 3), (300, 1500, 3)):
            self.assertEqual(normalize_size(np.zeros(shape, np.uint8)).shape, (96, 480, 3))

    def test_quality_variants(self):
        image = np.full((80, 320, 3), 25, np.uint8)
        cv2.putText(image, "GJ01AB1234", (3, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (170,170,170), 2)
        variants, quality = enhanced_variants(image)
        self.assertIn("low_light", quality["flags"])
        self.assertIn("night_gamma", [name for name, im in variants])
        self.assertIn("shadow_normalized", [name for name, im in variants])
        self.assertTrue(all(im.dtype == np.uint8 and im.ndim == 3 for name, im in variants))

    def test_unrecoverable_image(self):
        self.assertTrue(assess_quality(np.full((60,200,3), 255, np.uint8))["unreadable"])
        with patch.object(pipeline, "get_ocr_reader") as reader:
            self.assertEqual(pipeline.read_plate_text(np.zeros((60,200,3), np.uint8)), [])
            reader.assert_not_called()

    def test_motion_and_glare_variants(self):
        image = np.zeros((96, 400, 3), np.uint8)
        cv2.putText(image, "GJ01AB1234", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2)
        blurred = cv2.GaussianBlur(image, (13, 13), 4)
        names = [name for name, _ in enhanced_variants(blurred)[0]]
        self.assertIn("motion_0", names)
        bright = 255-image
        self.assertIn("highlight_compressed", [name for name, _ in enhanced_variants(bright)[0]])

    def test_vehicle_appearance_extraction_succeeds_on_valid_crop(self):
        path = write_temp_image("appearance_valid.jpg", synthetic_vehicle_image((95, 95, 95)))
        result = analyze_vehicle_appearance(path, dict(x=112, y=158, width=146, height=40))
        self.assertTrue(result.available, result.error)
        self.assertIsNotNone(result.crop)
        self.assertIsNotNone(result.embedding)
        self.assertEqual(result.embedding_model, "opencv-hsv-hog-v1")
        self.assertEqual(result.embedding_version, "appearance-v1")
        self.assertGreater(result.quality, 0)

    def test_invalid_empty_vehicle_crop_is_safe(self):
        self.assertIsNone(normalize_vehicle_crop(np.zeros((4, 4, 3), np.uint8)))
        result = analyze_vehicle_appearance(str(pathlib.Path(TEMP.name) / "missing.jpg"))
        self.assertFalse(result.available)
        self.assertEqual(result.error, "image_unreadable")

    def test_vehicle_type_and_color_can_be_unavailable_without_breaking(self):
        path = write_temp_image("appearance_ambiguous.jpg", synthetic_vehicle_image((80, 120, 160)))
        result = analyze_vehicle_appearance(path)
        self.assertIsNone(result.vehicle_type)
        self.assertIn(result.vehicle_color, {None, "gray", "white", "yellow", "red", "green", "blue", "black"})

    def test_embedding_format_and_normalization(self):
        embedding, model, version = create_embedding(synthetic_vehicle_image((95, 95, 95)))
        serialized = serialize_embedding(embedding)
        parsed = deserialize_embedding(serialized)
        self.assertEqual(model, "opencv-hsv-hog-v1")
        self.assertEqual(version, "appearance-v1")
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(parsed), 32)
        self.assertAlmostEqual(float(np.linalg.norm(parsed)), 1.0, places=4)

    def test_appearance_similarity_high_for_identical_and_lower_for_different(self):
        gray_embedding = serialize_embedding(create_embedding(synthetic_vehicle_image((95, 95, 95)))[0])
        repeat_embedding = serialize_embedding(create_embedding(synthetic_vehicle_image((95, 95, 95)))[0])
        red_embedding = serialize_embedding(create_embedding(synthetic_vehicle_image((10, 10, 210)))[0])
        self.assertGreaterEqual(appearance_similarity(gray_embedding, repeat_embedding), .99)
        self.assertLess(appearance_similarity(gray_embedding, red_embedding), appearance_similarity(gray_embedding, repeat_embedding))

    def test_partial_history_stays_on_track(self):
        history = PartialPlateHistory()
        self.assertEqual(history.observe(1, "GJ01AB", 10), [])
        self.assertEqual(history.observe(2, "01AB1234", 11), [])
        self.assertEqual(history.observe(1, "01AB1234", 12), ["GJ01AB1234"])
        self.assertEqual(history.observe(1, "01AB1234", 40), [])

    def test_largest_box_compatibility_and_separate_candidates(self):
        image = np.zeros((300, 700, 3), np.uint8)
        boxes = [dict(x=30,y=60,width=100,height=40,layout="single_line",confidence=.9),
                 dict(x=240,y=60,width=300,height=70,layout="single_line",confidence=.9)]
        one = dict(text="GJ01AB1234", confidence=.9, status="ok", valid_format=True)
        two = dict(text="GJ01AB5678", confidence=.95, status="ok", valid_format=True)
        with patch.object(pipeline.cv2, "imread", return_value=image), patch.object(pipeline, "locate_plates", return_value=boxes), patch.object(pipeline, "read_plate_text", side_effect=[[one.copy() for _ in range(3)], [two]]), patch.object(pipeline, "_save_debug_crop", return_value=None):
            text, conf, status, encoded = pipeline.process_image("mock.jpg")
        self.assertEqual(text, "GJ01AB5678")
        self.assertEqual({item["detection_id"] for item in json.loads(encoded) if "detection_id" in item}, {1,2})

    def test_latest_frame_buffer(self):
        buffer = pipeline.LatestFrameBuffer()
        for i in range(10):
            buffer.put(i)
        self.assertEqual(buffer.get(), 9)
        self.assertIsNone(buffer.get(.001))

    def test_invalid_detector_results_do_not_block_fallback(self):
        image = np.zeros((300,700,3),np.uint8)
        boxes = [dict(x=100,y=60,width=30,height=20,confidence=.9)]
        noise = dict(text="R",confidence=.6,status="PENDING_REVIEW",valid_format=False)
        valid = dict(text="GJ01AB1234",confidence=.95,status="ok",valid_format=True)
        regions = [("contour_1",image,dict(x=100,y=80,width=200,height=50)),
                   ("contour_2",image,dict(x=0,y=160,width=300,height=80))]
        with patch.object(pipeline.cv2,"imread",return_value=image), patch.object(pipeline,"locate_plates",return_value=boxes), patch.object(pipeline,"fallback_ocr_regions",return_value=regions), patch.object(pipeline,"read_plate_text",side_effect=[[noise.copy()],[noise.copy()],[valid.copy()]]), patch.object(pipeline,"_save_debug_crop",return_value=None):
            text,confidence,status,encoded = pipeline.process_image("mock.jpg")
        self.assertEqual(text,"GJ01AB1234")
        self.assertEqual(status,"PENDING_REVIEW")
        recovered = next(c for c in json.loads(encoded) if c.get("text")==text)
        self.assertTrue(recovered["partial"])
        self.assertEqual(recovered["frame_edges"],["left"])
        self.assertLessEqual(confidence,.79)

    def test_perspective_corners_preserved(self):
        corners = np.array([[10,10],[210,25],[185,90],[5,80]], np.float32)
        ordered = pipeline._order_quad_points(corners)
        self.assertEqual({tuple(p) for p in ordered}, {tuple(p) for p in corners})

    def test_tiled_detection_restores_original_coordinates_and_nms(self):
        def result(x, width=160):
            box = MagicMock(xyxy=np.array([[x,100,x+width,140]]),conf=np.array([.9]))
            box.cls = np.array([0])
            return [MagicMock(boxes=[box],names={0:"plate"})]
        detector = MagicMock()
        detector.predict.side_effect = [result(1100),result(1100),result(76),result(1080,80)]
        with patch.object(pipeline,"get_plate_detector",return_value=detector):
            boxes = pipeline.locate_plates(np.zeros((900,2400,3),np.uint8))
        self.assertEqual(detector.predict.call_count,4)
        self.assertEqual({(b["x"],b["width"]) for b in boxes},{(1100,160),(2200,80)})

    def test_live_fallback_does_not_persist_cropped_read(self):
        from app.main import StreamNode
        node = StreamNode.__new__(StreamNode)
        node.sv = MagicMock()
        node.tracker = MagicMock()
        node.tracker.update_with_detections.return_value = MagicMock(xyxy=[])
        node.track_last_seen = {}
        node.track_last_event_at = {}
        node.lock = __import__("threading").Lock()
        node.camera_id = "CAM01"
        node._enqueue_live_event = MagicMock()
        image = np.zeros((300,700,3),np.uint8)
        box = dict(x=0,y=80,width=200,height=50)
        item = dict(text="GJ01AB1234",confidence=.95,status="ok",valid_format=True)
        with patch("app.main.locate_plates",return_value=[]), patch("app.main.fallback_ocr_regions",return_value=[("contour_1",image,box)]), patch("app.main.read_plate_text",return_value=[item]):
            node._process_frame(image,1)
        self.assertEqual(node.latest["detections"],[])
        node._enqueue_live_event.assert_not_called()

    def test_live_partial_cache_is_replaced_by_complete_observation(self):
        from app.main import StreamNode
        node = StreamNode.__new__(StreamNode)
        node.track_read_cache = {}
        node.track_last_ocr_at = {}
        node.track_max_confidence = {}
        image = np.zeros((80,320,3),np.uint8)
        cv2.putText(image,"GJ01AB1234",(5,55),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        item = dict(text="GJ01AB1234",confidence=.95,status="ok",valid_format=True)
        with patch("app.main.read_plate_text",side_effect=[[item.copy()],[item.copy()]]) as reader:
            _, first, _ = node._read_tracked_plate(1,image,frame_edges=["left"])
            self.assertEqual(first["status"],"PENDING_REVIEW")
            _, complete, cached = node._read_tracked_plate(1,image,frame_edges=[])
            self.assertFalse(cached)
            self.assertEqual(complete["status"],"ok")
            self.assertEqual(reader.call_count,2)

    def test_live_best_frame_survives_degraded_read(self):
        from app.main import StreamNode
        node = StreamNode.__new__(StreamNode)
        good = dict(text="GJ01AB1234",confidence=.95,status="ok",valid_format=True)
        node.track_read_cache = {1:dict(candidates=[good],best=good)}
        node.track_last_ocr_at = {1:0}
        node.track_max_confidence = {1:100}
        worse = dict(text="GJ01",confidence=.4,status="PENDING_REVIEW",valid_format=False)
        quality = dict(score=10,unreadable=False)
        with patch("app.main.read_plate_text",return_value=[worse]), patch("app.main.assess_quality",return_value=quality):
            _,best,cached = node._read_tracked_plate(1,np.zeros((80,300,3),np.uint8))
        self.assertTrue(cached)
        self.assertEqual(best["text"],"GJ01AB1234")


class APITests(unittest.TestCase):
    def setUp(self):
        with SessionLocal() as db:
            db.query(CameraRoadConnection).delete()
            db.query(HotlistNotification).delete()
            db.query(HotlistAlert).delete()
            db.query(HotlistEntry).delete()
            db.query(VehicleMatchCandidate).delete()
            db.query(PlateEvent).delete()
            db.query(Vehicle).delete()
            db.commit()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_camera_validation_unknown_and_zero(self):
        for coordinates in (dict(lat=91,lng=2), dict(lat=2), dict(lat="NaN",lng=2)):
            response = self.client.post("/api/cameras", json=dict(camera_id="INVALID", **coordinates))
            self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/cameras", json=dict(camera_id="UNKNOWN"))
        self.assertEqual(response.status_code, 200)
        cameras = self.client.get("/api/cameras").json()
        unknown = next(c for c in cameras if c["camera_id"] == "UNKNOWN")
        self.assertIsNone(unknown["lat"])
        self.assertFalse(unknown["location_known"])
        self.assertEqual(self.client.patch("/api/cameras/UNKNOWN", json={"lat":0,"lng":0}).status_code, 200)
        unknown = next(c for c in self.client.get("/api/cameras").json() if c["camera_id"] == "UNKNOWN")
        self.assertTrue(unknown["location_known"])
        self.assertEqual(unknown["lat"], 0)
        self.client.patch("/api/cameras/UNKNOWN", json={"lat":None,"lng":None})

    def test_seed_preserves_location(self):
        self.client.patch("/api/cameras/CAM01", json={"lat":20,"lng":70})
        seed()
        camera = next(c for c in self.client.get("/api/cameras").json() if c["camera_id"]=="CAM01")
        self.assertEqual(camera["lat"], 20)

    def test_configured_default_location(self):
        with patch.dict(os.environ, {"ANPR_DEFAULT_LAT":"20","ANPR_DEFAULT_LNG":"70"}):
            self.assertEqual(resolve_location({}), (20,70,True))
        with patch.dict(os.environ, {"ANPR_DEFAULT_LAT":"NaN","ANPR_DEFAULT_LNG":"70"}):
            with self.assertRaises(ValueError):
                resolve_location({})

    def test_camera_road_connection_crud_and_lookup(self):
        self.client.patch("/api/cameras/CAM01", json={"lat":23.0,"lng":72.0})
        self.client.patch("/api/cameras/CAM02", json={"lat":23.05,"lng":72.05})
        body = {
            "source_camera_id":"CAM01",
            "destination_camera_id":"CAM02",
            "distance_meters":4200,
            "direction":"north-east",
            "road_name":"Demo road",
            "road_type":"arterial",
        }
        created = self.client.post("/api/camera-road-connections", json=body)
        self.assertEqual(created.status_code, 201, created.text)
        data = created.json()
        self.assertEqual(data["source_camera_id"], "CAM01")
        self.assertEqual(data["destination_camera_id"], "CAM02")
        self.assertEqual(data["distance_meters"], 4200)
        self.assertEqual(data["direction"], "NE")
        self.assertEqual(data["distance_source"], "manual")
        self.assertIsNotNone(data["straight_line_reference"])
        network = self.client.get("/api/camera-road-network").json()
        self.assertEqual(len(network["connections"]), 1)
        outgoing = self.client.get("/api/cameras/CAM01/road-connections").json()
        self.assertEqual(outgoing["connections"][0]["destination_camera_id"], "CAM02")
        specific = self.client.get("/api/camera-road-connections/CAM01/CAM02").json()
        self.assertEqual(specific["road_name"], "Demo road")

    def test_camera_road_connection_validation_and_duplicates(self):
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":0,
        }).status_code, 422)
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":100,
            "direction":"sideways",
        }).status_code, 422)
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAMXX","destination_camera_id":"CAM02","distance_meters":100,
        }).status_code, 404)
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM01","distance_meters":100,
        }).status_code, 422)
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        }).status_code, 201)
        self.assertEqual(self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4300,
        }).status_code, 409)

    def test_camera_road_connection_update_disable_and_transition(self):
        created = self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
            "direction":"E","provider":"manual",
        }).json()
        updated = self.client.patch(f"/api/camera-road-connections/{created['id']}", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4300,
            "direction":"NE","provider":"manual",
        }).json()
        self.assertEqual(updated["distance_meters"], 4300)
        self.assertEqual(updated["direction"], "NE")
        with SessionLocal() as db:
            transition = camera_transition(db, "CAM01", "CAM02")
            self.assertTrue(transition["can_transition"])
            self.assertEqual(transition["distance_meters"], 4300)
        disabled = self.client.delete(f"/api/camera-road-connections/{created['id']}").json()
        self.assertFalse(disabled["active"])
        self.assertEqual(self.client.get("/api/camera-road-network").json()["connections"], [])
        self.assertEqual(len(self.client.get("/api/camera-road-network?include_inactive=true").json()["connections"]), 1)
        with SessionLocal() as db:
            transition = camera_transition(db, "CAM01", "CAM02")
            self.assertFalse(transition["can_transition"])

    def test_trajectory_includes_configured_road_transition(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
            "direction":"NE","road_name":"Demo corridor",
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime.utcnow()-dt.timedelta(minutes=10)
            db.get(PlateEvent, second.id).timestamp = dt.datetime.utcnow()-dt.timedelta(minutes=1)
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["hops"][1]["road_distance_meters"], 4200)
        self.assertEqual(route["hops"][1]["road_distance_source"], "manual")
        self.assertEqual(route["hops"][1]["road_direction"], "NE")
        self.assertTrue(route["hops"][1]["road_transition"]["can_transition"])

    def test_travel_time_and_speed_calculation(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = start
            db.get(PlateEvent, second.id).timestamp = start + dt.timedelta(seconds=378)
            db.commit()
            segment = calculate_travel_segment(db, db.get(PlateEvent, first.id), db.get(PlateEvent, second.id))
        self.assertEqual(segment["travel_time_seconds"], 378)
        self.assertAlmostEqual(segment["estimated_speed_kmh"], 40.0, places=2)
        self.assertAlmostEqual(segment["estimated_speed_mps"], 11.111, places=3)
        self.assertEqual(segment["speed_status"], "calculated")

    def test_zero_and_negative_travel_time_rejected(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        stamp = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            left, right = db.get(PlateEvent, first.id), db.get(PlateEvent, second.id)
            left.timestamp = stamp
            right.timestamp = stamp
            self.assertEqual(calculate_travel_segment(db, left, right)["speed_status"], "invalid_time")
            right.timestamp = stamp - dt.timedelta(seconds=1)
            self.assertEqual(calculate_travel_segment(db, left, right)["speed_status"], "invalid_time")

    def test_missing_and_inactive_road_connection_are_unavailable(self):
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, second.id).timestamp = dt.datetime(2026, 9, 13, 10, 5, 0)
            db.commit()
            segment = calculate_travel_segment(db, db.get(PlateEvent, first.id), db.get(PlateEvent, second.id))
            self.assertEqual(segment["speed_status"], "missing_road_connection")
        created = self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        }).json()
        self.client.delete(f"/api/camera-road-connections/{created['id']}")
        with SessionLocal() as db:
            segment = calculate_travel_segment(db, db.get(PlateEvent, first.id), db.get(PlateEvent, second.id))
            self.assertEqual(segment["speed_status"], "missing_road_connection")

    def test_missing_distance_and_invalid_timestamp_handling(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            left, right = db.get(PlateEvent, first.id), db.get(PlateEvent, second.id)
            left.timestamp = None
            right.timestamp = dt.datetime(2026, 9, 13, 10, 5, 0)
            self.assertEqual(calculate_travel_segment(db, left, right)["speed_status"], "invalid_time")
            left.timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            connection = db.query(CameraRoadConnection).filter_by(source_camera_id="CAM01",
                                                                  destination_camera_id="CAM02").one()
            connection.distance_meters = None
            with db.no_autoflush:
                segment = calculate_travel_segment(db, left, right)
            self.assertEqual(segment["speed_status"], "invalid_distance")

    def test_trajectory_response_contains_speed_fields(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = start
            db.get(PlateEvent, second.id).timestamp = start + dt.timedelta(seconds=378)
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        hop = route["hops"][1]
        self.assertTrue(hop["speed_available"])
        self.assertEqual(hop["speed_status"], "calculated")
        self.assertEqual(hop["travel_time_seconds"], 378)
        self.assertAlmostEqual(hop["estimated_speed_kmh"], 40.0, places=2)
        self.assertEqual(route["speed_summary"]["valid_speed_segments"], 1)

    def test_vehicle_speed_history_average_ignores_unavailable_segments(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        events = [
            persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM03",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
        ]
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            for index, event in enumerate(events):
                db.get(PlateEvent, event.id).timestamp = start + [dt.timedelta(seconds=0),
                                                                  dt.timedelta(seconds=378),
                                                                  dt.timedelta(seconds=600)][index]
            db.commit()
            history = vehicle_speed_history(db, events[0].vehicle_id)
        self.assertEqual(history["summary"]["valid_speed_segments"], 1)
        self.assertEqual(history["summary"]["unavailable_speed_segments"], 1)
        self.assertAlmostEqual(history["summary"]["average_estimated_speed_kmh"], 40.0, places=2)
        response = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/speed-history")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["segments"]), 2)

    def test_speed_history_preserves_vehicle_id_and_invalid_ocr_rules(self):
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        vehicle_id = first.vehicle_id
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        self.client.get(f"/api/vehicles/{vehicle_id}/speed-history")
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, vehicle_id)
            self.assertEqual(db.get(PlateEvent, second.id).vehicle_id, vehicle_id)
        persist_plate_event(dict(camera_id="CAM03",plate_text="D",confidence=.99,status="ok"))
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 1)

    def test_concurrent_duplicate_filter(self):
        value = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.9,status="ok")
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: persist_plate_event(value), range(12)))
        self.assertEqual(len({event.id for event,duplicate in results}), 1)
        self.assertEqual(sum(not duplicate for event,duplicate in results), 1)
        other, duplicate = persist_plate_event({**value, "camera_id":"CAM02"})
        self.assertFalse(duplicate)

    def test_duplicate_window_expiry_and_upgrade(self):
        values = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.6,status="PENDING_REVIEW")
        original, _ = persist_plate_event(values)
        improved, duplicate = persist_plate_event({**values, "confidence":.95, "status":"ok"})
        self.assertTrue(duplicate)
        self.assertEqual(improved.status, "ok")
        with SessionLocal() as db:
            db.get(PlateEvent, original.id).timestamp = dt.datetime.utcnow()-dt.timedelta(seconds=60)
            db.commit()
        newer, duplicate = persist_plate_event(values)
        self.assertFalse(duplicate)
        self.assertNotEqual(newer.id, original.id)

    def test_stream_writer_and_upload_share_deduplication(self):
        from app.database import AsyncPlateEventWriter
        values = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok")
        first,_ = persist_plate_event(values)
        async def write_stream():
            writer = AsyncPlateEventWriter()
            await writer.start()
            await writer.queue.put({**values,"track_id":88})
            await writer.stop()
            self.assertEqual(writer.failures,0)
        asyncio.run(write_stream())
        with SessionLocal() as db:
            self.assertEqual(db.query(PlateEvent).count(),1)
            self.assertEqual(db.query(PlateEvent).first().id,first.id)

    def test_global_vehicle_id_creation_and_cross_camera_history(self):
        first, duplicate = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",
                                                    confidence=.95,status="ok",bbox_x=1,bbox_y=2,
                                                    bbox_width=30,bbox_height=10,track_id="track-a"))
        self.assertFalse(duplicate)
        second, duplicate = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",
                                                     confidence=.96,status="ok"))
        self.assertFalse(duplicate)
        self.assertIsNotNone(first.vehicle_id)
        self.assertEqual(first.vehicle_id, second.vehicle_id)
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(),1)
            vehicle = db.query(Vehicle).first()
            self.assertEqual(vehicle.vehicle_id, first.vehicle_id)
            self.assertEqual(vehicle.primary_plate_text,"GJ01AB1234")
        lookup = self.client.get("/api/vehicles/lookup",params={"plate_text":"GJ01AB1234"}).json()
        self.assertEqual(lookup["global_vehicle_id"],first.vehicle_id)
        self.assertEqual(lookup["sightings"],2)
        by_id = self.client.get("/api/vehicles/lookup",params={"global_vehicle_id":first.vehicle_id}).json()
        self.assertEqual(by_id["plate_text"],"GJ01AB1234")
        history = self.client.get(f"/api/vehicles/{first.vehicle_id}/history").json()
        self.assertEqual([item["camera_id"] for item in history["observations"]],["CAM01","CAM02"])
        self.assertEqual(history["observations"][0]["bbox"],{"x":1.0,"y":2.0,"width":30.0,"height":10.0})
        observations = self.client.get(f"/api/vehicles/{first.vehicle_id}/observations").json()
        self.assertEqual(len(observations["observations"]),2)

    def test_scan_adds_appearance_without_changing_global_vehicle_id(self):
        item = candidate("GJ01AB1234", dict(x=112, y=158, width=146, height=40), 1)
        output = (item["text"], .94, "ok", json.dumps([item]))
        image = encode_jpeg(synthetic_vehicle_image((96, 96, 96), item["text"]))
        with patch("app.main.process_image", return_value=output):
            first = self.client.post("/api/scan", data={"camera_id":"CAM01"},
                                     files={"file":("vehicle.jpg",image,"image/jpeg")}).json()
            second = self.client.post("/api/scan", data={"camera_id":"CAM02"},
                                      files={"file":("vehicle.jpg",image,"image/jpeg")}).json()
        self.assertTrue(first["appearance_available"])
        self.assertEqual(first["global_vehicle_id"], second["global_vehicle_id"])
        with SessionLocal() as db:
            events = db.query(PlateEvent).order_by(PlateEvent.id.asc()).all()
            self.assertEqual(db.query(Vehicle).count(), 1)
            self.assertTrue(all(event.appearance_embedding for event in events))
            self.assertEqual([event.camera_id for event in events], ["CAM01", "CAM02"])
        history = self.client.get(f"/api/vehicles/{first['global_vehicle_id']}/history").json()
        self.assertTrue(history["observations"][0]["appearance_available"])
        self.assertEqual(history["observations"][0]["camera_id"], "CAM01")

    def test_unreadable_plate_can_store_appearance_without_fake_plate(self):
        unreadable = dict(text=None, raw_text="", confidence=0.0, status="PENDING_REVIEW",
                          valid_format=False, bbox=dict(x=112,y=158,width=146,height=40), detection_id=1)
        image = encode_jpeg(synthetic_vehicle_image((20, 20, 200)))
        with patch("app.main.process_image", return_value=(None, 0.0, "PENDING_REVIEW", json.dumps([unreadable]))):
            response = self.client.post("/api/scan", data={"camera_id":"CAM04"},
                                        files={"file":("unreadable.jpg",image,"image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body["plate_text"])
        self.assertIsNone(body["global_vehicle_id"])
        self.assertEqual(body["camera_id"], "CAM04")
        self.assertTrue(body["appearance_available"])
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 0)
            event = db.query(PlateEvent).first()
            self.assertIsNone(event.plate_text)
            self.assertIsNone(event.vehicle_id)
            self.assertIsNotNone(event.appearance_embedding)

    def test_invalid_garbage_ocr_can_store_appearance_without_vehicle(self):
        junk = dict(text="D", raw_text="D", confidence=.99, status="ok", valid_format=False,
                    bbox=dict(x=112,y=158,width=146,height=40), detection_id=1)
        image = encode_jpeg(synthetic_vehicle_image((95, 95, 95)))
        with patch("app.main.process_image", return_value=("D", .99, "ok", json.dumps([junk]))):
            body = self.client.post("/api/scan", data={"camera_id":"CAM03"},
                                    files={"file":("junk.jpg",image,"image/jpeg")}).json()
        self.assertEqual(body["plate_text"], "D")
        self.assertEqual(body["status"], "PENDING_REVIEW")
        self.assertIsNone(body["global_vehicle_id"])
        self.assertTrue(body["appearance_available"])
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 0)
            self.assertIsNone(db.query(PlateEvent).first().vehicle_id)

    def test_event_appearance_api_and_similarity_endpoint(self):
        image_a = encode_jpeg(synthetic_vehicle_image((95, 95, 95), "GJ01AB1234"))
        image_b = encode_jpeg(synthetic_vehicle_image((10, 10, 210), "GJ01AB5678"))
        item_a = candidate("GJ01AB1234", dict(x=112,y=158,width=146,height=40), 1)
        item_b = candidate("GJ01AB5678", dict(x=112,y=158,width=146,height=40), 1)
        with patch("app.main.process_image", side_effect=[
            (item_a["text"], .94, "ok", json.dumps([item_a])),
            (item_b["text"], .94, "ok", json.dumps([item_b])),
        ]):
            first = self.client.post("/api/scan", data={"camera_id":"CAM01"},
                                     files={"file":("a.jpg",image_a,"image/jpeg")}).json()
            second = self.client.post("/api/scan", data={"camera_id":"CAM02"},
                                      files={"file":("b.jpg",image_b,"image/jpeg")}).json()
        appearance = self.client.get(f"/api/events/{first['event_id']}/appearance").json()
        self.assertTrue(appearance["appearance_available"])
        self.assertEqual(appearance["camera_id"], "CAM01")
        self.assertNotIn("appearance_embedding", appearance)
        similarity = self.client.get("/api/appearance/similarity", params={
            "left_event_id": first["event_id"],
            "right_event_id": second["event_id"],
        }).json()
        self.assertTrue(similarity["left_available"])
        self.assertTrue(similarity["right_available"])
        self.assertIsNotNone(similarity["appearance_similarity"])
        self.assertIsNone(similarity["same_vehicle_decision"])

    def test_matching_same_valid_plate_high_confidence(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        second = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB1234",
                                          confidence=.96, status="ok", **appearance_values()))[0]
        self.assertEqual(first.vehicle_id, second.vehicle_id)
        with SessionLocal() as db:
            match = db.query(VehicleMatchCandidate).one()
            self.assertEqual(match.state, "HIGH_CONFIDENCE")
            self.assertEqual(match.decision, "confirmed_existing")
            self.assertEqual(match.candidate_vehicle_id, first.vehicle_id)
            evidence = json.loads(match.plate_evidence)
            self.assertTrue(evidence["used"])
            self.assertTrue(evidence["exact"])

    def test_matching_similar_appearance_different_plates_requires_review(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        second = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB5678",
                                          confidence=.95, status="ok", **appearance_values()))[0]
        self.assertNotEqual(first.vehicle_id, second.vehicle_id)
        with SessionLocal() as db:
            match = db.query(VehicleMatchCandidate).one()
            self.assertEqual(match.state, "MEDIUM_CONFIDENCE")
            self.assertEqual(match.review_status, "pending")
            self.assertNotEqual(db.get(PlateEvent, first.id).vehicle_id, db.get(PlateEvent, second.id).vehicle_id)

    def test_matching_unreadable_plate_similar_appearance_candidate(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        unreadable = persist_plate_event(dict(camera_id="CAM03", plate_text=None,
                                             confidence=0, status="PENDING_REVIEW", **appearance_values()))[0]
        self.assertIsNone(unreadable.vehicle_id)
        with SessionLocal() as db:
            match = db.query(VehicleMatchCandidate).one()
            self.assertEqual(match.state, "MEDIUM_CONFIDENCE")
            self.assertEqual(match.candidate_vehicle_id, first.vehicle_id)
            self.assertFalse(json.loads(match.plate_evidence)["used"])

    def test_matching_unreadable_different_appearance_low_confidence(self):
        persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                 confidence=.95, status="ok", **appearance_values((95,95,95))))[0]
        persist_plate_event(dict(camera_id="CAM02", plate_text=None, confidence=0,
                                 status="PENDING_REVIEW", **appearance_values((10,10,210))))[0]
        with SessionLocal() as db:
            self.assertEqual(db.query(VehicleMatchCandidate).count(), 0)

    def test_matching_multiple_existing_vehicle_ids_are_not_merged(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        second = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB5678",
                                          confidence=.95, status="ok", **appearance_values()))[0]
        self.assertNotEqual(first.vehicle_id, second.vehicle_id)
        match_id = self.client.get("/api/matches/review").json()["items"][0]["match_id"]
        response = self.client.post(f"/api/matches/{match_id}/review", json={"action":"accept"})
        self.assertEqual(response.status_code, 409)
        with SessionLocal() as db:
            self.assertNotEqual(db.get(PlateEvent, first.id).vehicle_id, db.get(PlateEvent, second.id).vehicle_id)

    def test_matching_invalid_ocr_is_never_plate_evidence(self):
        left = PlateEvent(camera_id="CAM01", plate_text="D", confidence=.99, status="ok",
                          timestamp=dt.datetime.utcnow(), **appearance_values())
        right = PlateEvent(camera_id="CAM02", plate_text="112", confidence=.99, status="ok",
                           timestamp=dt.datetime.utcnow()+dt.timedelta(seconds=30), **appearance_values())
        comparison = compare_observations(left, right)
        self.assertFalse(comparison["plate_evidence"]["used"])
        self.assertNotIn("plate_similarity", comparison["factors_used"])

    def test_matching_missing_embedding_uses_valid_plate_safely(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB1234",
                                          confidence=.95, status="ok"))[0]
        with SessionLocal() as db:
            match = db.query(VehicleMatchCandidate).one()
            self.assertEqual(match.state, "HIGH_CONFIDENCE")
            self.assertIsNone(match.appearance_similarity)
            self.assertEqual(match.candidate_vehicle_id, first.vehicle_id)
        self.assertEqual(first.vehicle_id, second.vehicle_id)

    def test_matching_invalid_temporal_order_is_low_confidence(self):
        later = PlateEvent(camera_id="CAM02", plate_text="GJ01AB1234", confidence=.95, status="ok",
                           timestamp=dt.datetime.utcnow()+dt.timedelta(minutes=10), **appearance_values())
        earlier = PlateEvent(camera_id="CAM01", plate_text="GJ01AB1234", confidence=.95, status="ok",
                             timestamp=dt.datetime.utcnow(), **appearance_values())
        comparison = compare_observations(later, earlier)
        self.assertLess(comparison["time_delta_seconds"], 0)
        self.assertEqual(comparison["state"], "LOW_CONFIDENCE")

    def test_matching_review_accept_and_reject(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        second = persist_plate_event(dict(camera_id="CAM02", plate_text=None, confidence=0,
                                          status="PENDING_REVIEW", **appearance_values()))[0]
        pending = self.client.get("/api/matches/review").json()["items"]
        self.assertEqual(len(pending), 1)
        accepted = self.client.post(f"/api/matches/{pending[0]['match_id']}/review",
                                    json={"action":"accept"}).json()
        self.assertEqual(accepted["review_status"], "accepted")
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, second.id).vehicle_id, first.vehicle_id)

        third = persist_plate_event(dict(camera_id="CAM03", plate_text=None, confidence=0,
                                         status="PENDING_REVIEW", **appearance_values()))[0]
        reject_match = next(item for item in self.client.get("/api/matches/review").json()["items"]
                            if item["observation_b_id"] == third.id)
        rejected = self.client.post(f"/api/matches/{reject_match['match_id']}/review",
                                    json={"action":"reject"}).json()
        self.assertEqual(rejected["review_status"], "rejected")
        self.assertEqual(rejected["decision"], "rejected")

    def test_global_vehicle_history_is_chronological(self):
        older = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        newer = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, older.id).timestamp = dt.datetime.utcnow()-dt.timedelta(minutes=10)
            db.get(PlateEvent, newer.id).timestamp = dt.datetime.utcnow()-dt.timedelta(minutes=1)
            db.commit()
        history = self.client.get(f"/api/vehicles/{older.vehicle_id}/history").json()
        self.assertEqual([item["event_id"] for item in history["observations"]],[older.id,newer.id])

    def test_invalid_ocr_does_not_create_global_vehicle(self):
        persist_plate_event(dict(camera_id="CAM01",plate_text="D",confidence=.99,status="ok"))
        persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.79,status="PENDING_REVIEW"))
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(),0)
            self.assertTrue(all(event.vehicle_id is None for event in db.query(PlateEvent).all()))
        self.assertEqual(self.client.get("/api/vehicles/lookup",params={"plate_text":"D"}).status_code,404)

    def test_vehicle_list_and_events_include_global_vehicle_id(self):
        event, _ = persist_plate_event(dict(camera_id="CAM03",plate_text="GJ01HV8768",confidence=.96,status="ok"))
        vehicles = self.client.get("/api/vehicles").json()
        self.assertEqual(vehicles[0]["global_vehicle_id"],event.vehicle_id)
        self.assertEqual(vehicles[0]["plate_text"],"GJ01HV8768")
        events = self.client.get("/api/events").json()
        self.assertEqual(events[0]["global_vehicle_id"],event.vehicle_id)
        self.assertEqual(events[0]["camera_id"],"CAM03")

    def test_upload_persists_all_and_largest_mode(self):
        box1 = dict(x=20,y=40,width=100,height=40)
        box2 = dict(x=200,y=40,width=300,height=60)
        items = [candidate("GJ01AB1234",box1,1), candidate("GJ01AB5678",box2,2)]
        output = ("GJ01AB5678",.94,"ok",json.dumps(items))
        with patch("app.main.process_image", return_value=output):
            result = self.client.post("/api/scan", data={"camera_id":"CAM01"}, files={"file":("image.jpg",b"test","image/jpeg")})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(len(result.json()["detections"]), 2)
            self.assertEqual(result.json()["plate_text"], "GJ01AB5678")
            again = self.client.post("/api/process-frame", data={"camera_id":"CAM01"}, files={"frame":("image.jpg",b"test","image/jpeg")})
            self.assertTrue(all(d["duplicate"] for d in again.json()["detections"]))
            largest = self.client.post("/api/scan?selection=largest",data={"camera_id":"CAM02"}, files={"file":("image.jpg",b"test","image/jpeg")})
            self.assertEqual(len(largest.json()["detections"]), 1)
            self.assertEqual(largest.json()["plate_text"],"GJ01AB5678")

    def test_auto_frame_rejects_incomplete_ocr(self):
        junk = dict(text="D",raw_text="D",confidence=.79,status="PENDING_REVIEW",
                    valid_format=False,bbox=dict(x=20,y=40,width=120,height=40),detection_id=1)
        with patch("app.main.process_image", return_value=("D",.79,"PENDING_REVIEW",json.dumps([junk]))):
            response = self.client.post("/api/process-frame",data={"camera_id":"CAM03"},files={"frame":("image.jpg",b"test","image/jpeg")})
        self.assertEqual(response.status_code,200,response.text)
        body = response.json()
        self.assertIsNone(body["event_id"])
        self.assertEqual(body["camera_id"],"CAM03")
        self.assertIsNone(body["plate_text"])
        self.assertIsNone(body["raw_text"])
        self.assertEqual(body["confidence"],0.0)
        self.assertEqual(body["status"],"scanning")
        self.assertFalse(body["needs_review"])
        self.assertEqual(body["detections"],[])
        with SessionLocal() as db:
            self.assertEqual(db.query(PlateEvent).count(),0)

    def test_vehicle_list_only_shows_valid_confirmed_plates(self):
        with SessionLocal() as db:
            db.add_all([
                PlateEvent(camera_id="CAM01",plate_text="ALTO",status="ok",confidence=.99),
                PlateEvent(camera_id="CAM01",plate_text="GJ01",status="PENDING_REVIEW",confidence=.79),
                PlateEvent(camera_id="CAM02",plate_text="GJ01HV8768",status="ok",confidence=.96),
            ])
            db.commit()
        vehicles = self.client.get("/api/vehicles").json()
        self.assertEqual([vehicle["plate_text"] for vehicle in vehicles],["GJ01HV8768"])
        self.assertEqual(vehicles[0]["last_camera"],"CAM02")

    def test_frame_edge_requires_review(self):
        item = candidate("GJ01AB1234",dict(x=0,y=0,width=100,height=40),1)
        item["partial"] = True
        with patch("app.main.process_image", return_value=(item["text"],.94,"ok",json.dumps([item]))):
            result = self.client.post("/api/scan",data={"camera_id":"CAM01"},files={"file":("image.jpg",b"test","image/jpeg")}).json()
            self.assertTrue(result["needs_review"])
            self.assertEqual(result["status"], "PENDING_REVIEW")

    def test_filtered_search_matches_filtered_trajectory(self):
        now = dt.datetime.utcnow()
        self.client.patch("/api/cameras/CAM01",json={"lat":23,"lng":72})
        self.client.patch("/api/cameras/CAM02",json={"lat":24,"lng":73})
        with SessionLocal() as db:
            for cam, minutes in (("CAM01",5),("CAM01",120),("CAM02",5)):
                db.add(PlateEvent(camera_id=cam,plate_text="GJ01AB1234",status="ok",confidence=.95,timestamp=now-dt.timedelta(minutes=minutes)))
            db.commit()
        params = dict(start=(now-dt.timedelta(minutes=30)).isoformat()+"Z",end=now.isoformat()+"Z",lat=23,lng=72,radius_m=1000)
        search = self.client.get("/api/search/plates",params={"q":"GJ01AB1234",**params}).json()
        route = self.client.get("/api/trajectory/GJ01AB1234",params=params).json()
        self.assertEqual(len(search["matches"]), 1)
        self.assertEqual([m["event_id"] for m in search["matches"]], [h["event_id"] for h in route["hops"]])

    def test_bad_filters(self):
        for params in (dict(lat=91,lng=2),dict(lat=2),dict(start="2026-09-10",end="2026-09-01")):
            self.assertEqual(self.client.get("/api/search/plates",params={"q":"GJ01AB1234",**params}).status_code,422)
        self.assertEqual(self.client.get("/api/search/plates",params={"q":"GJ01AB1234","radius_m":0}).status_code,422)

    def test_search_does_not_drop_matches_after_1000_rows(self):
        now = dt.datetime.utcnow()
        with SessionLocal() as db:
            db.add_all([PlateEvent(camera_id="CAM01",plate_text="MH02ZZ9999",timestamp=now) for _ in range(1001)])
            db.add(PlateEvent(camera_id="CAM01",plate_text="GJ01AB1234",timestamp=now-dt.timedelta(minutes=1)))
            db.commit()
        data = self.client.get("/api/search/plates",params={"q":"GJ01AB1234","similarity":1}).json()
        self.assertEqual(len(data["matches"]),1)

    def test_event_socket_heartbeat(self):
        with self.client.websocket_connect("/ws/events") as socket:
            self.assertEqual(socket.receive_json()["type"],"connected")
            socket.send_json({"action":"ping"})
            self.assertEqual(socket.receive_json()["type"],"pong")

    def add_hotlist(self, **overrides):
        body = dict(plate_text="GJ01AB1234",reason="Reported stolen",reference="TEST-1",category="stolen")
        response = self.client.post("/api/hotlist",json={**body,**overrides})
        self.assertEqual(response.status_code,201,response.text)
        return response.json()

    def test_hotlist_crud_validation_and_expiry(self):
        entry = self.add_hotlist(plate_text="GJ 01 AB 1234")
        self.assertEqual(entry["plate_text"],"GJ01AB1234")
        self.assertEqual(self.client.post("/api/hotlist",json=dict(plate_text="GJ01AB1234",reason="duplicate")).status_code,409)
        for body in (dict(plate_text="GJ01",reason="test"),dict(plate_text="GJ01AB1234",reason=" "),dict(plate_text="GJ01AB1234",reason="test",category="invalid")):
            self.assertEqual(self.client.post("/api/hotlist",json=body).status_code,422)
        response = self.client.put(f"/api/hotlist/{entry['id']}",json=dict(plate_text=entry["plate_text"],reason="Resolved",active=False))
        self.assertEqual(response.status_code,200)
        values = dict(camera_id="CAM01",plate_text=entry["plate_text"],confidence=.95,status="ok")
        persist_plate_event(values)
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["total"],0)
        self.client.put(f"/api/hotlist/{entry['id']}",json=dict(plate_text=entry["plate_text"],reason="Expired",expires_at="2000-01-01T00:00:00Z"))
        persist_plate_event(values)
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["total"],0)
        self.assertTrue(self.client.get("/api/hotlist").json()["items"][0]["expired"])
        self.assertEqual(self.client.delete(f"/api/hotlist/{entry['id']}").status_code,200)

    def test_hotlist_exact_match_concurrency_and_camera_scope(self):
        self.add_hotlist()
        values = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _:persist_plate_event(values),range(12)))
        alerts = self.client.get("/api/hotlist-alerts").json()
        self.assertEqual(alerts["total"],1)
        self.assertEqual(alerts["items"][0]["match_status"],"matched")
        persist_plate_event({**values,"plate_text":"GJ01AB1235"})
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["total"],1)
        persist_plate_event({**values,"camera_id":"CAM02"})
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["total"],2)
        self.assertEqual(self.client.get("/api/hotlist-alerts?camera_id=CAM01").json()["total"],1)

    def test_hotlist_review_upgrade_and_acknowledgement(self):
        self.add_hotlist()
        values = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.65,status="PENDING_REVIEW")
        persist_plate_event(values)
        first = self.client.get("/api/hotlist-alerts").json()["items"][0]
        self.assertEqual(first["match_status"],"review")
        ack = self.client.post(f"/api/hotlist-alerts/{first['id']}/acknowledge").json()
        self.assertIsNotNone(ack["acknowledged_at"])
        self.assertEqual(self.client.get("/api/hotlist-alerts?unacknowledged=true").json()["total"],0)
        persist_plate_event({**values,"confidence":.96,"status":"ok"})
        upgraded = self.client.get("/api/hotlist-alerts?unacknowledged=true").json()["items"][0]
        self.assertEqual(upgraded["id"],first["id"])
        self.assertEqual(upgraded["match_status"],"matched")
        self.assertGreater(upgraded["revision"],ack["revision"])

    def test_hotlist_scan_response_and_partial_caution(self):
        self.add_hotlist()
        item = candidate("GJ01AB1234",dict(x=0,y=10,width=120,height=40),1)
        item["partial"] = True
        with patch("app.main.process_image",return_value=(item["text"],.94,"ok",json.dumps([item]))):
            response = self.client.post("/api/scan",data={"camera_id":"CAM01"},files={"file":("test.jpg",b"test","image/jpeg")})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()["hotlist_alerts"][0]["match_status"],"review")

    def test_hotlist_corrected_plate_retracts_old_alert(self):
        self.add_hotlist()
        event,_ = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))
        response = self.client.post(f"/api/events/{event.id}/correct",data={"corrected_text":"GJ01AB5678"})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["items"][0]["match_status"],"retracted")
        self.assertEqual(self.client.get("/api/hotlist-alerts?unacknowledged=true").json()["total"],0)
        self.client.post(f"/api/events/{event.id}/correct",data={"corrected_text":"GJ01AB1234"})
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["items"][0]["match_status"],"matched")

    def test_hotlist_clear_history_preserves_alert_snapshots(self):
        self.add_hotlist()
        values = dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok")
        persist_plate_event(values)
        self.assertEqual(self.client.delete("/api/events").status_code,200)
        old = self.client.get("/api/hotlist-alerts").json()["items"][0]
        self.assertIsNone(old["event_id"])
        persist_plate_event(values)
        self.assertEqual(self.client.get("/api/hotlist-alerts").json()["total"],2)
        self.assertEqual(self.client.get("/api/hotlist").json()["total"],1)

    def test_hotlist_stream_writer_broadcasts_durable_alert(self):
        from app.main import event_writer
        self.add_hotlist()
        with self.client.websocket_connect("/ws/events") as socket:
            socket.receive_json()
            event_writer.enqueue_from_thread(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))
            message = socket.receive_json()
            self.assertEqual(message["type"],"hotlist_alert")
            self.assertEqual(message["alert"]["plate_text"],"GJ01AB1234")
        self.assertEqual(self.client.get("/api/hotlist-alerts?unacknowledged=true").json()["total"],1)

    def test_hotlist_correction_creates_new_match(self):
        self.add_hotlist()
        event,_ = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1235",confidence=.65,status="PENDING_REVIEW"))
        with self.client.websocket_connect("/ws/events") as socket:
            socket.receive_json()
            socket.send_json(dict(action="correct_event",event_id=event.id,corrected_text="GJ01AB1234"))
            messages = [socket.receive_json(),socket.receive_json()]
            self.assertEqual({m["type"] for m in messages},{"event_corrected","hotlist_alert"})


if __name__ == "__main__":
    unittest.main()
