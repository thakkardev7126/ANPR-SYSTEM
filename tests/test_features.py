"""Regression coverage for the 13-feature implementation; never opens the demo DB."""
import os
import sys
import tempfile
import pathlib
import json
import base64
import datetime as dt
import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

TEMP = tempfile.TemporaryDirectory(prefix="anpr-tests-")
os.environ["ANPR_DATABASE_PATH"] = str(pathlib.Path(TEMP.name) / "tests.db")
os.environ["ANPR_UPLOAD_DIR"] = str(pathlib.Path(TEMP.name) / "uploads")
os.environ["ANPR_ORIGINAL_EVIDENCE_DIR"] = str(pathlib.Path(TEMP.name) / "evidence" / "originals")
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
from app.database import SessionLocal, PlateEvent, Camera, CameraRoadConnection, Vehicle, VehicleMatchCandidate, VehicleAnomaly, PlateSuspicionEvent, RouteAnomalyEvent, persist_plate_event, engine, HotlistEntry, HotlistAlert, HotlistNotification, PCRVehicle, EnforcementIncident, AuditLog, User, EdgeDevice, EdgeObservation, RetentionRun, TrafficJunction, SignalPhase, SignalRecommendation, SignalSimulationState
from app.audit import append_audit_log, verify_audit_chain
from app.auth import hash_password
from app.crypto import decrypt_sensitive, encrypt_sensitive, generate_key
from app.privacy import create_privacy_safe_derivative
from app.main import app
from app.seed_cameras import seed
from app.road_network import camera_transition
from app.travel_time import calculate_travel_segment, vehicle_speed_history
from app.trajectory import build_vehicle_trajectory
from app.vehicle_appearance import (
    analyze_vehicle_appearance,
    appearance_similarity,
    create_embedding,
    deserialize_embedding,
    normalize_vehicle_crop,
    serialize_embedding,
)
from app.vehicle_matching import compare_observations, process_observation_matches
from app.edge import FrameSamplingConfig, process_due_edge_observations, run_retention


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
    def test_privacy_face_and_person_blur_preserves_plate_region(self):
        image = synthetic_vehicle_image()
        cv2.circle(image, (42, 40), 16, (220, 190, 170), -1)
        source = write_temp_image("privacy_source.jpg", image)
        target = str(pathlib.Path(TEMP.name) / "privacy_safe.jpg")
        plate_box = {"x": 112, "y": 158, "width": 146, "height": 40}
        with patch("app.privacy.detect_faces", return_value=[{"x": 24, "y": 22, "width": 36, "height": 36}]), \
             patch("app.privacy.detect_persons", return_value=[{"x": 18, "y": 60, "width": 68, "height": 130}]):
            result = create_privacy_safe_derivative(source, target, [plate_box])
        self.assertEqual(result.status, "processed")
        masked = cv2.imread(target)
        self.assertIsNotNone(masked)
        self.assertGreater(np.mean(np.abs(masked[60:180, 18:86].astype(int) - image[60:180, 18:86].astype(int))), 1)
        self.assertLess(np.mean(np.abs(masked[158:198, 112:258].astype(int) - image[158:198, 112:258].astype(int))), 2)

    def test_privacy_no_person_image_and_detector_failure_are_safe(self):
        image = synthetic_vehicle_image()
        source = write_temp_image("privacy_empty.jpg", image)
        target = str(pathlib.Path(TEMP.name) / "privacy_empty_safe.jpg")
        with patch("app.privacy.detect_faces", return_value=[]), patch("app.privacy.detect_persons", return_value=[]):
            result = create_privacy_safe_derivative(source, target, [])
        self.assertEqual(result.status, "processed")
        self.assertTrue(pathlib.Path(target).exists())
        with patch("app.privacy.detect_faces", side_effect=RuntimeError("detector unavailable")):
            failed = create_privacy_safe_derivative(source, str(pathlib.Path(TEMP.name) / "privacy_failed.jpg"), [])
        self.assertEqual(failed.status, "failed")

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


class Phase11CryptoTests(unittest.TestCase):
    def test_aes_256_gcm_encrypts_decrypts_and_uses_unique_nonces(self):
        key = generate_key()
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": key, "ANPR_ENCRYPTION_KEY_VERSION": "v1"}, clear=False):
            first = encrypt_sensitive("sensitive-path.jpg", resource_type="plate_event", purpose="original_image_path")
            second = encrypt_sensitive("sensitive-path.jpg", resource_type="plate_event", purpose="original_image_path")
            self.assertEqual(first.key_version, "v1")
            self.assertNotEqual(first.nonce, second.nonce)
            self.assertNotIn("sensitive-path", first.ciphertext)
            self.assertEqual(
                decrypt_sensitive(first.ciphertext, first.nonce, first.key_version,
                                  resource_type="plate_event", purpose="original_image_path"),
                "sensitive-path.jpg",
            )

    def test_aes_256_gcm_rejects_tampering_wrong_key_and_missing_key(self):
        key = generate_key()
        wrong_key = generate_key()
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": key, "ANPR_ENCRYPTION_KEY_VERSION": "v1"}, clear=False):
            encrypted = encrypt_sensitive("secret", resource_type="plate_event", purpose="privacy_metadata")
            tampered = encrypted.ciphertext[:-2] + "AA"
            with self.assertRaises(Exception):
                decrypt_sensitive(tampered, encrypted.nonce, encrypted.key_version,
                                  resource_type="plate_event", purpose="privacy_metadata")
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": wrong_key, "ANPR_ENCRYPTION_KEY_VERSION": "v1"}, clear=False):
            with self.assertRaises(Exception):
                decrypt_sensitive(encrypted.ciphertext, encrypted.nonce, encrypted.key_version,
                                  resource_type="plate_event", purpose="privacy_metadata")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception):
                encrypt_sensitive("secret", resource_type="plate_event", purpose="privacy_metadata")

    def test_key_version_selection_supports_rotation_without_reencrypting_old_data(self):
        v1 = generate_key()
        v2 = generate_key()
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": v1, "ANPR_ENCRYPTION_KEY_VERSION": "v1"}, clear=False):
            old = encrypt_sensitive("old", resource_type="plate_event", purpose="original_image_path")
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": v2, "ANPR_ENCRYPTION_KEY_VERSION": "v2", "ANPR_ENCRYPTION_KEY_V1": v1}, clear=False):
            new = encrypt_sensitive("new", resource_type="plate_event", purpose="original_image_path")
            self.assertEqual(new.key_version, "v2")
            self.assertEqual(decrypt_sensitive(old.ciphertext, old.nonce, old.key_version,
                                               resource_type="plate_event", purpose="original_image_path"), "old")
            self.assertEqual(decrypt_sensitive(new.ciphertext, new.nonce, new.key_version,
                                               resource_type="plate_event", purpose="original_image_path"), "new")


class APITests(unittest.TestCase):
    def setUp(self):
        with SessionLocal() as db:
            db.query(AuditLog).delete()
            db.query(SignalSimulationState).delete()
            db.query(SignalRecommendation).delete()
            db.query(SignalPhase).delete()
            db.query(TrafficJunction).delete()
            db.query(RetentionRun).delete()
            db.query(EdgeObservation).delete()
            db.query(EdgeDevice).delete()
            db.query(CameraRoadConnection).delete()
            db.query(EnforcementIncident).delete()
            db.query(PCRVehicle).delete()
            db.query(HotlistNotification).delete()
            db.query(HotlistAlert).delete()
            db.query(HotlistEntry).delete()
            db.query(RouteAnomalyEvent).delete()
            db.query(PlateSuspicionEvent).delete()
            db.query(VehicleAnomaly).delete()
            db.query(VehicleMatchCandidate).delete()
            db.query(PlateEvent).delete()
            db.query(Vehicle).delete()
            db.commit()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.login_as("admin_demo", "AdminDemo!2026")

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def login_as(self, username, password):
        response = self.client.post("/api/auth/login", json={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["access_token"]
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        return token

    def unauthenticated_client(self):
        context = TestClient(app)
        return context, context.__enter__()

    def test_phase8_auth_rejects_unauthenticated_invalid_and_inactive_users(self):
        context, anonymous = self.unauthenticated_client()
        try:
            self.assertEqual(anonymous.get("/api/vehicles").status_code, 401)
            self.assertEqual(anonymous.get("/api/audit").status_code, 401)
            self.assertEqual(anonymous.post("/api/auth/login", json={
                "username": "admin_demo",
                "password": "wrong-password",
            }).status_code, 401)
        finally:
            context.__exit__(None, None, None)
        with SessionLocal() as db:
            db.add(User(username="inactive_demo", password_hash=hash_password("InactiveDemo!2026"),
                        role="TRAFFIC_OPERATOR", active=False,
                        created_at=dt.datetime.utcnow(), updated_at=dt.datetime.utcnow()))
            db.commit()
        self.assertEqual(self.client.post("/api/auth/login", json={
            "username": "inactive_demo",
            "password": "InactiveDemo!2026",
        }).status_code, 401)

    def test_phase11_cookie_session_authenticates_api_websocket_and_logout(self):
        context, browser = self.unauthenticated_client()
        try:
            self.assertEqual(browser.get("/api/cameras").status_code, 401)
            login = browser.post("/api/auth/login", json={
                "username": "admin_demo",
                "password": "AdminDemo!2026",
            })
            self.assertEqual(login.status_code, 200, login.text)
            self.assertIn("anpr_session=", login.headers.get("set-cookie", ""))
            self.assertEqual(browser.get("/api/cameras").status_code, 200)
            with browser.websocket_connect("/ws/events") as socket:
                self.assertEqual(socket.receive_json()["type"], "connected")
            self.assertEqual(browser.post("/api/auth/logout").status_code, 200)
            self.assertEqual(browser.get("/api/cameras").status_code, 401)
        finally:
            context.__exit__(None, None, None)

    def test_phase11_passwords_are_hashed_and_not_returned_or_audited(self):
        response = self.client.post("/api/auth/login", json={"username": "admin_demo", "password": "AdminDemo!2026"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("AdminDemo!2026", response.text)
        self.assertNotIn("TrafficDemo!2026", response.text)
        with SessionLocal() as db:
            user = db.query(User).filter(User.username == "admin_demo").one()
            self.assertNotEqual(user.password_hash, "AdminDemo!2026")
            self.assertTrue(user.password_hash.startswith("pbkdf2_sha256$"))
            self.assertFalse(any("AdminDemo!2026" in (row.details or "") for row in db.query(AuditLog).all()))

    def test_phase11_security_headers_cookie_flags_and_cors(self):
        response = self.client.get("/api/system/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertIn("frame-ancestors 'none'", response.headers.get("content-security-policy", ""))
        self.assertNotIn("strict-transport-security", {k.lower(): v for k, v in response.headers.items()})
        with patch.dict(os.environ, {"ANPR_COOKIE_SECURE": "true", "ANPR_COOKIE_SAMESITE": "strict"}, clear=False):
            login = self.client.post("/api/auth/login", json={"username": "admin_demo", "password": "AdminDemo!2026"})
        cookie = login.headers.get("set-cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("SameSite=strict", cookie)

        allowed = self.client.options("/api/system/status", headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "GET",
        })
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://127.0.0.1:8000")
        self.assertNotEqual(allowed.headers.get("access-control-allow-origin"), "*")
        denied = self.client.options("/api/system/status", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        })
        self.assertNotEqual(denied.headers.get("access-control-allow-origin"), "*")

    def test_phase8_roles_have_expected_api_boundaries(self):
        self.assertEqual(self.client.get("/api/audit/verify").status_code, 200)
        self.assertEqual(self.client.get("/api/users").status_code, 200)

        self.login_as("traffic_demo", "TrafficDemo!2026")
        self.assertEqual(self.client.get("/api/traffic/dashboard").status_code, 200)
        self.assertEqual(self.client.get("/api/vehicles").status_code, 200)
        self.assertEqual(self.client.get("/api/audit").status_code, 403)
        self.assertEqual(self.client.get("/api/users").status_code, 403)
        self.assertEqual(self.client.post("/api/pcr", json={
            "pcr_id": "PCR-T-DENIED",
            "latitude": 23.0,
            "longitude": 72.0,
        }).status_code, 403)

        self.login_as("pcr_demo", "PcrDemo!2026")
        self.assertEqual(self.client.get("/api/pcr").status_code, 200)
        self.assertEqual(self.client.get("/api/incidents").status_code, 200)
        self.assertEqual(self.client.get("/api/audit").status_code, 403)
        self.assertEqual(self.client.get("/api/users").status_code, 403)
        self.assertEqual(self.client.get("/api/traffic/dashboard").status_code, 403)

    def test_phase8_privacy_safe_display_and_original_evidence_rbac(self):
        item = candidate("GJ01AB1234", dict(x=112, y=158, width=146, height=40), 1)
        with patch("app.main.process_image", return_value=(item["text"], .94, "ok", json.dumps([item]))), \
             patch("app.privacy.detect_faces", return_value=[{"x": 20, "y": 18, "width": 40, "height": 40}]), \
             patch("app.privacy.detect_persons", return_value=[]):
            response = self.client.post("/api/scan", data={"camera_id": "CAM01"},
                                        files={"file": ("privacy.jpg", encode_jpeg(synthetic_vehicle_image()), "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("_privacy", body["image_path"])
        self.assertTrue(body["original_image_available"])
        self.assertEqual(self.client.get(body["image_path"]).status_code, 200)
        self.assertEqual(self.client.get(body["original_image_path"]).status_code, 200)

        self.login_as("traffic_demo", "TrafficDemo!2026")
        events = self.client.get("/api/events").json()
        self.assertIn("_privacy", events[0]["image_path"])
        self.assertEqual(self.client.get(events[0]["image_path"]).status_code, 200)
        self.assertEqual(self.client.get(events[0]["original_image_path"]).status_code, 403)

    def test_phase11_encrypted_evidence_reference_is_not_plaintext_in_database(self):
        item = candidate("GJ01AB1234", dict(x=112, y=158, width=146, height=40), 1)
        key = generate_key()
        with patch.dict(os.environ, {"ANPR_ENCRYPTION_KEY": key, "ANPR_ENCRYPTION_KEY_VERSION": "v1"}, clear=False), \
             patch("app.main.process_image", return_value=(item["text"], .94, "ok", json.dumps([item]))), \
             patch("app.privacy.detect_faces", return_value=[]), patch("app.privacy.detect_persons", return_value=[]):
            response = self.client.post("/api/scan", data={"camera_id": "CAM01"},
                                        files={"file": ("encrypted.jpg", encode_jpeg(synthetic_vehicle_image()), "image/jpeg")})
            original_response = self.client.get(response.json()["original_image_path"])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["original_image_available"])
        self.assertEqual(original_response.status_code, 200)
        with SessionLocal() as db:
            row = db.get(PlateEvent, body["event_id"])
            self.assertIsNone(row.original_image_path)
            self.assertIsNotNone(row.original_image_path_ciphertext)
            self.assertIsNotNone(row.original_image_path_nonce)
            self.assertEqual(row.original_image_path_key_version, "v1")
            self.assertIsNone(row.privacy_metadata)
            self.assertIsNotNone(row.privacy_metadata_ciphertext)

    def test_phase11_upload_path_traversal_and_arbitrary_files_are_blocked(self):
        self.assertEqual(self.client.get("/uploads/../backend/app/auth.py").status_code, 404)
        self.assertEqual(self.client.get("/uploads/%2e%2e/backend/app/auth.py").status_code, 404)
        self.assertEqual(self.client.get("/api/evidence/999999/image").status_code, 404)

    def test_phase11_upload_size_limit_rejects_large_payloads(self):
        with patch.dict(os.environ, {"ANPR_MAX_UPLOAD_SIZE": "32"}, clear=False):
            response = self.client.post("/api/scan", data={"camera_id": "CAM01"},
                                        files={"file": ("large.jpg", b"x" * 64, "image/jpeg")})
        self.assertEqual(response.status_code, 413)

    def test_phase8_audit_log_hash_chain_and_tamper_detection(self):
        lookup = self.client.get("/api/vehicles")
        self.assertEqual(lookup.status_code, 200)
        with SessionLocal() as db:
            actions = [row.action for row in db.query(AuditLog).order_by(AuditLog.id.asc()).all()]
            self.assertIn("vehicle_lookup", actions)
            self.assertTrue(verify_audit_chain(db)["valid"])
            record = db.query(AuditLog).filter(AuditLog.action == "vehicle_lookup").first()
            self.assertEqual(record.username, "admin_demo")
            self.assertNotIn("AdminDemo!2026", json.dumps([row.details for row in db.query(AuditLog).all()]))
            record.reason = "tampered"
            db.commit()
            self.assertFalse(verify_audit_chain(db)["valid"])

    def test_phase8_audit_detects_deleted_reordered_and_fake_records(self):
        with SessionLocal() as db:
            db.query(AuditLog).delete()
            db.commit()
            append_audit_log(db, action="first", resource_type="test", success=True)
            append_audit_log(db, action="second", resource_type="test", success=True)
            append_audit_log(db, action="third", resource_type="test", success=True)
            db.commit()
            self.assertTrue(verify_audit_chain(db)["valid"])
            middle = db.query(AuditLog).filter(AuditLog.action == "second").one()
            db.delete(middle)
            db.commit()
            self.assertFalse(verify_audit_chain(db)["valid"])

        with SessionLocal() as db:
            db.query(AuditLog).delete()
            db.commit()
            append_audit_log(db, action="first", resource_type="test", success=True)
            append_audit_log(db, action="second", resource_type="test", success=True)
            db.commit()
            rows = db.query(AuditLog).order_by(AuditLog.id.asc()).all()
            rows[0].current_hash, rows[1].current_hash = rows[1].current_hash, rows[0].current_hash
            db.commit()
            self.assertFalse(verify_audit_chain(db)["valid"])

        with SessionLocal() as db:
            db.query(AuditLog).delete()
            db.commit()
            append_audit_log(db, action="first", resource_type="test", success=True)
            fake = AuditLog(audit_id="fake-audit-record", timestamp=dt.datetime.utcnow(),
                            action="fake", resource_type="test", success=True,
                            previous_hash="bad", current_hash="bad")
            db.add(fake)
            db.commit()
            self.assertFalse(verify_audit_chain(db)["valid"])

    def edge_observation(self, observation_id="edge-obs-1", camera_id="CAM01", plate_text="GJ01AB1234", **extra):
        payload = {
            "observation_id": observation_id,
            "camera_id": camera_id,
            "edge_device_id": "EDGE-DEMO-01",
            "frame_id": f"frame-{observation_id}",
            "sequence_number": 30,
            "capture_timestamp": dt.datetime.utcnow().isoformat(),
            "plate_text": plate_text,
            "ocr_confidence": .94 if plate_text else 0.0,
            "processing_status": "OK" if plate_text else "NO_PLATE",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "bbox": {"x": 112, "y": 158, "width": 146, "height": 40},
            "privacy_status": "metadata_only",
            "metadata": {"test": True},
        }
        payload.update(extra)
        return payload

    def test_phase9_frame_sampling_configuration(self):
        config = FrameSamplingConfig(source_fps=30, processing_fps=2, max_processing_fps=5)
        self.assertEqual(config.sampling_interval, 15)
        self.assertTrue(config.should_process(30))
        self.assertFalse(config.should_process(31))

    def test_phase9_edge_single_batch_ingestion_and_idempotency(self):
        first = self.client.post("/api/edge/observations", json=self.edge_observation())
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["status"], "PROCESSED")
        self.assertIsNotNone(first.json()["plate_event_id"])
        duplicate = self.client.post("/api/edge/observations", json=self.edge_observation())
        self.assertEqual(duplicate.status_code, 200, duplicate.text)
        self.assertTrue(duplicate.json()["duplicate"])
        batch = self.client.post("/api/edge/observations/batch", json={
            "observations": [
                self.edge_observation("edge-obs-2", "CAM02", "GJ01AB5678"),
                self.edge_observation("edge-obs-3", "CAM03", None),
            ]
        })
        self.assertEqual(batch.status_code, 200, batch.text)
        self.assertEqual(batch.json()["received"], 2)
        with SessionLocal() as db:
            self.assertEqual(db.query(EdgeObservation).count(), 3)
            self.assertEqual(db.query(PlateEvent).filter(PlateEvent.plate_text == "GJ01AB1234").count(), 1)
            self.assertEqual(db.query(PlateEvent).filter(PlateEvent.plate_text == "GJ01AB5678").count(), 1)

    def test_phase9_edge_validation_retry_and_health_status(self):
        bad = self.client.post("/api/edge/observations", json={"observation_id": "bad"})
        self.assertEqual(bad.status_code, 422)
        with patch("app.edge.persist_plate_event", side_effect=RuntimeError("database temporarily unavailable")):
            failed = self.client.post("/api/edge/observations", json=self.edge_observation("edge-fail-1"))
        self.assertEqual(failed.status_code, 200, failed.text)
        self.assertEqual(failed.json()["status"], "QUEUED")
        self.assertEqual(failed.json()["retry_count"], 1)
        self.assertEqual(len(process_due_edge_observations(limit=10)), 0)
        with SessionLocal() as db:
            row = db.query(EdgeObservation).filter(EdgeObservation.observation_id == "edge-fail-1").one()
            row.next_retry_at = dt.datetime.utcnow() - dt.timedelta(seconds=1)
            db.commit()
        self.assertEqual(len(process_due_edge_observations(limit=10)), 1)
        status = self.client.get("/api/edge/status").json()
        self.assertGreaterEqual(status["observations_received"], 1)
        self.assertIn(status["queue"]["mode"], {"local", "redis"})

    def test_phase9_scalability_simulation_10_100_500_cameras(self):
        for count in (10, 100, 500):
            response = self.client.post("/api/edge/simulation/start", json={
                "camera_count": count,
                "observation_rate": 1,
                "duration_seconds": .05,
                "duplicate_rate": .02,
                "failure_rate": .01,
                "source_fps": 30,
                "processing_fps": 5,
            })
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["label"], "SIMULATION / DEMO")
            self.assertEqual(body["metrics"]["camera_count"], count)
        status = self.client.get("/api/edge/status").json()
        self.assertGreaterEqual(status["registered_cameras"], 500)
        self.assertFalse(status["simulation"]["active"])

    def test_phase9_retention_dry_run_delete_and_missing_file(self):
        raw_dir = pathlib.Path(TEMP.name) / "edge-retention"
        raw_dir.mkdir(exist_ok=True)
        old_file = raw_dir / "old.jpg"
        old_file.write_bytes(b"old-image")
        missing_file = raw_dir / "missing.jpg"
        old_time = dt.datetime.utcnow() - dt.timedelta(days=31)
        with SessionLocal() as db:
            db.add(PlateEvent(camera_id="CAM01", plate_text="GJ01AB1234", confidence=.95, status="ok",
                              timestamp=old_time, original_image_path=str(old_file),
                              privacy_image_path="/uploads/privacy-old.jpg"))
            db.add(PlateEvent(camera_id="CAM02", plate_text="GJ01AB5678", confidence=.95, status="ok",
                              timestamp=old_time, original_image_path=str(missing_file)))
            db.commit()
        dry = self.client.post("/api/retention/run", json={"dry_run": True, "raw_image_days": 30}).json()
        self.assertEqual(dry["eligible_raw_images"], 2)
        self.assertTrue(old_file.exists())
        actual = self.client.post("/api/retention/run", json={"dry_run": False, "raw_image_days": 30}).json()
        self.assertEqual(actual["deleted_raw_images"], 1)
        self.assertEqual(actual["missing_raw_images"], 1)
        self.assertFalse(old_file.exists())
        archive_file = raw_dir / "archive-old.jpg"
        archive_file.write_bytes(b"archive-image")
        with SessionLocal() as db:
            db.add(PlateEvent(camera_id="CAM03", plate_text="GJ01AB9999", confidence=.95, status="ok",
                              timestamp=old_time, original_image_path=str(archive_file)))
            db.commit()
        archive_dir = raw_dir / "archive"
        with patch.dict(os.environ, {"ANPR_RETENTION_ARCHIVE_DIR": str(archive_dir)}):
            archived = run_retention(dry_run=False, raw_image_days=30, action="archive", actor="admin_demo")
        self.assertEqual(archived["archived_raw_images"], 1)
        self.assertFalse(archive_file.exists())
        self.assertEqual(len(list(archive_dir.iterdir())), 1)
        with SessionLocal() as db:
            self.assertEqual(db.query(PlateEvent).count(), 3)
            self.assertEqual(db.query(RetentionRun).count(), 3)
            self.assertTrue(verify_audit_chain(db)["valid"])

    def test_phase9_rbac_and_audit_for_edge_admin_actions(self):
        context, unauth = self.unauthenticated_client()
        try:
            self.assertEqual(unauth.get("/api/edge/status").status_code, 401)
            self.assertEqual(unauth.post("/api/edge/simulation/start", json={"camera_count": 10}).status_code, 401)
        finally:
            context.__exit__(None, None, None)
        self.assertEqual(self.client.get("/api/edge/status").status_code, 200)
        self.assertEqual(self.client.post("/api/edge/simulation/start", json={"camera_count": 10}).status_code, 200)
        self.login_as("traffic_demo", "TrafficDemo!2026")
        self.assertEqual(self.client.get("/api/edge/status").status_code, 200)
        self.assertEqual(self.client.post("/api/edge/simulation/start", json={"camera_count": 10}).status_code, 403)
        self.assertEqual(self.client.post("/api/retention/run", json={"dry_run": True}).status_code, 403)
        self.login_as("pcr_demo", "PcrDemo!2026")
        self.assertEqual(self.client.get("/api/edge/status").status_code, 403)
        self.login_as("admin_demo", "AdminDemo!2026")
        with SessionLocal() as db:
            actions = [row.action for row in db.query(AuditLog).all()]
            self.assertIn("edge_simulation_change", actions)
            self.assertIn("retention_execution", actions)
            self.assertTrue(verify_audit_chain(db)["valid"])

    def signal_junction_payload(self, junction_id="JUNC-TEST-01", active=True):
        return {
            "junction_id": junction_id,
            "name": "Test Smart Signal Junction",
            "camera_ids": ["CAM01", "CAM02", "CAM03", "CAM04"],
            "active": active,
            "controller_mode": "simulation",
        }

    def create_signal_junction_with_phases(self, junction_id="JUNC-TEST-01", min_green=5, max_green=30):
        response = self.client.post("/api/signals/junctions", json=self.signal_junction_payload(junction_id))
        self.assertIn(response.status_code, {200, 201}, response.text)
        for index, phase in enumerate((
            ("NORTH_SOUTH_GREEN", "NORTH_SOUTH", ["CAM01", "CAM02"]),
            ("EAST_WEST_GREEN", "EAST_WEST", ["CAM03", "CAM04"]),
        )):
            phase_response = self.client.post(f"/api/signals/junctions/{junction_id}/phases", json={
                "phase_id": phase[0],
                "movement": phase[1],
                "movement_camera_ids": phase[2],
                "min_green_seconds": min_green,
                "max_green_seconds": max_green,
                "yellow_seconds": 3,
                "all_red_seconds": 1,
                "display_order": index,
            })
            self.assertEqual(phase_response.status_code, 201, phase_response.text)
        return response.json()

    def seed_signal_demand(self):
        for index in range(8):
            persist_plate_event(dict(camera_id="CAM03", plate_text=f"GJ01AB{1200 + index}", confidence=.95, status="ok"), window_seconds=0)
            persist_plate_event(dict(camera_id="CAM04", plate_text=f"GJ01AB{2200 + index}", confidence=.95, status="ok"), window_seconds=0)
        for index in range(2):
            persist_plate_event(dict(camera_id="CAM01", plate_text=f"GJ01CD{3200 + index}", confidence=.95, status="ok"), window_seconds=0)

    def test_phase10_junction_configuration_and_validation(self):
        body = self.create_signal_junction_with_phases()
        self.assertEqual(body["junction_id"], "JUNC-TEST-01")
        fetched = self.client.get("/api/signals/junctions/JUNC-TEST-01")
        self.assertEqual(fetched.status_code, 200, fetched.text)
        self.assertEqual(len(fetched.json()["phases"]), 2)
        invalid = self.client.post("/api/signals/junctions/JUNC-TEST-01/phases", json={
            "phase_id": "BAD_PHASE",
            "movement": "BAD",
            "movement_camera_ids": ["CAM01"],
            "min_green_seconds": 40,
            "max_green_seconds": 10,
            "yellow_seconds": 0,
            "all_red_seconds": 0,
        })
        self.assertEqual(invalid.status_code, 422)
        inactive = self.client.patch("/api/signals/junctions/JUNC-TEST-01", json={"active": False})
        self.assertEqual(inactive.status_code, 200, inactive.text)
        recommendation = self.client.post("/api/signals/recommendation", json={"junction_id": "JUNC-TEST-01"})
        self.assertEqual(recommendation.status_code, 422)

    def test_phase10_demand_recommendation_fairness_and_safety(self):
        self.create_signal_junction_with_phases(min_green=6, max_green=35)
        self.seed_signal_demand()
        with SessionLocal() as db:
            phase = db.query(SignalPhase).filter(SignalPhase.phase_id == "NORTH_SOUTH_GREEN").one()
            phase.last_served_at = dt.datetime.utcnow() - dt.timedelta(seconds=240)
            db.commit()
        demand = self.client.get("/api/signals/junctions/JUNC-TEST-01/demand").json()
        self.assertIn(demand["data_quality"], {"LOW", "GOOD"})
        movement_demand = {item["movement"]: item for item in demand["movements"]}
        self.assertGreater(movement_demand["EAST_WEST"]["demand_score"], movement_demand["NORTH_SOUTH"]["demand_score"])
        response = self.client.post("/api/signals/recommendation", json={"junction_id": "JUNC-TEST-01", "cycle_seconds": 70})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["mode"], "RECOMMENDATION_ONLY")
        self.assertFalse(body["safety"]["physical_control"])
        self.assertTrue(body["safety"]["yellow_preserved"])
        self.assertTrue(body["safety"]["all_red_preserved"])
        phases = {item["movement"]: item for item in body["phases"]}
        for item in phases.values():
            self.assertGreaterEqual(item["recommended_green_seconds"], item["min_green_seconds"])
            self.assertLessEqual(item["recommended_green_seconds"], item["max_green_seconds"])
            self.assertGreaterEqual(item["yellow_seconds"], 3)
            self.assertGreaterEqual(item["all_red_seconds"], 1)
        self.assertGreaterEqual(phases["EAST_WEST"]["recommended_green_seconds"], phases["NORTH_SOUTH"]["recommended_green_seconds"])
        body_demand = {item["movement"]: item for item in body["demand"]["movements"]}
        self.assertIn("waiting_time", body_demand["NORTH_SOUTH"]["reasons"])
        self.assertTrue(body["explanation"]["summary"])

    def test_phase10_insufficient_data_returns_safe_default(self):
        self.create_signal_junction_with_phases()
        response = self.client.post("/api/signals/recommendation", json={"junction_id": "JUNC-TEST-01", "cycle_seconds": 60})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["data_quality"], "INSUFFICIENT_DATA")
        self.assertLessEqual(body["recommendation_reliability"], .3)
        greens = [phase["recommended_green_seconds"] for phase in body["phases"]]
        self.assertLessEqual(max(greens) - min(greens), 1)
        self.assertIn("insufficient_data", body["explanation"]["reason_codes"])

    def test_phase10_signal_simulator_and_controller_adapter(self):
        self.create_signal_junction_with_phases(min_green=5, max_green=20)
        recommendation = self.client.post("/api/signals/recommendation", json={"junction_id": "JUNC-TEST-01", "cycle_seconds": 30}).json()
        start = self.client.post("/api/signals/simulation/JUNC-TEST-01/start", json={
            "recommendation_id": recommendation["recommendation_id"],
        })
        self.assertEqual(start.status_code, 200, start.text)
        self.assertTrue(start.json()["active"])
        self.assertEqual(start.json()["phase_kind"], "GREEN")
        tick_green = self.client.post("/api/signals/simulation/JUNC-TEST-01/tick", json={"seconds": start.json()["remaining_seconds"]})
        self.assertEqual(tick_green.status_code, 200, tick_green.text)
        self.assertEqual(tick_green.json()["phase_kind"], "YELLOW")
        tick_yellow = self.client.post("/api/signals/simulation/JUNC-TEST-01/tick", json={"seconds": 3})
        self.assertEqual(tick_yellow.json()["phase_kind"], "ALL_RED")
        reset = self.client.post("/api/signals/simulation/JUNC-TEST-01/reset")
        self.assertFalse(reset.json()["active"])
        apply = self.client.post("/api/signals/controller/apply", json={"recommendation_id": recommendation["recommendation_id"]})
        self.assertEqual(apply.status_code, 200, apply.text)
        self.assertIn("SIMULATION MODE", apply.json()["mode"])
        self.assertFalse(apply.json()["physical_control"])

    def test_phase10_signal_rbac_and_audit(self):
        context, anonymous = self.unauthenticated_client()
        try:
            self.assertEqual(anonymous.get("/api/signals/junctions").status_code, 401)
        finally:
            context.__exit__(None, None, None)
        self.create_signal_junction_with_phases("JUNC-RBAC-01")
        self.assertEqual(self.client.get("/api/signals/junctions").status_code, 200)
        self.login_as("traffic_demo", "TrafficDemo!2026")
        self.assertEqual(self.client.get("/api/signals/junctions").status_code, 200)
        self.assertEqual(self.client.post("/api/signals/recommendation", json={"junction_id": "JUNC-RBAC-01"}).status_code, 200)
        self.assertEqual(self.client.post("/api/signals/junctions", json=self.signal_junction_payload("JUNC-DENIED-01")).status_code, 403)
        self.login_as("pcr_demo", "PcrDemo!2026")
        self.assertEqual(self.client.get("/api/signals/junctions").status_code, 403)
        self.login_as("admin_demo", "AdminDemo!2026")
        self.client.post("/api/signals/simulation/JUNC-RBAC-01/start", json={})
        with SessionLocal() as db:
            actions = [row.action for row in db.query(AuditLog).all()]
            self.assertIn("signal_configuration_change", actions)
            self.assertIn("signal_recommendation_generate", actions)
            self.assertIn("signal_simulation_change", actions)
            self.assertTrue(verify_audit_chain(db)["valid"])

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

    def test_complete_multi_camera_trajectory_summary_and_hops(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
            "direction":"NE",
        })
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM02","destination_camera_id":"CAM03","distance_meters":3800,
            "direction":"E",
        })
        events = [
            persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM03",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM04",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
        ]
        stamps = [
            dt.datetime(2026, 9, 13, 10, 0, 0),
            dt.datetime(2026, 9, 13, 10, 6, 18),
            dt.datetime(2026, 9, 13, 10, 12, 0),
            dt.datetime(2026, 9, 13, 10, 20, 0),
        ]
        with SessionLocal() as db:
            for event, stamp in zip(events, stamps):
                db.get(PlateEvent, event.id).timestamp = stamp
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual([item["camera_id"] for item in route["observations"]],
                         ["CAM01","CAM02","CAM03","CAM04"])
        self.assertEqual(len(route["trajectory_hops"]), 3)
        first, second, third = route["trajectory_hops"]
        self.assertEqual(first["source_camera_id"], "CAM01")
        self.assertEqual(first["destination_camera_id"], "CAM02")
        self.assertEqual(first["distance_meters"], 4200)
        self.assertEqual(first["travel_time_seconds"], 378)
        self.assertAlmostEqual(first["estimated_speed_kmh"], 40.0, places=2)
        self.assertEqual(first["continuity"], "connected")
        self.assertEqual(second["distance_meters"], 3800)
        self.assertEqual(second["travel_time_seconds"], 342)
        self.assertAlmostEqual(second["estimated_speed_kmh"], 40.0, places=2)
        self.assertEqual(second["continuity"], "connected")
        self.assertEqual(third["continuity"], "not_connected")
        self.assertEqual(third["speed_status"], "missing_road_connection")
        self.assertFalse(third["speed_available"])
        self.assertIsNone(third["distance_meters"])
        summary = route["summary"]
        self.assertEqual(summary["total_observations"], 4)
        self.assertEqual(summary["valid_segments"], 2)
        self.assertEqual(summary["unavailable_segments"], 1)
        self.assertEqual(summary["total_road_distance_meters"], 8000)
        self.assertEqual(summary["total_travel_time_seconds"], 720)
        self.assertAlmostEqual(summary["average_estimated_speed_kmh"], 40.0, places=2)

    def test_vehicle_trajectory_endpoint_and_history_compatibility(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, second.id).timestamp = dt.datetime(2026, 9, 13, 10, 6, 18)
            db.commit()
        history = self.client.get(f"/api/vehicles/{first.vehicle_id}/history").json()
        trajectory = self.client.get(f"/api/vehicles/{first.vehicle_id}/trajectory").json()
        self.assertEqual([item["event_id"] for item in history["observations"]],
                         [item["event_id"] for item in trajectory["observations"]])
        self.assertEqual(history["summary"]["valid_segments"], 1)
        self.assertEqual(trajectory["summary"]["valid_segments"], 1)
        self.assertEqual(trajectory["vehicle"]["global_vehicle_id"], first.vehicle_id)

    def test_trajectory_orders_out_of_order_insertion_and_same_timestamp(self):
        events = [
            persist_plate_event(dict(camera_id="CAM03",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
            persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0],
        ]
        same = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            db.get(PlateEvent, events[0].id).timestamp = dt.datetime(2026, 9, 13, 10, 10, 0)
            db.get(PlateEvent, events[1].id).timestamp = same
            db.get(PlateEvent, events[2].id).timestamp = same
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual([item["camera_id"] for item in route["observations"]],
                         ["CAM01","CAM02","CAM03"])
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "invalid_time")

    def test_complete_trajectory_single_missing_unknown_and_inactive_cases(self):
        single = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["summary"]["total_observations"], 1)
        self.assertEqual(route["trajectory_hops"], [])

        unknown = persist_plate_event(dict(camera_id="CAMXX",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, single.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, unknown.id).timestamp = dt.datetime(2026, 9, 13, 10, 5, 0)
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "invalid_observation")

        with SessionLocal() as db:
            db.get(PlateEvent, unknown.id).camera_id = "CAM02"
            db.get(PlateEvent, unknown.id).timestamp = None
            db.commit()
        route = self.client.get(f"/api/vehicles/{single.vehicle_id}/trajectory").json()
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "invalid_time")

        with SessionLocal() as db:
            db.get(PlateEvent, unknown.id).timestamp = dt.datetime(2026, 9, 13, 10, 5, 0)
            db.add(CameraRoadConnection(source_camera_id="CAM01", destination_camera_id="CAM02",
                                        distance_meters=4200, active=False,
                                        created_at=dt.datetime.utcnow(), updated_at=dt.datetime.utcnow()))
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "not_connected")

    def test_complete_trajectory_missing_road_distance_and_invalid_time(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, second.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "invalid_time")

        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, second.id).timestamp = dt.datetime(2026, 9, 13, 10, 5, 0)
            connection = db.query(CameraRoadConnection).one()
            connection.distance_meters = None
            with db.no_autoflush:
                route = build_vehicle_trajectory(db, first.vehicle_id)
        self.assertEqual(route["trajectory_hops"][0]["speed_status"], "invalid_distance")
        self.assertEqual(route["trajectory_hops"][0]["continuity"], "invalid_observation")

    def test_trajectory_preserves_matching_evidence_and_invalid_ocr_rules(self):
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",
                                         confidence=.95,status="ok", **appearance_values()))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",
                                          confidence=.95,status="ok", **appearance_values()))[0]
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = dt.datetime(2026, 9, 13, 10, 0, 0)
            db.get(PlateEvent, second.id).timestamp = dt.datetime(2026, 9, 13, 10, 6, 18)
            db.commit()
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        evidence = route["trajectory_hops"][0]["matching_evidence"]
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["decision"], "confirmed_existing")
        self.assertGreaterEqual(evidence["confidence"], .88)

        persist_plate_event(dict(camera_id="CAM03", plate_text="D", confidence=.99, status="ok",
                                 **appearance_values()))
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 1)

    def _fixed_pair(self, seconds):
        first = persist_plate_event(dict(camera_id="CAM01",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        second = persist_plate_event(dict(camera_id="CAM02",plate_text="GJ01AB1234",confidence=.95,status="ok"))[0]
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            db.get(PlateEvent, first.id).timestamp = start
            db.get(PlateEvent, second.id).timestamp = start + dt.timedelta(seconds=seconds)
            db.commit()
        return first, second

    def _clear_road_connections(self):
        with SessionLocal() as db:
            db.query(CameraRoadConnection).delete()
            db.commit()

    def test_impossible_travel_policy_normal_warning_and_impossible(self):
        self._fixed_pair(378)
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
            "road_type":"arterial",
        })
        normal = self.client.get("/api/trajectory/GJ01AB1234").json()["trajectory_hops"][0]
        self.assertEqual(normal["anomaly_status"], "normal")
        self.assertEqual(self.client.get("/api/anomalies").json()["total"], 0)

        self.client.delete("/api/events")
        self._clear_road_connections()
        first, _ = self._fixed_pair(151.2)
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        warning = self.client.get("/api/trajectory/GJ01AB1234").json()["trajectory_hops"][0]
        self.assertEqual(warning["anomaly_status"], "warning")
        self.assertEqual(warning["anomaly_severity"], "warning")
        self.assertAlmostEqual(warning["estimated_speed_kmh"], 100.0, places=1)
        self.assertEqual(self.client.get(f"/api/vehicles/{first.vehicle_id}/anomalies").json()["total"], 1)

        self.client.delete("/api/events")
        self._clear_road_connections()
        self._fixed_pair(60)
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
            "road_name":"Demo corridor",
        })
        impossible = self.client.get("/api/trajectory/GJ01AB1234").json()["trajectory_hops"][0]
        self.assertEqual(impossible["anomaly_status"], "impossible_travel")
        evidence = impossible["anomaly_evidence"]
        self.assertEqual(evidence["severity"], "critical")
        self.assertAlmostEqual(evidence["estimated_speed_kmh"], 252.0, places=1)
        self.assertEqual(evidence["allowed_speed_kmh"], 160.0)
        self.assertAlmostEqual(evidence["excess_ratio"], 1.575, places=3)
        self.assertIn("252.0 km/h", evidence["explanation"])
        anomalies = self.client.get("/api/anomalies", params={"anomaly_type":"impossible_travel"}).json()
        self.assertEqual(anomalies["total"], 1)
        self.assertEqual(anomalies["items"][0]["severity"], "critical")
        self.assertIn("252.0 km/h", anomalies["items"][0]["explanation"])

    def test_impossible_travel_ignores_unavailable_or_invalid_segments(self):
        self._fixed_pair(60)
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["anomaly_status"], "unavailable")
        self.assertEqual(self.client.get("/api/anomalies").json()["total"], 0)

        self.client.delete("/api/events")
        self._clear_road_connections()
        self._fixed_pair(60)
        created = self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        }).json()
        self.client.delete(f"/api/camera-road-connections/{created['id']}")
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["anomaly_status"], "unavailable")
        self.assertEqual(self.client.get("/api/anomalies").json()["total"], 0)

        self.client.delete("/api/events")
        self._clear_road_connections()
        self._fixed_pair(0)
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(route["trajectory_hops"][0]["speed_status"], "invalid_time")
        self.assertEqual(route["trajectory_hops"][0]["anomaly_status"], "unavailable")
        self.assertEqual(self.client.get("/api/anomalies").json()["total"], 0)

        self.client.delete("/api/events")
        self._clear_road_connections()
        first, second = self._fixed_pair(60)
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        with SessionLocal() as db:
            connection = db.query(CameraRoadConnection).one()
            connection.distance_meters = None
            with db.no_autoflush:
                route = build_vehicle_trajectory(db, first.vehicle_id)
        self.assertEqual(route["trajectory_hops"][0]["speed_status"], "invalid_distance")
        self.assertEqual(route["trajectory_hops"][0]["anomaly_status"], "unavailable")
        with SessionLocal() as db:
            self.assertEqual(db.query(VehicleAnomaly).count(), 0)

    def test_impossible_travel_duplicate_prevention_and_identity_safety(self):
        first, second = self._fixed_pair(60)
        vehicle_id = first.vehicle_id
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        self.client.get("/api/trajectory/GJ01AB1234")
        self.client.get("/api/trajectory/GJ01AB1234")
        anomalies = self.client.get("/api/anomalies").json()
        self.assertEqual(anomalies["total"], 1)
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, vehicle_id)
            self.assertEqual(db.get(PlateEvent, second.id).vehicle_id, vehicle_id)
        persist_plate_event(dict(camera_id="CAM03", plate_text="D", confidence=.99, status="ok"))
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 1)

    def _valid_event(self, camera_id="CAM01", plate_text="GJ01AB1234", **extra):
        values = dict(camera_id=camera_id, plate_text=plate_text, confidence=.95, status="ok")
        values.update(extra)
        return persist_plate_event(values)[0]

    def _set_timestamps(self, pairs):
        with SessionLocal() as db:
            for event, timestamp in pairs:
                db.get(PlateEvent, event.id).timestamp = timestamp
            db.commit()

    def _test_embedding(self, values):
        return serialize_embedding(values)

    def test_cloned_plate_normal_plate_remains_normal(self):
        first = self._valid_event("CAM01")
        second = self._valid_event("CAM02")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 6, 18)),
        ])
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        self.assertEqual(summary["classification"], "normal")
        self.assertEqual(summary["total"], 0)
        self.assertFalse(any(item["type"] == "impossible_travel" for item in summary["evaluated_evidence"]))

    def test_cloned_plate_reuses_impossible_travel_evidence(self):
        first = self._valid_event("CAM01")
        second = self._valid_event("CAM02")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 1, 0)),
        ])
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        self.assertEqual(summary["classification"], "suspicious")
        self.assertTrue(any(item["type"] == "impossible_travel" for item in summary["evaluated_evidence"]))
        self.assertEqual(self.client.get("/api/anomalies").json()["total"], 1)

    def test_cloned_plate_simultaneous_sighting_evidence(self):
        first = self._valid_event("CAM01")
        second = self._valid_event("CAM03")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 0, 5)),
        ])
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        self.assertEqual(summary["classification"], "suspicious")
        overlap = [item for item in summary["evaluated_evidence"] if item["type"] == "simultaneous_sighting"]
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["time_delta_seconds"], 5.0)

    def test_cloned_plate_appearance_type_and_color_conflict_evidence(self):
        first = self._valid_event("CAM01", appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                  appearance_model="test", appearance_embedding_version="test",
                                  vehicle_type="car", vehicle_color="white")
        second = self._valid_event("CAM04", appearance_embedding=self._test_embedding([0, 1, 0, 0]),
                                   appearance_model="test", appearance_embedding_version="test",
                                   vehicle_type="motorcycle", vehicle_color="black")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 10, 0)),
        ])
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        evidence_types = {item["type"] for item in summary["evaluated_evidence"]}
        self.assertIn("appearance_conflict", evidence_types)
        self.assertIn("vehicle_type_conflict", evidence_types)
        self.assertIn("vehicle_color_conflict", evidence_types)
        self.assertGreaterEqual(summary["suspicion_score"], 7.0)

    def test_cloned_plate_multiple_evidence_high_suspicion(self):
        first = self._valid_event("CAM01", appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                  appearance_model="test", appearance_embedding_version="test",
                                  vehicle_type="car")
        second = self._valid_event("CAM02", appearance_embedding=self._test_embedding([0, 1, 0, 0]),
                                   appearance_model="test", appearance_embedding_version="test",
                                   vehicle_type="motorcycle")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 1, 0)),
        ])
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        self.assertEqual(summary["classification"], "high_suspicion")
        evidence_types = {item["type"] for item in summary["evaluated_evidence"]}
        self.assertIn("impossible_travel", evidence_types)
        self.assertIn("appearance_conflict", evidence_types)
        self.assertIn("vehicle_type_conflict", evidence_types)
        self.assertFalse(summary["confirmed_cloned_plate"])

    def test_cloned_plate_weak_color_only_not_high_suspicion(self):
        first = self._valid_event("CAM01", vehicle_color="white")
        second = self._valid_event("CAM04", vehicle_color="black")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 10, 0)),
        ])
        summary = self.client.get("/api/plates/GJ01AB1234/suspicion").json()
        self.assertEqual(summary["classification"], "normal")
        self.assertEqual(summary["total"], 0)
        self.assertIn("vehicle_color_conflict", {item["type"] for item in summary["evaluated_evidence"]})

    def test_cloned_plate_invalid_ocr_not_used(self):
        persist_plate_event(dict(camera_id="CAM01", plate_text="D", confidence=.99, status="ok"))
        persist_plate_event(dict(camera_id="CAM02", plate_text="D", confidence=.99, status="ok"))
        summary = self.client.get("/api/plates/D/suspicion").json()
        self.assertEqual(summary["classification"], "normal")
        self.assertEqual(summary["evaluated_evidence"], [])
        self.assertEqual(self.client.get("/api/plate-suspicions").json()["total"], 0)
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 0)

    def test_cloned_plate_duplicate_evaluation_review_and_trajectory_fields(self):
        first = self._valid_event("CAM01", appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                  appearance_model="test", appearance_embedding_version="test")
        second = self._valid_event("CAM02", appearance_embedding=self._test_embedding([0, 1, 0, 0]),
                                   appearance_model="test", appearance_embedding_version="test")
        vehicle_id = first.vehicle_id
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 1, 0)),
        ])
        self.client.post("/api/camera-road-connections", json={
            "source_camera_id":"CAM01","destination_camera_id":"CAM02","distance_meters":4200,
        })
        self.client.get("/api/plates/GJ01AB1234/suspicion")
        self.client.get("/api/plates/GJ01AB1234/suspicion")
        queue = self.client.get("/api/plate-suspicions").json()
        self.assertEqual(queue["total"], 1)
        suspicion = queue["items"][0]
        self.assertIn(suspicion["classification"], {"suspicious", "high_suspicion"})
        reviewed = self.client.patch(f"/api/plate-suspicions/{suspicion['id']}/review",
                                     json={"status":"acknowledged", "reviewed_by":"tester"}).json()
        self.assertEqual(reviewed["status"], "acknowledged")
        route = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertIn(route["trajectory_hops"][0]["plate_suspicion_status"], {"suspicious", "high_suspicion"})
        self.assertGreater(route["trajectory_hops"][0]["plate_suspicion_score"], 0)
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, vehicle_id)
            self.assertEqual(db.get(PlateEvent, second.id).vehicle_id, vehicle_id)

    def _connect(self, source, destination, distance=1000, direction=None):
        payload = {"source_camera_id": source, "destination_camera_id": destination, "distance_meters": distance}
        if direction:
            payload["direction"] = direction
        return self.client.post("/api/camera-road-connections", json=payload)

    def _route_event(self, camera_id, plate_text="GJ01AB1234"):
        return persist_plate_event(
            dict(camera_id=camera_id, plate_text=plate_text, confidence=.95, status="ok"),
            window_seconds=0,
        )[0]

    def _route_sequence(self, cameras, seconds_step=300, plate_text="GJ01AB1234"):
        events = [self._route_event(camera_id, plate_text=plate_text) for camera_id in cameras]
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        with SessionLocal() as db:
            for index, event in enumerate(events):
                db.get(PlateEvent, event.id).timestamp = start + dt.timedelta(seconds=index * seconds_step)
            db.commit()
        return events

    def test_route_anomaly_insufficient_history(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03"])
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        self.assertEqual(summary["classification"], "insufficient_history")
        self.assertEqual(summary["total"], 0)

    def test_route_anomaly_normal_historical_route(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM01", "CAM02", "CAM03"])
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        self.assertEqual(summary["classification"], "normal")
        self.assertEqual(summary["route_anomaly_score"], 0.0)
        self.assertEqual(summary["total"], 0)

    def test_route_anomaly_new_but_valid_route_is_unusual(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        self._connect("CAM02", "CAM04")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM01", "CAM02", "CAM04"])
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        self.assertEqual(summary["classification"], "unusual_route")
        self.assertEqual(summary["evidence"]["unseen_transition_count"], 1)
        self.assertAlmostEqual(summary["evidence"]["route_deviation_ratio"], .5)

    def test_route_anomaly_disconnected_and_high_deviation_evidence(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM04", "CAM01", "CAM04"])
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        self.assertEqual(summary["classification"], "high_route_anomaly")
        disconnected = summary["evidence"]["disconnected_transitions"]
        self.assertTrue(any(item["source_camera_id"] == "CAM01" and item["destination_camera_id"] == "CAM04"
                            for item in disconnected))
        self.assertEqual(summary["evidence"]["unseen_transition_count"], 2)

    def test_route_anomaly_direction_conflict_when_available(self):
        self._connect("CAM01", "CAM02", direction="NE")
        self._connect("CAM02", "CAM03", direction="E")
        self._connect("CAM03", "CAM02", direction="W")
        self._connect("CAM02", "CAM01", direction="SW")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM03", "CAM02", "CAM01"])
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        conflicts = summary["evidence"]["direction_conflicts"]
        self.assertTrue(any(item["source_camera_id"] == "CAM02" and item["destination_camera_id"] == "CAM01"
                            for item in conflicts))

    def test_route_anomaly_includes_impossible_and_plate_suspicion_context(self):
        self._connect("CAM01", "CAM02", distance=4200)
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM01", "CAM02"], seconds_step=300)
        with SessionLocal() as db:
            db.get(PlateEvent, events[-2].id).timestamp = dt.datetime(2026, 9, 13, 11, 0, 0)
            db.get(PlateEvent, events[-1].id).timestamp = dt.datetime(2026, 9, 13, 11, 1, 0)
            db.commit()
        self.client.get("/api/trajectory/GJ01AB1234")
        self.client.get("/api/plates/GJ01AB1234/suspicion")
        summary = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies").json()
        self.assertTrue(summary["evidence"]["impossible_travel_context"])
        self.assertTrue(summary["evidence"]["plate_suspicion_context"])

    def test_route_anomaly_invalid_ocr_no_fake_route_identity(self):
        persist_plate_event(dict(camera_id="CAM01", plate_text="D", confidence=.99, status="ok"), window_seconds=0)
        persist_plate_event(dict(camera_id="CAM02", plate_text="D", confidence=.99, status="ok"), window_seconds=0)
        self.assertEqual(self.client.get("/api/route-anomalies").json()["total"], 0)
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 0)

    def test_route_anomaly_duplicate_review_trajectory_and_identity_safety(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM04", "CAM01", "CAM04"])
        vehicle_id = events[0].vehicle_id
        self.client.get(f"/api/vehicles/{vehicle_id}/route-anomalies")
        self.client.get(f"/api/vehicles/{vehicle_id}/route-anomalies")
        queue = self.client.get("/api/route-anomalies").json()
        self.assertEqual(queue["total"], 1)
        route_item = queue["items"][0]
        reviewed = self.client.patch(f"/api/route-anomalies/{route_item['id']}/review",
                                     json={"status":"dismissed", "reviewed_by":"tester"}).json()
        self.assertEqual(reviewed["status"], "dismissed")
        trajectory = self.client.get("/api/trajectory/GJ01AB1234").json()
        self.assertEqual(trajectory["route_anomaly"]["classification"], "high_route_anomaly")
        self.assertEqual(trajectory["trajectory_hops"][-1]["route_anomaly_status"], "high_route_anomaly")
        with SessionLocal() as db:
            self.assertTrue(all(db.get(PlateEvent, event.id).vehicle_id == vehicle_id for event in events))

    def test_vehicle_investigation_lookup_summary_history_and_speed(self):
        first = self._valid_event("CAM01", vehicle_color="white", vehicle_type="car",
                                  appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                  appearance_model="test", appearance_embedding_version="test",
                                  appearance_quality=9.0)
        second = self._valid_event("CAM02", vehicle_color="white", vehicle_type="car",
                                   appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                   appearance_model="test", appearance_embedding_version="test",
                                   appearance_quality=9.0)
        self._set_timestamps([
            (second, dt.datetime(2026, 9, 13, 10, 6, 18)),
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
        ])
        self._connect("CAM01", "CAM02", distance=4200)

        by_id = self.client.get(f"/api/vehicles/{first.vehicle_id}/investigation")
        self.assertEqual(by_id.status_code, 200, by_id.text)
        data = by_id.json()
        self.assertEqual(data["summary"]["global_vehicle_id"], first.vehicle_id)
        self.assertEqual(data["summary"]["primary_plate_text"], "GJ01AB1234")
        self.assertEqual(data["summary"]["observation_count"], 2)
        self.assertEqual(data["summary"]["camera_count"], 2)
        self.assertEqual(data["summary"]["trajectory_segment_count"], 1)
        self.assertEqual(data["summary"]["status_summary"], "No concerns")
        self.assertEqual([item["event_id"] for item in data["history"]["observations"]], [first.id, second.id])
        self.assertAlmostEqual(data["trajectory"]["trajectory_hops"][0]["estimated_speed_kmh"], 40.0, places=1)
        self.assertEqual(data["appearance"]["available_observation_count"], 2)
        self.assertGreaterEqual(data["appearance"]["comparisons"][0]["appearance_similarity"], .99)
        self.assertEqual(data["explanations"], [])

        by_plate = self.client.get("/api/plates/GJ01AB1234/investigation").json()
        self.assertEqual(by_plate["summary"]["global_vehicle_id"], first.vehicle_id)
        self.assertIn("review_actions", by_plate)

    def test_vehicle_investigation_includes_impossible_plate_suspicion_and_timeline(self):
        self._connect("CAM01", "CAM02", distance=4200)
        first = self._valid_event("CAM01", appearance_embedding=self._test_embedding([1, 0, 0, 0]),
                                  appearance_model="test", appearance_embedding_version="test",
                                  vehicle_type="car")
        second = self._valid_event("CAM02", appearance_embedding=self._test_embedding([0, 1, 0, 0]),
                                   appearance_model="test", appearance_embedding_version="test",
                                   vehicle_type="motorcycle")
        self._set_timestamps([
            (first, dt.datetime(2026, 9, 13, 10, 0, 0)),
            (second, dt.datetime(2026, 9, 13, 10, 1, 0)),
        ])
        self.client.get("/api/trajectory/GJ01AB1234")
        self.client.get("/api/plates/GJ01AB1234/suspicion")

        data = self.client.get(f"/api/vehicles/{first.vehicle_id}/investigation").json()
        evidence_types = {item["type"] for item in data["explanations"]}
        self.assertIn("impossible_travel", evidence_types)
        self.assertIn("plate_suspicion", evidence_types)
        self.assertIn("appearance_conflict", evidence_types)
        self.assertIn(data["summary"]["plate_suspicion"], {"suspicious", "high_suspicion"})
        self.assertEqual(data["anomalies"]["total"], 1)
        self.assertGreaterEqual(data["plate_suspicions"]["total"], 1)
        self.assertTrue(any(item["type"] == "plate_suspicion" for item in data["timeline"]))
        self.assertFalse(data["summary"]["confirmed_cloned_plate"])
        self.assertFalse(data["summary"]["confirmed_criminal_activity"])

    def test_vehicle_investigation_includes_route_anomaly_context(self):
        self._connect("CAM01", "CAM02")
        self._connect("CAM02", "CAM03")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03", "CAM04", "CAM01", "CAM04"])
        self.client.get(f"/api/vehicles/{events[0].vehicle_id}/route-anomalies")

        data = self.client.get(f"/api/vehicles/{events[0].vehicle_id}/investigation").json()
        self.assertEqual(data["route_anomalies"]["classification"], "high_route_anomaly")
        self.assertEqual(data["summary"]["route_anomaly"], "high_route_anomaly")
        self.assertIn("route_anomaly", {item["type"] for item in data["explanations"]})
        self.assertTrue(any(item["type"] == "route_anomaly" for item in data["timeline"]))
        self.assertEqual(data["trajectory"]["route_anomaly"]["classification"], "high_route_anomaly")

    def test_vehicle_investigation_unknown_and_invalid_ocr_safety(self):
        self.assertEqual(self.client.get("/api/vehicles/VEH-NOTFOUND/investigation").status_code, 404)
        self.assertEqual(self.client.get("/api/plates/GJ01AB1234/investigation").status_code, 404)
        persist_plate_event(dict(camera_id="CAM01", plate_text="D", confidence=.99, status="ok"), window_seconds=0)
        persist_plate_event(dict(camera_id="CAM02", plate_text="112", confidence=.99, status="ok"), window_seconds=0)
        self.assertEqual(self.client.get("/api/plates/D/investigation").status_code, 404)
        with SessionLocal() as db:
            self.assertEqual(db.query(Vehicle).count(), 0)

    def test_vehicle_investigation_existing_apis_remain_compatible(self):
        self._connect("CAM01", "CAM02", distance=4200)
        first, second = self._fixed_pair(378)
        vehicle_id = first.vehicle_id
        investigation = self.client.get(f"/api/vehicles/{vehicle_id}/investigation").json()
        self.assertEqual(investigation["summary"]["global_vehicle_id"], vehicle_id)
        self.assertEqual(self.client.get(f"/api/vehicles/{vehicle_id}/history").status_code, 200)
        self.assertEqual(self.client.get("/api/trajectory/GJ01AB1234").status_code, 200)
        self.assertEqual(self.client.get(f"/api/vehicles/{vehicle_id}/anomalies").status_code, 200)
        self.assertEqual(self.client.get("/api/plates/GJ01AB1234/suspicion").status_code, 200)
        self.assertEqual(self.client.get(f"/api/vehicles/{vehicle_id}/route-anomalies").status_code, 200)
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, vehicle_id)
            self.assertEqual(db.get(PlateEvent, second.id).vehicle_id, vehicle_id)

    def _traffic_event(self, camera_id, plate_text, timestamp, **extra):
        values = dict(camera_id=camera_id, plate_text=plate_text, confidence=.95, status="ok")
        values.update(extra)
        event = persist_plate_event(values, window_seconds=0)[0]
        with SessionLocal() as db:
            db.get(PlateEvent, event.id).timestamp = timestamp
            db.commit()
        return event

    def test_traffic_summary_camera_counts_and_invalid_ocr_safety(self):
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        first = self._traffic_event("CAM01", "GJ01AB1234", start)
        self._traffic_event("CAM01", "GJ01AB5678", start + dt.timedelta(minutes=5))
        persist_plate_event(dict(camera_id="CAM01", plate_text="D", confidence=.99, status="ok"), window_seconds=0)

        summary = self.client.get("/api/traffic/summary").json()
        self.assertEqual(summary["total_observations"], 3)
        self.assertEqual(summary["unique_vehicle_count"], 2)
        self.assertEqual(summary["busiest_camera"]["camera_id"], "CAM01")
        cameras = self.client.get("/api/traffic/cameras", params={"camera_id":"CAM01"}).json()["items"]
        self.assertEqual(cameras[0]["observation_count"], 3)
        self.assertEqual(cameras[0]["unique_vehicle_count"], 2)
        self.assertEqual(cameras[0]["missing_identity_observation_count"], 1)
        self.assertEqual(self.client.get("/api/traffic/cameras", params={"camera_id":"CAMXX"}).status_code, 404)
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, first.vehicle_id)

    def test_traffic_timeseries_hourly_daily_and_filtering(self):
        base = dt.datetime(2026, 9, 13, 10, 15, 0)
        self._traffic_event("CAM01", "GJ01AB1234", base)
        self._traffic_event("CAM02", "GJ01AB5678", base + dt.timedelta(hours=1))
        self._traffic_event("CAM02", "GJ01AB9999", base + dt.timedelta(days=1))
        hourly = self.client.get("/api/traffic/timeseries", params={"bucket":"hour"}).json()["items"]
        self.assertTrue(any(item["time_bucket"].startswith("2026-09-13T10:00:00") and item["camera_id"] == "CAM01"
                            for item in hourly))
        daily = self.client.get("/api/traffic/timeseries", params={"bucket":"day"}).json()["items"]
        self.assertEqual(len({item["time_bucket"] for item in daily}), 2)
        filtered = self.client.get("/api/traffic/summary", params={
            "start": base.isoformat(),
            "end": (base + dt.timedelta(hours=2)).isoformat(),
        }).json()
        self.assertEqual(filtered["total_observations"], 2)
        self.assertEqual(self.client.get("/api/traffic/timeseries", params={"bucket":"week"}).status_code, 422)

    def test_traffic_flow_od_matrix_and_road_speed_reuse(self):
        self._connect("CAM01", "CAM02", distance=4200, direction="NE")
        self._connect("CAM02", "CAM03", distance=3800, direction="E")
        events = self._route_sequence(["CAM01", "CAM02", "CAM03"], seconds_step=378)
        with SessionLocal() as db:
            db.get(PlateEvent, events[2].id).timestamp = dt.datetime(2026, 9, 13, 10, 12, 40)
            db.commit()

        flow = self.client.get("/api/traffic/flow").json()["items"]
        self.assertTrue(any(item["origin"] == "CAM01" and item["destination"] == "CAM02"
                            and item["vehicle_count"] == 1 for item in flow))
        od = self.client.get("/api/traffic/od-matrix", params={"bucket":"hour"}).json()["items"]
        self.assertTrue(any(item["origin"] == "CAM02" and item["destination"] == "CAM03"
                            and item["vehicle_count"] == 1 for item in od))
        roads = self.client.get("/api/traffic/roads").json()["items"]
        cam01_cam02 = next(item for item in roads if item["source_camera_id"] == "CAM01"
                           and item["destination_camera_id"] == "CAM02")
        self.assertAlmostEqual(cam01_cam02["average_estimated_speed_kmh"], 40.0, places=1)
        self.assertEqual(cam01_cam02["valid_speed_segment_count"], 1)
        self.assertEqual(cam01_cam02["utilization"]["metric_note"],
                         "Relative utilization/activity; no physical road capacity is configured.")

    def test_traffic_density_congestion_heatmap_and_lane_status(self):
        with patch.dict(os.environ, {
            "ANPR_TRAFFIC_LOW_THRESHOLD": "1",
            "ANPR_TRAFFIC_MEDIUM_THRESHOLD": "2",
            "ANPR_CONGESTION_WATCH_SPEED_KMH": "50",
            "ANPR_CONGESTION_CONGESTED_SPEED_KMH": "10",
            "ANPR_CONGESTION_WATCH_VOLUME": "1",
            "ANPR_CONGESTION_CONGESTED_VOLUME": "1",
        }):
            self._connect("CAM01", "CAM02", distance=1000)
            first = self._traffic_event("CAM01", "GJ01AB1234", dt.datetime(2026, 9, 13, 10, 0, 0))
            second = self._traffic_event("CAM02", "GJ01AB1234", dt.datetime(2026, 9, 13, 10, 5, 0))
            density = self.client.get("/api/traffic/density", params={"camera_id":"CAM01"}).json()["items"][0]
            self.assertEqual(density["density_level"], "MEDIUM")
            congestion = self.client.get("/api/traffic/congestion").json()["items"]
            road = next(item for item in congestion if item["source_camera_id"] == "CAM01")
            self.assertEqual(road["congestion_status"], "WATCH")
            heatmap = self.client.get("/api/traffic/heatmap").json()["points"]
            self.assertTrue(any(item["camera_id"] == "CAM01" and item["traffic_count"] == 1 for item in heatmap))
            lanes = self.client.get("/api/traffic/lanes").json()
            self.assertEqual(lanes["status"], "NO_DATA")
            self.assertFalse(lanes["available"])
            with SessionLocal() as db:
                self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, db.get(PlateEvent, second.id).vehicle_id)

    def test_traffic_dwell_missing_timestamps_and_empty_dataset(self):
        empty = self.client.get("/api/traffic/dashboard").json()
        self.assertEqual(empty["summary"]["total_observations"], 0)
        start = dt.datetime(2026, 9, 13, 10, 0, 0)
        first = self._traffic_event("CAM01", "GJ01AB1234", start)
        second = self._traffic_event("CAM01", "GJ01AB1234", start + dt.timedelta(minutes=7))
        missing = self._traffic_event("CAM02", "GJ01AB5678", start + dt.timedelta(minutes=9))
        with SessionLocal() as db:
            db.get(PlateEvent, missing.id).timestamp = None
            db.commit()
        dwell = self.client.get("/api/traffic/dwell", params={"bucket":"hour"}).json()["items"]
        cam01 = next(item for item in dwell if item["camera_id"] == "CAM01")
        self.assertEqual(cam01["dwell_status"], "estimated")
        self.assertEqual(cam01["average_dwell_seconds"], 420)
        timeseries = self.client.get("/api/traffic/timeseries").json()["items"]
        self.assertFalse(any(item["camera_id"] == "CAM02" for item in timeseries))
        with SessionLocal() as db:
            self.assertEqual(db.get(PlateEvent, first.id).vehicle_id, db.get(PlateEvent, second.id).vehicle_id)

    def test_traffic_existing_apis_still_work(self):
        self._connect("CAM01", "CAM02", distance=4200)
        first, _second = self._fixed_pair(378)
        self.assertEqual(self.client.get("/api/traffic/dashboard").status_code, 200)
        self.assertEqual(self.client.get(f"/api/vehicles/{first.vehicle_id}/history").status_code, 200)
        self.assertEqual(self.client.get("/api/trajectory/GJ01AB1234").status_code, 200)
        self.assertEqual(self.client.get("/api/vehicles").status_code, 200)

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

    def test_matching_unreadable_plate_similar_appearance_is_suppressed(self):
        first = persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB1234",
                                         confidence=.95, status="ok", **appearance_values()))[0]
        unreadable = persist_plate_event(dict(camera_id="CAM03", plate_text=None,
                                             confidence=0, status="PENDING_REVIEW", **appearance_values()))[0]
        self.assertIsNone(unreadable.vehicle_id)
        with SessionLocal() as db:
            self.assertEqual(db.query(VehicleMatchCandidate).count(), 0)
            self.assertIsNotNone(db.get(PlateEvent, first.id).vehicle_id)

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
        self.assertEqual(comparison["state"], "LOW_CONFIDENCE")

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
        second = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB5678", confidence=.95,
                                          status="ok", **appearance_values()))[0]
        pending = self.client.get("/api/matches/review").json()["items"]
        self.assertEqual(len(pending), 1)
        accepted = self.client.post(f"/api/matches/{pending[0]['match_id']}/review",
                                    json={"action":"accept"})
        self.assertEqual(accepted.status_code, 409)
        with SessionLocal() as db:
            self.assertNotEqual(db.get(PlateEvent, second.id).vehicle_id, first.vehicle_id)

        third = persist_plate_event(dict(camera_id="CAM03", plate_text="GJ01AB9999", confidence=.95,
                                         status="ok", **appearance_values()))[0]
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

    def test_auto_frame_persists_reviewable_partial_ocr(self):
        junk = dict(text="D",raw_text="D",confidence=.79,status="PENDING_REVIEW",
                    needs_review=True, reviewable=True, valid_format=False,
                    bbox=dict(x=20,y=40,width=120,height=40),detection_id=1)
        with patch("app.main.process_image", return_value=("D",.79,"PENDING_REVIEW",json.dumps([junk]))):
            response = self.client.post("/api/process-frame",data={"camera_id":"CAM03"},files={"frame":("image.jpg",b"test","image/jpeg")})
        self.assertEqual(response.status_code,200,response.text)
        body = response.json()
        self.assertIsNotNone(body["event_id"])
        self.assertEqual(body["camera_id"],"CAM03")
        self.assertEqual(body["plate_text"], "D")
        self.assertEqual(body["raw_text"], "D")
        self.assertEqual(body["confidence"], .79)
        self.assertEqual(body["status"], "PENDING_REVIEW")
        self.assertTrue(body["needs_review"])
        with SessionLocal() as db:
            self.assertEqual(db.query(PlateEvent).count(),1)

    def test_auto_frame_ignores_zero_confidence_unreadable(self):
        unreadable = dict(text=None, raw_text="", confidence=0.0, status="PENDING_REVIEW",
                          needs_review=True, reviewable=False, valid_format=False,
                          violations=["unreadable_crop"], detection_id=1)
        with patch("app.main.process_image", return_value=(None,0.0,"PENDING_REVIEW",json.dumps([unreadable]))):
            response = self.client.post("/api/process-frame",data={"camera_id":"CAM03"},files={"frame":("image.jpg",b"test","image/jpeg")})
        self.assertEqual(response.status_code,200,response.text)
        body = response.json()
        self.assertIsNone(body["event_id"])
        self.assertIsNone(body["plate_text"])
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

    def test_phase11_websocket_rejects_unauthenticated_client(self):
        context, anonymous = self.unauthenticated_client()
        try:
            with self.assertRaises(Exception):
                with anonymous.websocket_connect("/ws/events"):
                    pass
        finally:
            context.__exit__(None, None, None)

    def test_phase11_edge_auth_required_blocks_anonymous_and_accepts_edge_token(self):
        payload = {
            "observation_id": "edge-sec-1",
            "camera_id": "CAM01",
            "edge_device_id": "EDGE-SEC-01",
            "capture_timestamp": dt.datetime.utcnow().isoformat(),
            "plate_text": "GJ01AB1234",
            "ocr_confidence": 0.95,
        }
        with patch.dict(os.environ, {"ANPR_EDGE_AUTH_REQUIRED": "true", "ANPR_EDGE_TOKEN": "edge-secret"}, clear=False):
            denied = self.client.post("/api/edge/observations", json=payload)
            self.assertEqual(denied.status_code, 401)
            accepted = self.client.post("/api/edge/observations", json=payload,
                                        headers={"Authorization": "Bearer edge-secret"})
        self.assertEqual(accepted.status_code, 200, accepted.text)
        with SessionLocal() as db:
            device = db.query(EdgeDevice).filter(EdgeDevice.edge_device_id == "EDGE-SEC-01").one()
            self.assertIsNotNone(device.credential_hash)
            self.assertNotEqual(device.credential_hash, "edge-secret")

    def test_phase11_edge_batch_limit_and_malformed_input(self):
        payload = {"observations": [
            {
                "observation_id": f"edge-batch-{index}",
                "camera_id": "CAM01",
                "edge_device_id": "EDGE-BATCH",
                "capture_timestamp": dt.datetime.utcnow().isoformat(),
            }
            for index in range(3)
        ]}
        with patch.dict(os.environ, {"ANPR_MAX_EDGE_BATCH_SIZE": "2"}, clear=False):
            self.assertEqual(self.client.post("/api/edge/observations/batch", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/api/edge/observations", json={"camera_id": "CAM01"}).status_code, 422)

    def test_phase11_security_events_preserve_audit_chain(self):
        context, anonymous = self.unauthenticated_client()
        try:
            self.assertEqual(anonymous.get("/api/vehicles").status_code, 401)
        finally:
            context.__exit__(None, None, None)
        with SessionLocal() as db:
            self.assertGreaterEqual(db.query(AuditLog).count(), 1)
            self.assertTrue(verify_audit_chain(db)["valid"])

    def add_pcr(self, pcr_id="PCR01", lat=23.0325, lng=72.5245, status="AVAILABLE", **extra):
        body = dict(pcr_id=pcr_id, name=pcr_id, vehicle_identifier=f"DEMO-{pcr_id}",
                    latitude=lat, longitude=lng, status=status,
                    last_seen_at=dt.datetime.utcnow().isoformat(), contact_channel="Demo PCR GPS")
        body.update(extra)
        response = self.client.post("/api/pcr", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_pcr_creation_retrieval_and_location_update(self):
        created = self.add_pcr("PCR10")
        self.assertEqual(created["pcr_id"], "PCR10")
        self.assertEqual(created["location_status"], "LIVE")
        listed = self.client.get("/api/pcr").json()
        self.assertEqual(listed["items"][0]["pcr_id"], "PCR10")
        one = self.client.get("/api/pcr/PCR10").json()
        self.assertEqual(one["vehicle_identifier"], "DEMO-PCR10")
        stamp = dt.datetime.utcnow().replace(microsecond=0)
        updated = self.client.post("/api/pcr/PCR10/location", json={
            "latitude": 23.04,
            "longitude": 72.53,
            "timestamp": stamp.isoformat(),
        })
        self.assertEqual(updated.status_code, 200, updated.text)
        data = updated.json()
        self.assertEqual(data["latitude"], 23.04)
        self.assertEqual(data["longitude"], 72.53)

    def test_pcr_coordinate_and_timestamp_validation(self):
        self.assertEqual(self.client.post("/api/pcr", json={
            "pcr_id":"PCRBAD", "latitude":91, "longitude":72.5,
        }).status_code, 422)
        self.assertEqual(self.client.post("/api/pcr", json={
            "pcr_id":"PCRBAD", "latitude":23.0, "longitude":181,
        }).status_code, 422)
        self.add_pcr("PCR11")
        self.assertEqual(self.client.post("/api/pcr/PCR11/location", json={
            "latitude": 23.0, "longitude": 181, "timestamp": dt.datetime.utcnow().isoformat(),
        }).status_code, 422)
        self.assertEqual(self.client.post("/api/pcr/PCR11/location", json={
            "latitude": 23.0, "longitude": 72.0,
        }).status_code, 422)
        self.assertEqual(self.client.get("/api/pcr/PCRXX").status_code, 404)

    def test_nearest_pcr_filters_busy_stale_and_unavailable(self):
        self.client.patch("/api/cameras/CAM02", json={"lat":23.0325,"lng":72.5145})
        now = dt.datetime.utcnow()
        with patch.dict(os.environ, {"ANPR_PCR_LOCATION_MAX_AGE_SECONDS":"60"}):
            self.add_pcr("PCR01", 23.0500, 72.5400, "AVAILABLE", last_seen_at=now.isoformat())
            self.add_pcr("PCR02", 23.0330, 72.5240, "AVAILABLE", last_seen_at=now.isoformat())
            self.add_pcr("PCR03", 23.0328, 72.5150, "BUSY", last_seen_at=now.isoformat())
            self.add_pcr("PCR04", 23.0331, 72.5160, "AVAILABLE",
                         last_seen_at=(now - dt.timedelta(minutes=10)).isoformat())
            data = self.client.get("/api/pcr/nearest", params={"camera_id":"CAM02"}).json()
        self.assertEqual(data["recommended_pcr"]["pcr_id"], "PCR02")
        self.assertEqual(data["recommended_pcr"]["location_status"], "LIVE")
        self.assertTrue(any(item["pcr_id"] == "PCR03" and not item["eligible"] for item in data["candidates"]))
        self.assertTrue(any(item["pcr_id"] == "PCR04" and item["location_status"] == "STALE" for item in data["candidates"]))

    def test_nearest_pcr_no_available_and_no_camera_coordinates(self):
        self.client.patch("/api/cameras/CAM02", json={"lat":23.0325,"lng":72.5145})
        self.add_pcr("PCR01", 23.0330, 72.5240, "BUSY")
        self.add_pcr("PCR02", 23.0340, 72.5250, "OFFLINE")
        data = self.client.get("/api/pcr/nearest", params={"camera_id":"CAM02"}).json()
        self.assertIsNone(data["recommended_pcr"])
        self.assertEqual(data["reason"], "NO_AVAILABLE_PCR")
        self.client.post("/api/cameras", json={"camera_id":"CAMNOLOC"})
        missing = self.client.get("/api/pcr/nearest", params={"camera_id":"CAMNOLOC"}).json()
        self.assertIsNone(missing["recommended_pcr"])
        self.assertEqual(missing["reason"], "NO_CAMERA_COORDINATES")

    def test_hotlist_match_creates_incident_with_vehicle_and_trajectory_links(self):
        self.add_pcr("PCR02", 23.0330, 72.5240, "AVAILABLE")
        self.add_hotlist()
        event,_ = persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB1234",
                                           confidence=.95, status="ok"), window_seconds=0)
        incidents = self.client.get("/api/incidents").json()
        self.assertEqual(incidents["total"], 1)
        incident = incidents["items"][0]
        self.assertEqual(incident["hotlist_alert_id"], self.client.get("/api/hotlist-alerts").json()["items"][0]["id"])
        self.assertEqual(incident["plate_text"], "GJ01AB1234")
        self.assertEqual(incident["global_vehicle_id"], event.vehicle_id)
        self.assertEqual(incident["recommended_pcr_id"], "PCR02")
        self.assertEqual(incident["pcr_location_status"], "LIVE")
        self.assertEqual(incident["recommendation_status"], "DISPATCH_RECOMMENDED")
        self.assertIn(f"/api/vehicles/{event.vehicle_id}/trajectory", incident["trajectory_reference"])
        self.assertIn(f"/api/vehicles/{event.vehicle_id}/investigation", incident["investigation_reference"])
        investigation = self.client.get(incident["investigation_reference"]).json()
        self.assertEqual(investigation["summary"]["global_vehicle_id"], event.vehicle_id)

    def test_duplicate_incident_prevention_and_invalid_ocr_safety(self):
        self.add_pcr("PCR02", 23.0330, 72.5240, "AVAILABLE")
        self.add_hotlist()
        values = dict(camera_id="CAM02", plate_text="GJ01AB1234", confidence=.95, status="ok")
        first,_ = persist_plate_event(values)
        persist_plate_event(values)
        with SessionLocal() as db:
            alert = db.query(HotlistAlert).filter_by(event_id=first.id).one()
            self.assertEqual(db.query(EnforcementIncident).filter_by(hotlist_alert_id=alert.id).count(), 1)
        self.add_hotlist(plate_text="GJ01AB5678")
        persist_plate_event(dict(camera_id="CAM01", plate_text="GJ01AB5678",
                                 confidence=.65, status="PENDING_REVIEW"), window_seconds=0)
        alerts = self.client.get("/api/hotlist-alerts").json()
        self.assertEqual(alerts["total"], 2)
        self.assertEqual(self.client.get("/api/incidents").json()["total"], 1)
        persist_plate_event(dict(camera_id="CAM03", plate_text="D", confidence=.99, status="ok"), window_seconds=0)
        self.assertEqual(self.client.get("/api/incidents").json()["total"], 1)

    def test_incident_status_update_and_hotlist_acknowledgement_link(self):
        self.add_pcr("PCR02", 23.0330, 72.5240, "AVAILABLE")
        self.add_hotlist()
        persist_plate_event(dict(camera_id="CAM02", plate_text="GJ01AB1234",
                                 confidence=.95, status="ok"), window_seconds=0)
        incident = self.client.get("/api/incidents").json()["items"][0]
        updated = self.client.post(f"/api/incidents/{incident['incident_id']}/status",
                                   json={"status":"ACKNOWLEDGED", "updated_by":"tester"})
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["status"], "ACKNOWLEDGED")
        self.assertEqual(self.client.post(f"/api/incidents/{incident['incident_id']}/status",
                                          json={"status":"DISPATCHED"}).status_code, 422)
        alert_id = incident["hotlist_alert_id"]
        with SessionLocal() as db:
            db.get(EnforcementIncident, incident["incident_id"]).status = "DISPATCH_RECOMMENDED"
            db.commit()
        ack = self.client.post(f"/api/hotlist-alerts/{alert_id}/acknowledge").json()
        self.assertIsNotNone(ack["acknowledged_at"])
        self.assertEqual(self.client.get(f"/api/incidents/{incident['incident_id']}").json()["status"], "ACKNOWLEDGED")

    def test_law_enforcement_websocket_event(self):
        from app.main import event_writer
        self.add_pcr("PCR02", 23.0330, 72.5240, "AVAILABLE")
        self.add_hotlist()
        with self.client.websocket_connect("/ws/events") as socket:
            socket.receive_json()
            event_writer.enqueue_from_thread(dict(camera_id="CAM02", plate_text="GJ01AB1234",
                                                  confidence=.95, status="ok"))
            messages = [socket.receive_json(), socket.receive_json()]
        types = {message["type"] for message in messages}
        self.assertIn("hotlist_alert", types)
        self.assertIn("law_enforcement_alert", types)
        law = next(message for message in messages if message["type"] == "law_enforcement_alert")
        self.assertEqual(law["recommended_pcr_id"], "PCR02")
        self.assertEqual(law["incident"]["vehicle_id"], law["vehicle_id"])

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
