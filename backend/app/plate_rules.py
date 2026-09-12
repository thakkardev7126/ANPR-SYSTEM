"""
India-specific plate post-processing helpers.

The OCR/detection models do the learned work; this module applies the small
deterministic rules that are expected in an ANPR stack: positional character
correction, flexible district parsing, state/district sanity checks, and plate
background colour classification from OpenCV HSV masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

import cv2
import numpy as np


OCR_ACCEPT_CONFIDENCE = 0.80
PENDING_REVIEW_STATUS = "PENDING_REVIEW"
AMBIGUOUS_PLATE_CHARS = {"V", "Y"}

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN",
    "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB",
}

# Broad RTO district-code ranges. Unknown/newer offices are not hard-failed;
# they are marked for review so the operator can confirm them.
STATE_DISTRICT_RANGES = {
    "AP": range(1, 40), "AR": range(1, 25), "AS": range(1, 34),
    "BR": range(1, 58), "CG": range(1, 31), "CH": range(1, 5),
    "DD": range(1, 4), "DL": range(1, 15), "DN": range(1, 3),
    "GA": range(1, 13), "GJ": range(1, 49), "HP": range(1, 97),
    "HR": range(1, 100), "JH": range(1, 25), "JK": range(1, 23),
    "KA": range(1, 71), "KL": range(1, 87), "LA": range(1, 3),
    "LD": range(1, 2), "MH": range(1, 54), "ML": range(1, 12),
    "MN": range(1, 8), "MP": range(1, 71), "MZ": range(1, 9),
    "NL": range(1, 12), "OD": range(1, 36), "PB": range(1, 72),
    "PY": range(1, 6), "RJ": range(1, 53), "SK": range(1, 5),
    "TN": range(1, 100), "TR": range(1, 9), "TS": range(1, 40),
    "UK": range(1, 21), "UP": range(1, 97), "WB": range(1, 100),
}

LETTER_POSITION_CORRECTIONS = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"
}
NUMBER_POSITION_CORRECTIONS = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
    "A": "4", "S": "5", "G": "6", "J": "6", "B": "8",
}


@dataclass
class PlateRuleResult:
    raw_text: str
    normalized_text: str | None
    valid_format: bool
    confidence_hint: float
    state_code: str | None = None
    district_code: str | None = None
    vehicle_category: str | None = None
    color_confidence: float = 0.0
    violations: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


def sanitize_plate_text(raw_text: str | None) -> str:
    text = unicodedata.normalize("NFKC", raw_text or "").upper()
    return re.sub(r"[^A-Z0-9\u2191]", "", text)


def strip_hsrp_noise(raw_text: str | None) -> str:
    text = sanitize_plate_text(raw_text)
    # Only strip a leading logo when the remainder starts with a plate prefix.
    for logo in ("INDIA", "IND", "IN"):
        if text.startswith(logo):
            suffix = text[len(logo):]
            state = "".join(LETTER_POSITION_CORRECTIONS.get(c, c) for c in suffix[:2])
            if state in INDIAN_STATE_CODES or re.match(r"(?:[0-9]{2}BH|[0-9]{1,3}(?:CD|CC|UN)|\u2191)", suffix):
                return suffix
    return "" if text in {"IN", "IND", "INDIA"} else text


def _correct_chars(value: str, mapping: dict[str, str]) -> tuple[str, int]:
    corrected = "".join(mapping.get(c, c) for c in value)
    return corrected, sum(a != b for a, b in zip(value, corrected))


def _result(text, normalized, changes=0, state=None, district=None, violations=None):
    violations = violations or []
    return PlateRuleResult(
        raw_text=text, normalized_text=normalized, valid_format=not violations,
        confidence_hint=min(0.70 if violations else 0.96, max(0.60, 0.96 - changes * 0.04)),
        state_code=state, district_code=district, violations=violations,
        corrections=[f"positional:{text}->{normalized}"] if text != normalized else [],
    )


def _classify_state_plate(text: str) -> PlateRuleResult | None:
    if not 8 <= len(text) <= 11:
        return None
    state, state_changes = _correct_chars(text[:2], LETTER_POSITION_CORRECTIONS)
    if state not in INDIAN_STATE_CODES:
        return None
    candidates = []
    for district_len in (2, 1):
        series_len = len(text) - 2 - district_len - 4
        if not 1 <= series_len <= 3:
            continue
        district_raw = text[2:2 + district_len]
        series_raw = text[2 + district_len:-4]
        serial_raw = text[-4:]
        # Never reinterpret an extra leading/trailing serial digit as a series.
        if series_len == 3 and not series_raw.isalpha():
            continue
        district, dc = _correct_chars(district_raw, NUMBER_POSITION_CORRECTIONS)
        series, sc = _correct_chars(series_raw, LETTER_POSITION_CORRECTIONS)
        serial, nc = _correct_chars(serial_raw, NUMBER_POSITION_CORRECTIONS)
        if not (district.isdigit() and series.isalpha() and serial.isdigit()):
            continue
        violations = []
        value = int(district)
        if value == 0 or (state in STATE_DISTRICT_RANGES and value not in STATE_DISTRICT_RANGES[state]):
            violations.append(f"district_out_of_range:{state}{value:02d}")
        normalized = f"{state}{value:02d}{series}{serial}"
        result = _result(text, normalized, state_changes + dc + sc + nc, state, f"{value:02d}", violations)
        score = (not violations, district_raw.isdigit(), series_raw.isalpha(), district_len == 2, -dc-sc-nc)
        candidates.append((score, result))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _classify_bh_plate(text: str) -> PlateRuleResult | None:
    if len(text) != 10 or text[2:4] != "BH":
        return None
    year, yc = _correct_chars(text[:2], NUMBER_POSITION_CORRECTIONS)
    serial, nc = _correct_chars(text[4:8], NUMBER_POSITION_CORRECTIONS)
    series, sc = _correct_chars(text[8:], LETTER_POSITION_CORRECTIONS)
    normalized = f"{year}BH{serial}{series}"
    if not re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{2}", normalized):
        return None
    return _result(text, normalized, yc + nc + sc, "BH")


def _classify_special_plate(text: str) -> PlateRuleResult | None:
    diplomatic = re.fullmatch(r"([0-9]{1,3})(CDP|CD|CC|UN|IOD|IOC)([0-9]{1,4})([A-Z]?)", text)
    if diplomatic:
        return _result(text, text, state=diplomatic[2])
    # Preserve the broad arrow in either printed position; canonicalize to front.
    military = re.fullmatch(r"(?:\u2191([0-9]{2})([A-Z])|([0-9]{2})([A-Z])\u2191)([0-9]{1,6})([A-Z]{1,3})", text)
    if military:
        normalized = "\u2191" + (military[1] or military[3]) + (military[2] or military[4]) + military[5] + military[6]
        return _result(text, normalized, state="MILITARY")
    return None


def normalize_plate_text(raw_text: str | None, vehicle_category: str | None = None,
                         color_confidence: float = 0.0) -> PlateRuleResult:
    original = sanitize_plate_text(raw_text)
    text = strip_hsrp_noise(raw_text)
    corrections = [f"hsrp:{original}->{text}"] if original != text else []
    if text.startswith("6J"):
        text = "GJ" + text[2:]
        corrections.append("prefix:6J->GJ")
    if text.startswith("GJ6J"):
        text = "GJ06" + text[4:]
        corrections.append("prefix:GJ6J->GJ06")
    result = _classify_bh_plate(text) or _classify_special_plate(text) or _classify_state_plate(text)
    if result is None:
        result = PlateRuleResult(raw_text=original, normalized_text=None, valid_format=False,
                                 confidence_hint=0.0, violations=["empty_ocr" if not text else "format_mismatch"])
    result.raw_text = original
    result.corrections = corrections + result.corrections
    result.vehicle_category = vehicle_category
    result.color_confidence = color_confidence
    return result


def classify_plate_color(crop_bgr) -> tuple[str, float]:
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    masks = {
        "private_white": cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 70, 255])),
        "commercial_yellow": cv2.inRange(hsv, np.array([15, 70, 80]), np.array([40, 255, 255])),
        "ev_green": cv2.inRange(hsv, np.array([35, 45, 45]), np.array([95, 255, 255])),
        "rental_black": cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 65])),
        "testing_red": (
            cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255])) |
            cv2.inRange(hsv, np.array([170, 70, 70]), np.array([179, 255, 255]))
        ),
    }
    total = float(crop_bgr.shape[0] * crop_bgr.shape[1])
    scores = {name: float(cv2.countNonZero(mask)) / total for name, mask in masks.items()}
    category, score = max(scores.items(), key=lambda item: item[1])
    return (category, round(score, 3)) if score >= 0.10 else ("unknown", round(score, 3))


def status_for_plate(confidence: float, rule_result: PlateRuleResult | None = None) -> str:
    if confidence < OCR_ACCEPT_CONFIDENCE:
        return PENDING_REVIEW_STATUS
    if rule_result and not rule_result.valid_format:
        return PENDING_REVIEW_STATUS
    return "ok"
