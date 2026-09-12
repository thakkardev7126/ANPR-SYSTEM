"""Bounded OpenCV preprocessing and quality assessment for plate crops."""
import math
import cv2
import numpy as np


def bbox_frame_edges(image, bbox):
    x, y, width, height = map(float, bbox)
    ih, iw = image.shape[:2]
    return [name for name, hit in (("left", x <= 2), ("top", y <= 2),
                                  ("right", x+width >= iw-2), ("bottom", y+height >= ih-2)) if hit]


def padded_plate_crop(image, bbox, margin_x=0.10, margin_y=0.20):
    x, y, width, height = map(float, bbox)
    ih, iw = image.shape[:2]
    pad_x, pad_y = max(8, width * margin_x), max(6, height * margin_y)
    bounds = (max(0, math.floor(x-pad_x)), max(0, math.floor(y-pad_y)),
              min(iw, math.ceil(x+width+pad_x)), min(ih, math.ceil(y+height+pad_y)))
    left, top, right, bottom = bounds
    return image[top:bottom, left:right].copy(), bbox_frame_edges(image, bbox)


def normalize_size(image, target_height=96, max_width=1600):
    height, width = image.shape[:2]
    scale = min(target_height / max(1, height), max_width / max(1, width))
    return cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def assess_quality(image):
    if image is None or image.size == 0:
        return {"sharpness": 0.0, "score": 0.0, "flags": ["empty_crop"], "unreadable": True}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    dark = float(np.mean(gray < 45))
    bright = float(np.mean(gray > 248))
    halves = [float(np.mean(part)) for part in (gray[:, :max(1, gray.shape[1]//2)],
                                               gray[:, gray.shape[1]//2:])]
    flags = []
    if sharpness < 100:
        flags.append("blur")
    if dark > .55:
        flags.append("low_light")
    if bright > .35:
        flags.append("glare")
    if abs(halves[0]-halves[1]) > 65:
        flags.append("split_shadow")
    if contrast < 25:
        flags.append("low_contrast")
    if min(gray.shape) < 18:
        flags.append("tiny_plate")
    unreadable = contrast < 3 or sharpness < 2 or min(gray.shape) < 8
    score = math.log1p(sharpness) + contrast / 64 - len(flags) * .3
    return {"sharpness": round(sharpness, 2), "contrast": round(contrast, 2),
            "score": round(score, 3), "flags": flags, "unreadable": unreadable}


def illumination_correct(gray):
    size = max(3, min(51, (min(gray.shape)//2) | 1))
    background = cv2.GaussianBlur(gray, (size, size), 0)
    return cv2.divide(gray, np.maximum(background, 1), scale=160)


def wiener_motion(gray, angle, length=7, noise=.04):
    # A short linear PSF gives a bounded hypothesis, not missing-character recovery.
    psf = np.zeros(gray.shape, np.float32)
    cy, cx = gray.shape[0]//2, gray.shape[1]//2
    dx, dy = math.cos(math.radians(angle))*length/2, math.sin(math.radians(angle))*length/2
    cv2.line(psf, (round(cx-dx), round(cy-dy)), (round(cx+dx), round(cy+dy)), 1, 1)
    psf /= max(float(psf.sum()), 1)
    transfer = np.fft.fft2(np.fft.ifftshift(psf))
    restored = np.fft.ifft2(np.fft.fft2(gray) * np.conj(transfer) / (np.abs(transfer)**2 + noise)).real
    return np.clip(restored, 0, 255).astype(np.uint8)


def enhanced_variants(crop):
    quality = assess_quality(crop)
    crop = normalize_size(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    local = illumination_correct(denoised)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(denoised)
    adaptive = cv2.adaptiveThreshold(local, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 9)
    variants = [("original", crop), ("denoised", denoised), ("clahe", clahe),
                ("shadow_normalized", local), ("adaptive", adaptive),
                ("inverted", cv2.bitwise_not(adaptive))]
    if "low_light" in quality["flags"]:
        lut = np.array([255 * (i/255)**.55 for i in range(256)], np.uint8)
        variants.append(("night_gamma", cv2.LUT(denoised, lut)))
    if "glare" in quality["flags"]:
        variants.append(("highlight_compressed", cv2.normalize(
            np.log1p(gray.astype(np.float32)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)))
    if "blur" in quality["flags"] and not quality["unreadable"]:
        variants.extend((f"motion_{angle}", wiener_motion(denoised, angle)) for angle in (0, 45, 90))
    return [(name, cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im)
            for name, im in variants], quality


class PartialPlateHistory:
    """Keep bounded, same-track overlap hypotheses; never confirm inferred text."""
    def __init__(self, ttl=20, max_tracks=256):
        self.ttl = ttl
        self.max_tracks = max_tracks
        self.observations = {}

    def observe(self, track_id, text, now):
        from app.plate_rules import strip_hsrp_noise, normalize_plate_text
        self.observations = {key: value for key, value in self.observations.items()
                             if now-value[0] < self.ttl}
        text = strip_hsrp_noise(text)
        history = self.observations.get(track_id, (now, []))[1]
        hypotheses = set()
        if len(text) >= 4:
            for previous in history:
                for left, right in ((previous, text), (text, previous)):
                    for size in range(min(len(left), len(right)), 3, -1):
                        if left[-size:] == right[:size]:
                            joined = left + right[size:]
                            rule = normalize_plate_text(joined)
                            if rule.valid_format and rule.normalized_text:
                                hypotheses.add(rule.normalized_text)
            history = (history + [text])[-6:]
            self.observations[track_id] = (now, history)
        while len(self.observations) > self.max_tracks:
            self.observations.pop(min(self.observations, key=lambda k: self.observations[k][0]))
        return sorted(hypotheses)
