import re
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output

# Common Windows install location. Tesseract does not need to be in PATH.
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if _WINDOWS_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)

OCR_SCALE = 2.0


def _read_variant(image, psm, scale=OCR_SCALE):
    data = pytesseract.image_to_data(
        image,
        lang="rus+eng",
        config=f"--oem 1 --psm {psm}",
        output_type=Output.DICT,
        timeout=6
    )
    words = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        # Keep low-confidence tokens: small x/х and thin digits are often weak.
        if confidence < 0:
            continue

        words.append({
            "text": text,
            "confidence": confidence,
            "x": round(data["left"][i] / scale),
            "y": round(data["top"][i] / scale),
            "width": max(1, round(data["width"][i] / scale)),
            "height": max(1, round(data["height"][i] / scale)),
            "line": (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i]
            )
        })
    return words


def _quality_score(words):
    if not words:
        return 0
    text = " ".join(word["text"] for word in words)
    confidence = sum(max(0, word["confidence"]) for word in words) / len(words)
    has_stack = bool(re.search(r"(?i)[xх×]\s*\d{1,6}", text))
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
    return len(words) * 1.5 + confidence / 20 + cyrillic_count * 0.15 + (20 if has_stack else 0)


def _intersection_over_union(a, b):
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _merge_variant_words(all_words):
    """Fuse detections at the same screen position without discarding unique rows."""
    merged = []
    for word in sorted(all_words, key=lambda item: item["confidence"], reverse=True):
        duplicate_index = None
        for index, existing in enumerate(merged):
            same_position = (
                abs((word["x"] + word["width"] / 2) -
                    (existing["x"] + existing["width"] / 2)) <=
                max(5, min(word["width"], existing["width"]) * 0.45)
                and abs((word["y"] + word["height"] / 2) -
                        (existing["y"] + existing["height"] / 2)) <=
                max(4, min(word["height"], existing["height"]) * 0.55)
            )
            if same_position or _intersection_over_union(word, existing) >= 0.35:
                duplicate_index = index
                break
        if duplicate_index is None:
            merged.append(word)
        else:
            existing = merged[duplicate_index]
            # Prefer the higher-confidence token, but retain the larger box.
            if word["confidence"] > existing["confidence"]:
                merged[duplicate_index] = {
                    **word,
                    "x": min(word["x"], existing["x"]),
                    "y": min(word["y"], existing["y"]),
                    "width": max(word["x"] + word["width"], existing["x"] + existing["width"]) -
                             min(word["x"], existing["x"]),
                    "height": max(word["y"] + word["height"], existing["y"] + existing["height"]) -
                              min(word["y"], existing["y"])
                }
    return sorted(merged, key=lambda item: (item["y"], item["x"]))


def recognize_text(image, fast=False):
    if image is None or image.size == 0:
        return []

    # Fast mode is used for the live trade panel: one OCR pass only.
    # Smaller scaling and no blur/threshold variants cut the work substantially.
    scale = 1.5 if fast else OCR_SCALE
    scaled = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_LINEAR if fast else cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    if fast:
        try:
            return _read_variant(gray, 6, scale=scale)
        except Exception:
            return []

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    inverted = cv2.bitwise_not(binary)
    all_words = []
    for processed, psm in [
        (gray, 6), (gray, 11),
        (binary, 6), (binary, 11),
        (inverted, 11)
    ]:
        try:
            all_words.extend(_read_variant(processed, psm, scale=scale))
        except Exception:
            continue
    return _merge_variant_words(all_words)
