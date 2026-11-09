import re
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output

# Common Windows install location. Tesseract does not need to be in PATH.
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if _WINDOWS_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)

OCR_SCALE = 3.0


def _read_variant(image, psm):
    data = pytesseract.image_to_data(
        image,
        lang="rus+eng",
        config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
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
            "x": round(data["left"][i] / OCR_SCALE),
            "y": round(data["top"][i] / OCR_SCALE),
            "width": max(1, round(data["width"][i] / OCR_SCALE)),
            "height": max(1, round(data["height"][i] / OCR_SCALE)),
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


def recognize_text(image):
    if image is None or image.size == 0:
        return []

    scaled = cv2.resize(
        image, None, fx=OCR_SCALE, fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # Keep one soft grayscale pass and create separate contrast/threshold
    # versions. Game fonts are often small, anti-aliased and low-contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
    binary = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    inverted = cv2.bitwise_not(binary)

    # Different segmentation modes recover text that a single threshold
    # misses, especially Cyrillic item names and small stack counters.
    variants = [
        (gray, 6), (gray, 11),
        (clahe, 6), (sharpened, 11),
        (binary, 6), (adaptive, 11),
        (inverted, 11)
    ]
    all_words = []
    for processed, psm in variants:
        try:
            all_words.extend(_read_variant(processed, psm))
        except Exception:
            # A failed preprocessing/mode must not discard detections from
            # the other OCR passes.
            continue

    return _merge_variant_words(all_words)
