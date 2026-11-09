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


def _read_variant(image, psm):
    data = pytesseract.image_to_data(
        image,
        lang="rus+eng",
        config=f"--oem 1 --psm {psm}",
        output_type=Output.DICT,
        timeout=5
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

        # Keep weak-confidence tokens too: the small x / х before a stack
        # count is often assigned a low confidence by Tesseract.
        if confidence < 0:
            continue

        words.append({
            "text": text,
            "confidence": confidence,
            "x": round(data["left"][i] / OCR_SCALE),
            "y": round(data["top"][i] / OCR_SCALE),
            "width": round(data["width"][i] / OCR_SCALE),
            "height": round(data["height"][i] / OCR_SCALE),
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
    # Favor variants that actually preserve item names and xN stack counts.
    return len(words) * 1.5 + confidence / 20 + cyrillic_count * 0.15 + (20 if has_stack else 0)


def recognize_text(image):
    if image is None or image.size == 0:
        return []

    scaled = cv2.resize(
        image, None, fx=OCR_SCALE, fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    inverted = cv2.bitwise_not(binary)

    # Sidebar rows are sparse, so PSM 11 complements the regular block mode.
    # Try both polarities because the game panel may use light or dark text.
    variants = [
        (binary, 6),
        (binary, 11),
        (inverted, 6),
        (inverted, 11)
    ]
    best_words = []
    best_score = -1
    for processed, psm in variants:
        try:
            words = _read_variant(processed, psm)
        except Exception:
            continue
        score = _quality_score(words)
        if score > best_score:
            best_words = words
            best_score = score

    return best_words
