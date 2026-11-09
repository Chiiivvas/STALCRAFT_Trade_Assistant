import cv2
import pytesseract
from pytesseract import Output
from pathlib import Path

# Common Windows install location. Tesseract does not need to be in PATH.
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if _WINDOWS_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)

# 1.5x keeps small game UI text readable while processing ~44% fewer pixels
# than the previous 2x upscale.
OCR_SCALE = 1.5
OCR_CONFIG = "--oem 1 --psm 6"


def recognize_text(image):
    if image is None or image.size == 0:
        return []

    scaled = cv2.resize(
        image, None, fx=OCR_SCALE, fy=OCR_SCALE,
        interpolation=cv2.INTER_LINEAR
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    data = pytesseract.image_to_data(
        processed,
        lang="rus+eng",
        config=OCR_CONFIG,
        output_type=Output.DICT,
        timeout=4
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
        if confidence < 20:
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
