import cv2
import pytesseract
from pytesseract import Output


# Use the standard Windows installation path; PATH is not required.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def recognize_text(image):
    if image is None or image.size == 0:
        return []

    enlarged = cv2.resize(
        image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    data = pytesseract.image_to_data(
        processed,
        lang="rus+eng",
        config="--oem 3 --psm 6",
        output_type=Output.DICT
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
            "x": int(data["left"][i] / 2),
            "y": int(data["top"][i] / 2),
            "width": int(data["width"][i] / 2),
            "height": int(data["height"][i] / 2),
            "line": (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i]
            )
        })
    return words
