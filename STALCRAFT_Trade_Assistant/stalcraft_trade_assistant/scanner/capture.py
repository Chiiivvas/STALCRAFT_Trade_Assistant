import cv2
import mss
import numpy as np


def capture_region(region):
    """Capture screen region [x, y, width, height] as a BGR image."""
    x, y, width, height = map(int, region)
    if width <= 0 or height <= 0:
        raise ValueError("Некорректный размер области сканирования")
    with mss.mss() as sct:
        shot = np.array(sct.grab({
            "left": x, "top": y, "width": width, "height": height
        }))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
