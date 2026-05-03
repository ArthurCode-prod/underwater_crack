"""Classical image-processing baselines."""

import cv2
import numpy as np


def postprocess(mask: np.ndarray, close_size=3, open_size=2) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)


def classical_masks(underwater_bgr_float: np.ndarray):
    bgr = np.clip(underwater_bgr_float * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)

    _, otsu = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4
    )

    med = float(np.median(blur))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    canny = cv2.Canny(blur, lower, upper)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, blackhat_mask = cv2.threshold(blackhat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "Otsu": postprocess(otsu, close_size=3, open_size=2),
        "Adaptive": postprocess(adaptive, close_size=3, open_size=2),
        "Canny": postprocess((canny > 0).astype(np.uint8), close_size=3, open_size=1),
        "Blackhat": postprocess(blackhat_mask, close_size=3, open_size=2),
    }
