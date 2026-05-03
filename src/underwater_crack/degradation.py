"""Synthetic underwater degradation model used in the paper."""

import cv2
import numpy as np


def degradation_profile(name: str):
    profiles = {
        "low": dict(r=(0.55, 0.70), g=(0.88, 1.00), b=(0.95, 1.00), scatter=(0.08, 0.16), noise=0.015, blur=21),
        "medium": dict(r=(0.35, 0.55), g=(0.80, 0.95), b=(0.90, 1.00), scatter=(0.20, 0.35), noise=0.030, blur=35),
        "high": dict(r=(0.20, 0.38), g=(0.68, 0.86), b=(0.82, 0.96), scatter=(0.34, 0.50), noise=0.045, blur=45),
        "severe": dict(r=(0.12, 0.28), g=(0.55, 0.76), b=(0.75, 0.92), scatter=(0.48, 0.64), noise=0.060, blur=55),
    }
    if name not in profiles:
        raise KeyError(f"Unknown degradation profile: {name}")
    return profiles[name]


def simulate_underwater(image_bgr: np.ndarray, seed: int, profile: str = "medium") -> np.ndarray:
    """Apply channel attenuation, scattering blur, and additive noise."""
    rng = np.random.default_rng(seed)
    p = degradation_profile(profile)
    image = image_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(image)

    r = r * rng.uniform(*p["r"])
    g = g * rng.uniform(*p["g"])
    b = b * rng.uniform(*p["b"])
    shifted = cv2.merge([b, g, r])

    blur_size = p["blur"] if p["blur"] % 2 == 1 else p["blur"] + 1
    blur = cv2.GaussianBlur(shifted, (blur_size, blur_size), 0)
    scatter = rng.uniform(*p["scatter"])
    scattered = shifted * (1.0 - scatter) + blur * scatter
    noise = rng.normal(0.0, p["noise"], scattered.shape).astype(np.float32)
    return np.clip(scattered + noise, 0.0, 1.0)
