"""Image and label loading helpers."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import DEFAULT_IMAGE_SIZE, IMAGE_EXTENSIONS


def read_image(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from paths that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def list_images(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_image_label(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for image_path in list_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.png"
        if not label_path.exists():
            label_path = label_dir / f"{image_path.stem}.jpg"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def resize_label(label_gray: np.ndarray, size=DEFAULT_IMAGE_SIZE) -> np.ndarray:
    label = cv2.resize(label_gray, size, interpolation=cv2.INTER_NEAREST)
    return (label > 127).astype(np.uint8)


def load_pair(image_path: Path, label_path: Path, size=DEFAULT_IMAGE_SIZE):
    image = read_image(image_path)
    label = resize_label(read_image(label_path, cv2.IMREAD_GRAYSCALE), size)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image, label
