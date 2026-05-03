"""Shared configuration for reproducible experiments."""

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

METHOD_ORDER = [
    "Otsu",
    "Adaptive",
    "Canny",
    "Blackhat",
    "UNet_no_TAFE",
    "TAFE_channel",
    "TAFE_spatial",
    "TAFE_full",
]

METHOD_LABELS = {
    "Otsu": "Otsu",
    "Adaptive": "Adaptive",
    "Canny": "Canny",
    "Blackhat": "Black-hat",
    "UNet_no_TAFE": "U-Net w/o TAFE",
    "TAFE_channel": "TAFE-C",
    "TAFE_spatial": "TAFE-S",
    "TAFE_full": "TAFE full",
}

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_SEED = 2026

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data" / "deepcrack"
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "best_patent_model.pth"
