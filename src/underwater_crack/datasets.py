"""PyTorch dataset for dynamic synthetic underwater degradation."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_IMAGE_SIZE
from .data_io import load_pair, pair_image_label
from .degradation import simulate_underwater


class SyntheticUnderwaterCrackDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, profile="medium", seed=2026, size=DEFAULT_IMAGE_SIZE):
        self.pairs = pair_image_label(image_dir, label_dir)
        self.profile = profile
        self.seed = seed
        self.size = size
        if not self.pairs:
            raise RuntimeError(f"No image-label pairs found under {image_dir} and {label_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        image, label = load_pair(image_path, label_path, self.size)
        degraded = simulate_underwater(image, self.seed + index, self.profile)
        tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).float()
        target = torch.from_numpy(label[None]).float()
        return tensor, target
