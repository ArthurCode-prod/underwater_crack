"""Model loading and batched inference."""

from pathlib import Path

import numpy as np
import torch

from .model import PatentUNet


def load_model(model_path: Path, device: torch.device):
    model = PatentUNet().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate_model_variants(model, tensors, device, batch_size=16):
    outputs = {}
    variants = {
        "UNet_no_TAFE": "identity",
        "TAFE_channel": "channel",
        "TAFE_spatial": "spatial",
        "TAFE_full": "full",
    }
    x = torch.from_numpy(np.stack(tensors)).float()
    with torch.inference_mode():
        for method, mode in variants.items():
            model.set_tafe_mode(mode)
            method_probs = []
            for start in range(0, len(x), batch_size):
                batch = x[start : start + batch_size].to(device)
                probs = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
                method_probs.extend(probs)
            outputs[method] = [(p >= 0.5).astype(np.uint8) for p in method_probs]
    model.set_tafe_mode("full")
    return outputs
