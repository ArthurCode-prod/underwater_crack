"""Evaluation pipeline without plotting or manuscript-generation code."""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .baselines import classical_masks
from .config import DEFAULT_IMAGE_SIZE, METHOD_LABELS
from .data_io import load_pair, pair_image_label, read_image, resize_label
from .degradation import simulate_underwater
from .inference import count_parameters, evaluate_model_variants, load_model
from .metrics import metrics_from_counts, metrics_from_masks, summarize


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset_stats(pairs, size=DEFAULT_IMAGE_SIZE):
    ratios, widths, heights = [], [], []
    for _, label_path in pairs:
        label = read_image(label_path, cv2.IMREAD_GRAYSCALE)
        heights.append(int(label.shape[0]))
        widths.append(int(label.shape[1]))
        resized = resize_label(label, size)
        ratios.append(float(resized.mean()))
    return {
        "count": len(pairs),
        "width_min": int(np.min(widths)),
        "width_max": int(np.max(widths)),
        "height_min": int(np.min(heights)),
        "height_max": int(np.max(heights)),
        "crack_ratio_mean": float(np.mean(ratios)),
        "crack_ratio_std": float(np.std(ratios, ddof=1)),
        "crack_ratio_min": float(np.min(ratios)),
        "crack_ratio_max": float(np.max(ratios)),
    }


def prepare_test_tensors(pairs, seed=2026, profile="medium", size=DEFAULT_IMAGE_SIZE):
    tensors, labels, underwater_images, stems = [], [], [], []
    for idx, (image_path, label_path) in enumerate(pairs):
        image, label = load_pair(image_path, label_path, size)
        underwater = simulate_underwater(image, seed + idx, profile=profile)
        tensors.append(underwater.transpose(2, 0, 1).astype(np.float32))
        labels.append(label)
        underwater_images.append(underwater)
        stems.append(image_path.stem)
    return tensors, labels, underwater_images, stems


def evaluate_once(pairs, model, device, seed=2026, profile="medium", batch_size=16):
    tensors, labels, underwater_images, stems = prepare_test_tensors(pairs, seed, profile)
    model_outputs = evaluate_model_variants(model, tensors, device, batch_size=batch_size)

    per_image_rows = []
    for idx, label in enumerate(labels):
        method_masks = classical_masks(underwater_images[idx])
        method_masks.update({key: value[idx] for key, value in model_outputs.items()})
        for method, mask in method_masks.items():
            metrics = metrics_from_masks(mask, label)
            row = {"image": stems[idx], "method": method}
            row.update(metrics)
            per_image_rows.append(row)
    return per_image_rows


def run_robustness(pairs, model, device, seed=2026, robustness_count=80, batch_size=16):
    rows = []
    subset = pairs[: min(len(pairs), robustness_count)]
    for profile in ["low", "medium", "high", "severe"]:
        tensors, labels, _, _ = prepare_test_tensors(subset, seed, profile)
        outputs = evaluate_model_variants(model, tensors, device, batch_size=batch_size)
        for method in ["UNet_no_TAFE", "TAFE_full"]:
            vals = [metrics_from_masks(mask, label) for mask, label in zip(outputs[method], labels)]
            total = {key: sum(v[key] for v in vals) for key in ["tp", "fp", "fn", "tn"]}
            micro = metrics_from_counts(total)
            rows.append(
                {
                    "profile": profile,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "n": len(vals),
                    "iou_mean": float(np.mean([v["iou"] for v in vals])),
                    "dice_mean": float(np.mean([v["dice"] for v in vals])),
                    "recall_mean": float(np.mean([v["recall"] for v in vals])),
                    "iou_micro": float(micro["iou"]),
                    "dice_micro": float(micro["dice"]),
                    "recall_micro": float(micro["recall"]),
                }
            )
    return rows


def run_evaluation(data_dir: Path, model_path: Path, output_dir: Path, seed=2026, batch_size=16, max_test=0):
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = pair_image_label(data_dir / "train_img", data_dir / "train_lab")
    test_pairs = pair_image_label(data_dir / "test_img", data_dir / "test_lab")
    if max_test:
        test_pairs = test_pairs[:max_test]
    if not train_pairs or not test_pairs:
        raise RuntimeError(f"No train/test pairs found in {data_dir}")

    stats = {
        "train": build_dataset_stats(train_pairs),
        "test": build_dataset_stats(test_pairs),
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "degradation_profile": "medium",
        "note": "Underwater images are generated dynamically from the air-environment crack images.",
    }
    (tables_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path.resolve(), device)

    start = time.perf_counter()
    per_image_rows = evaluate_once(test_pairs, model, device, seed=seed, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    summary_rows = summarize(per_image_rows)
    complexity_rows = [
        {
            "model": "PatentUNet-TAFE",
            "parameters": count_parameters(model),
            "parameters_M": round(count_parameters(model) / 1e6, 4),
            "weight_size_MB": round(model_path.stat().st_size / (1024 * 1024), 3),
            "device": str(device),
            "batched_variant_inference_FPS": round(len(test_pairs) * 4 / elapsed, 2) if elapsed > 0 else 0.0,
            "input_size": "256x256",
        }
    ]
    robustness_rows = run_robustness(test_pairs, model, device, seed=seed, batch_size=batch_size)

    write_csv(tables_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(tables_dir / "method_summary.csv", summary_rows)
    write_csv(tables_dir / "complexity.csv", complexity_rows)
    write_csv(tables_dir / "robustness.csv", robustness_rows)
    return summary_rows
