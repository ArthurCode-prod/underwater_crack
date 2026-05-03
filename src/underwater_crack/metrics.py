"""Segmentation metrics and summary helpers."""

import numpy as np

from .config import METHOD_LABELS, METHOD_ORDER


def metrics_from_masks(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())
    out = metrics_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    out.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out


def metrics_from_counts(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    eps = 1e-7
    return {
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
        "accuracy": (tp + tn + eps) / (tp + fp + fn + tn + eps),
    }


def summarize(per_image_rows):
    summary = []
    for method in METHOD_ORDER:
        rows = [r for r in per_image_rows if r["method"] == method]
        if not rows:
            continue
        total = {k: sum(r[k] for r in rows) for k in ["tp", "fp", "fn", "tn"]}
        micro = metrics_from_counts(total)
        row = {"method": method, "label": METHOD_LABELS[method], "n": len(rows)}
        for key in ["iou", "dice", "precision", "recall", "specificity", "accuracy"]:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{key}_micro"] = float(micro[key])
        summary.append(row)
    return summary
