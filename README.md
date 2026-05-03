# Underwater Crack Segmentation with Synthetic Turbidity Degradation

This repository contains the code, pretrained weights, and dataset split used in the manuscript:

**Fine-Grained Crack Segmentation under Simulated Turbid Underwater Degradation**

The repository is intentionally focused on reproducible experiments. It does not include manuscript-generation scripts or figure-rendering scripts.

## Repository Structure

```text
underwater_crack_release/
  data/deepcrack/             # train/test images and binary labels
  models/                     # pretrained PatentUNet-TAFE checkpoint
  results/tables/             # CSV/JSON tables reported in the manuscript
  scripts/
    run_evaluation.py         # reproduce metric tables
    train_model.py            # optional retraining entry point
  src/underwater_crack/
    data_io.py                # image/label pairing and loading
    degradation.py            # synthetic underwater degradation
    baselines.py              # Otsu, adaptive threshold, Canny, Black-hat
    model.py                  # PatentUNet and TAFE module
    inference.py              # checkpoint loading and ablation inference
    metrics.py                # IoU, Dice, precision, recall, specificity
    evaluate.py               # tabular evaluation pipeline
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For CPU-only evaluation, install the CPU build of PyTorch if needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproduce Evaluation Tables

```bash
python scripts/run_evaluation.py --data-dir data/deepcrack --model-path models/best_patent_model.pth --output-dir results
```

The command writes:

- `results/tables/per_image_metrics.csv`
- `results/tables/method_summary.csv`
- `results/tables/robustness.csv`
- `results/tables/complexity.csv`
- `results/tables/dataset_stats.json`

## Dataset Note

The included data split is based on the DeepCrack crack images and labels. The underwater scenes in the paper are not real underwater acquisitions; they are generated dynamically from the crack images through channel attenuation, scattering blur, and additive noise. This design is intended for controlled validation under simulated underwater degradation.

## Citation

If this repository is used, please cite the corresponding manuscript and acknowledge that the underwater data are synthetically degraded samples rather than real underwater field images.
