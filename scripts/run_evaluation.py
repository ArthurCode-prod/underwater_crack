"""Run evaluation and export CSV/JSON result tables."""

import argparse
from pathlib import Path

from underwater_crack.config import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_SEED
from underwater_crack.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_evaluation(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_test=args.max_test,
    )
    for row in rows:
        print(
            f"{row['label']}: "
            f"IoU={row['iou_mean']:.4f}, "
            f"Dice={row['dice_mean']:.4f}, "
            f"Precision={row['precision_mean']:.4f}, "
            f"Recall={row['recall_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
