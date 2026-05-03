"""Minimal training entry point for PatentUNet-TAFE."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from underwater_crack.datasets import SyntheticUnderwaterCrackDataset
from underwater_crack.losses import BCEDiceLoss
from underwater_crack.model import PatentUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/deepcrack"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("models/patentunet_tafe_retrained.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticUnderwaterCrackDataset(
        args.data_dir / "train_img",
        args.data_dir / "train_lab",
        profile="medium",
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = PatentUNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * images.size(0)
        print(f"epoch={epoch:03d} loss={total / len(dataset):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
