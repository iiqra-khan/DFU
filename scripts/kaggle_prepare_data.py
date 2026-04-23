from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import make_split_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kaggle-ready train/val CSV splits.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing CSV: {input_path}")

    frame = pd.read_csv(input_path)
    if args.label_column not in frame.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    train_csv, val_csv = make_split_csv(
        input_csv=str(input_path),
        output_dir=args.output_dir,
        label_column=args.label_column,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"Saved train split to: {train_csv}")
    print(f"Saved val split to: {val_csv}")
