from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def _normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def _resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = _normalize(image)
    return np.transpose(image, (2, 0, 1))


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return np.expand_dims(mask, axis=0)


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        image_dir = self.root_dir / "images"
        label_dir = self.root_dir / "labels"
        mask_dir = self.root_dir / "masks"

        self.label_dir = label_dir if label_dir.exists() else mask_dir
        self.image_paths = sorted(image_dir.glob("*"))
        if not self.image_paths or not self.label_dir.exists():
            raise ValueError(
                f"Expected images plus labels or masks under {self.root_dir}. "
                f"Looked for {image_dir} and {label_dir}/{mask_dir}."
            )

        label_lookup = {path.stem: path for path in self.label_dir.glob("*")}
        self.samples = []
        for image_path in self.image_paths:
            label_path = label_lookup.get(image_path.stem)
            if label_path is None:
                raise ValueError(f"Missing label for image: {image_path.name}")
            self.samples.append((image_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to load sample: {image_path.name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(_resize_rgb(image, self.image_size), dtype=torch.float32)
        mask_tensor = torch.tensor(_resize_mask(mask, self.image_size), dtype=torch.float32)
        return {"image": image_tensor, "target": mask_tensor}


class ClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str | None,
        image_size: int,
        label_column: str,
    ) -> None:
        self.frame = pd.read_csv(csv_path)
        if "image_path" not in self.frame.columns and "image_id" not in self.frame.columns:
            raise ValueError("CSV must include either image_path or image_id.")
        if label_column not in self.frame.columns:
            raise ValueError(f"CSV missing label column: {label_column}")

        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        if "image_path" in row and isinstance(row["image_path"], str):
            return Path(row["image_path"])
        if self.image_root is None:
            raise ValueError("image_root is required when CSV uses image_id.")
        return self.image_root / str(row["image_id"])

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(row)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(_resize_rgb(image, self.image_size), dtype=torch.float32)
        label = int(row[self.label_column])
        return {"image": image_tensor, "target": torch.tensor(label, dtype=torch.long)}


def make_segmentation_loaders(
    train_dir: str,
    val_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = SegmentationDataset(train_dir, image_size)
    val_dataset = SegmentationDataset(val_dir, image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_classification_loaders(
    train_csv: str,
    val_csv: str,
    image_root: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    label_column: str,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ClassificationDataset(train_csv, image_root, image_size, label_column)
    val_dataset = ClassificationDataset(val_csv, image_root, image_size, label_column)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_split_csv(
    input_csv: str,
    output_dir: str,
    label_column: str,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    frame = pd.read_csv(input_csv)
    train_frame, val_frame = train_test_split(
        frame,
        test_size=val_size,
        random_state=seed,
        stratify=frame[label_column],
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    train_frame.to_csv(train_csv, index=False)
    val_frame.to_csv(val_csv, index=False)
    return train_csv, val_csv
