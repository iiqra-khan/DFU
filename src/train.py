from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.models import forward_logits
from src.utils import ensure_dir, save_json


@dataclass
class EpochResult:
    loss: float
    metric: float


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def train_segmentation(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    output_path = ensure_dir(output_dir)

    best_metric = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"seg train {epoch}/{epochs}"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_result = evaluate_segmentation(model, val_loader, device, criterion)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_result.loss,
                "val_dice": val_result.metric,
            }
        )

        if val_result.metric > best_metric:
            best_metric = val_result.metric
            torch.save(model.state_dict(), output_path / "best_model.pt")

    save_json({"history": history, "best_val_dice": best_metric}, output_path / "metrics.json")


def evaluate_segmentation(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="seg val"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = forward_logits(model, images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            total_dice += dice_score_from_logits(logits, targets).item()

    batches = max(len(data_loader), 1)
    return EpochResult(loss=total_loss / batches, metric=total_dice / batches)


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    output_path = ensure_dir(output_dir)

    best_metric = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"cls train {epoch}/{epochs}"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_result = evaluate_classifier(model, val_loader, device, criterion)
        avg_train_loss = total_loss / max(len(train_loader), 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_result.loss,
                "val_weighted_f1": val_result.metric,
            }
        )

        if val_result.metric > best_metric:
            best_metric = val_result.metric
            torch.save(model.state_dict(), output_path / "best_model.pt")

    save_json({"history": history, "best_val_weighted_f1": best_metric}, output_path / "metrics.json")


def evaluate_classifier(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="cls val"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = forward_logits(model, images)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metric = f1_score(all_targets, all_preds, average="weighted")
    accuracy = accuracy_score(all_targets, all_preds)
    batches = max(len(data_loader), 1)
    result = EpochResult(loss=total_loss / batches, metric=metric)
    result.accuracy = accuracy  # type: ignore[attr-defined]
    return result
