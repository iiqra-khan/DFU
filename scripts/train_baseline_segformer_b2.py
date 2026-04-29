"""
Single-task SegFormer-B2 baseline for DPM Wagner severity classification.

This is intentionally separate from the two-stage transfer-learning pipeline so
paper baseline runs do not mix with Stage 1/Stage 2 checkpoints or metrics.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import DPMDataset, normalize_dpm_stratified
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegFormerB2BaselineClassifier(nn.Module):
    """SegFormer-B2 encoder with a classification head only."""

    def __init__(self, num_classes=4, pretrained_name='nvidia/mit-b2'):
        super().__init__()
        from transformers import SegformerModel

        self.encoder = SegformerModel.from_pretrained(pretrained_name)
        hidden_size = self.encoder.config.hidden_sizes[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        feat = outputs.hidden_states[-1]
        if feat.ndim == 3:
            batch_size, seq_len, channels = feat.shape
            side = int(math.sqrt(seq_len))
            feat = feat.permute(0, 2, 1).reshape(batch_size, channels, side, side)
        feat = self.pool(feat)
        return self.classifier(feat)


def get_baseline_transforms(img_size, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1_weighted = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    return avg_loss, f1_weighted


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1_weighted = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    f1_macro = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    accuracy = accuracy_score(labels_all, preds_all)
    per_class_f1 = f1_score(labels_all, preds_all, average=None, zero_division=0).tolist()
    conf_mat = confusion_matrix(labels_all, preds_all).tolist()
    return avg_loss, f1_weighted, f1_macro, accuracy, per_class_f1, conf_mat


def _has_split(dpm_root, split):
    split_root = Path(dpm_root) / split
    return split_root.exists() and any(p.is_dir() for p in split_root.iterdir())


def _build_loader(dataset, config, device, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=config.BASELINE_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=device == 'cuda',
    )


def run_single(config, dpm_root, run_id):
    seed_everything(config.BASELINE_SEED + run_id)

    device = config.DEVICE
    train_dataset = DPMDataset(
        dpm_root,
        split='train',
        transform=get_baseline_transforms(config.BASELINE_IMG_SIZE, train=True),
    )
    val_dataset = DPMDataset(
        dpm_root,
        split='valid',
        transform=get_baseline_transforms(config.BASELINE_IMG_SIZE, train=False),
    )
    test_dataset = None
    if _has_split(dpm_root, 'test'):
        test_dataset = DPMDataset(
            dpm_root,
            split='test',
            transform=get_baseline_transforms(config.BASELINE_IMG_SIZE, train=False),
        )

    train_loader = _build_loader(train_dataset, config, device, shuffle=True)
    val_loader = _build_loader(val_dataset, config, device, shuffle=False)
    test_loader = (
        _build_loader(test_dataset, config, device, shuffle=False)
        if test_dataset is not None
        else None
    )

    model = SegFormerB2BaselineClassifier(
        num_classes=config.BASELINE_NUM_CLASSES,
        pretrained_name=config.BASELINE_BACKBONE,
    ).to(device)
    if getattr(config, 'GRADIENT_CHECKPOINTING', False):
        try:
            model.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.BASELINE_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.BASELINE_EPOCHS,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = getattr(config, 'USE_AMP', False) and device == 'cuda'
    scaler = GradScaler() if use_amp else None

    best_f1_macro = -math.inf
    best_metrics = {}
    best_checkpoint_path = Path(config.BASELINE_OUTPUT_DIR) / f"best_baseline_run{run_id}.pth"

    for epoch in range(1, config.BASELINE_EPOCHS + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp
        )
        val_loss, f1_weighted, f1_macro, acc, per_class_f1, conf_mat = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()

        print(
            f"[Baseline Run {run_id}] Ep {epoch}/{config.BASELINE_EPOCHS} | "
            f"TrLoss={train_loss:.4f} TrF1={train_f1:.4f} | "
            f"ValLoss={val_loss:.4f} F1w={f1_weighted:.4f} "
            f"F1m={f1_macro:.4f} Acc={acc:.4f}"
        )

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_metrics = {
                'run': run_id,
                'epoch': epoch,
                'val_f1_weighted': float(f1_weighted),
                'val_f1_macro': float(f1_macro),
                'val_acc': float(acc),
                'per_class_f1': [float(x) for x in per_class_f1],
                'confusion_matrix': conf_mat,
            }
            torch.save(model.state_dict(), best_checkpoint_path)

    if test_loader is not None and best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        test_loss, test_f1_weighted, test_f1_macro, test_acc, test_per_class_f1, test_conf_mat = evaluate(
            model, test_loader, criterion, device, use_amp
        )
        best_metrics['test_metrics'] = {
            'test_loss': float(test_loss),
            'test_f1_weighted': float(test_f1_weighted),
            'test_f1_macro': float(test_f1_macro),
            'test_acc': float(test_acc),
            'test_per_class_f1': [float(x) for x in test_per_class_f1],
            'test_confusion_matrix': test_conf_mat,
        }
        print(
            f"[Baseline Run {run_id}] Test | "
            f"Loss={test_loss:.4f} F1w={test_f1_weighted:.4f} "
            f"F1m={test_f1_macro:.4f} Acc={test_acc:.4f}"
        )

    return best_metrics


def run_multiple(config, dpm_root):
    os.makedirs(config.BASELINE_OUTPUT_DIR, exist_ok=True)

    all_results = []
    for run_id in range(config.BASELINE_NUM_RUNS):
        metrics = run_single(config, dpm_root, run_id)
        all_results.append(metrics)

    f1_weighted = [r['val_f1_weighted'] for r in all_results]
    f1_macro = [r['val_f1_macro'] for r in all_results]
    accuracy = [r['val_acc'] for r in all_results]
    summary = {
        'model': 'SegFormer-B2 Single-Task Baseline',
        'dataset': 'DPM Wagner grade classification',
        'runs': all_results,
        'f1_weighted_mean': float(np.mean(f1_weighted)),
        'f1_weighted_std': float(np.std(f1_weighted)),
        'f1_macro_mean': float(np.mean(f1_macro)),
        'f1_macro_std': float(np.std(f1_macro)),
        'accuracy_mean': float(np.mean(accuracy)),
        'accuracy_std': float(np.std(accuracy)),
    }

    test_results = [r.get('test_metrics') for r in all_results if r.get('test_metrics')]
    if test_results:
        test_f1_weighted = [r['test_f1_weighted'] for r in test_results]
        test_f1_macro = [r['test_f1_macro'] for r in test_results]
        test_accuracy = [r['test_acc'] for r in test_results]
        summary.update({
            'test_f1_weighted_mean': float(np.mean(test_f1_weighted)),
            'test_f1_weighted_std': float(np.std(test_f1_weighted)),
            'test_f1_macro_mean': float(np.mean(test_f1_macro)),
            'test_f1_macro_std': float(np.std(test_f1_macro)),
            'test_accuracy_mean': float(np.mean(test_accuracy)),
            'test_accuracy_std': float(np.std(test_accuracy)),
        })

    print("\n========== BASELINE SUMMARY ==========")
    print(f"Weighted F1 : {summary['f1_weighted_mean']:.4f} +/- {summary['f1_weighted_std']:.4f}")
    print(f"Macro F1    : {summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")
    print(f"Accuracy    : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    if test_results:
        print("\n========== TEST SUMMARY ==========")
        print(f"Weighted F1 : {summary['test_f1_weighted_mean']:.4f} +/- {summary['test_f1_weighted_std']:.4f}")
        print(f"Macro F1    : {summary['test_f1_macro_mean']:.4f} +/- {summary['test_f1_macro_std']:.4f}")
        print(f"Accuracy    : {summary['test_accuracy_mean']:.4f} +/- {summary['test_accuracy_std']:.4f}")

    summary_path = Path(config.BASELINE_OUTPUT_DIR) / 'baseline_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved baseline summary to {summary_path}")

    return summary


def prepare_dpm_root(config, normalize=True):
    if not normalize:
        return config.DPM_PATH

    normalized_root = normalize_dpm_stratified(
        config.DPM_PATH,
        output_root=config.BASELINE_NORMALIZED_DPM_PATH,
        val_ratio=config.BASELINE_VAL_RATIO,
        seed=config.BASELINE_SEED,
    )
    config.DPM_PATH = str(normalized_root)
    print(f"DPM root: {config.DPM_PATH}")
    return config.DPM_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Train single-task SegFormer-B2 DPM baseline")
    parser.add_argument('--dpm-path', default=None, help='Override Config.DPM_PATH')
    parser.add_argument('--output-dir', default=None, help='Override Config.BASELINE_OUTPUT_DIR')
    parser.add_argument('--epochs', type=int, default=None, help='Override Config.BASELINE_EPOCHS')
    parser.add_argument('--runs', type=int, default=None, help='Override Config.BASELINE_NUM_RUNS')
    parser.add_argument('--batch-size', type=int, default=None, help='Override Config.BASELINE_BATCH_SIZE')
    parser.add_argument('--img-size', type=int, default=None, help='Override Config.BASELINE_IMG_SIZE')
    parser.add_argument('--no-normalize-dpm', action='store_true', help='Use DPM path exactly as provided')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dpm_path:
        Config.DPM_PATH = args.dpm_path
    if args.output_dir:
        Config.BASELINE_OUTPUT_DIR = args.output_dir
    if args.epochs is not None:
        Config.BASELINE_EPOCHS = args.epochs
    if args.runs is not None:
        Config.BASELINE_NUM_RUNS = args.runs
    if args.batch_size is not None:
        Config.BASELINE_BATCH_SIZE = args.batch_size
    if args.img_size is not None:
        Config.BASELINE_IMG_SIZE = args.img_size

    dpm_root = prepare_dpm_root(
        Config,
        normalize=getattr(Config, 'BASELINE_NORMALIZE_DPM', True) and not args.no_normalize_dpm,
    )
    run_multiple(Config, dpm_root)
