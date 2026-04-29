"""
Stage 2: Wagner Severity Grading using SegFormer-based Classifier

This module implements a 4-class Wagner severity classifier that uses a SegFormer encoder
pretrained on ulcer segmentation (Stage 1) to learn localization features, then fine-tunes
on Wagner grading.

Key features:
- SegFormer-B2 encoder loaded from Stage 1 (strong architectural compatibility)
- Progressive layer unfreezing: freeze encoder initially, gradually unfreeze during training
- Class-weighted CrossEntropyLoss to handle grade imbalance in DPM dataset
- Tracks both weighted F1 (main metric) and macro F1 (fairness metric)
- Supports ablation studies (scratch, ImageNet-only, two-stage transfer)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix
)
import os
from tqdm import tqdm
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import importlib
import random
from pathlib import Path

from dataset import DPMDataset, get_transforms
from config import Config

try:
    wandb = importlib.import_module("wandb")
except Exception:
    wandb = None


class SegFormerWagnerClassifier(nn.Module):
    """SegFormer-based 4-class Wagner classifier.
    
    Uses pretrained SegFormer-B2 encoder from Stage 1 segmentation,
    adds a global average pooling + classification head for 4-class prediction.
    """
    
    def __init__(self, num_classes=4, encoder_weights=None):
        super().__init__()

        transformers_module = importlib.import_module("transformers")
        segformer_model_cls = getattr(transformers_module, "SegformerModel", None)
        if segformer_model_cls is None:
            raise ImportError(
                "transformers is required for Stage 2. Install it with `pip install transformers`."
            )
        
        # Load the encoder-only SegFormer backbone to avoid decoder warnings.
        self.backbone = segformer_model_cls.from_pretrained(Config.SEGFORMER_MODEL)
        
        # Get encoder hidden size (for SegFormer-B2: 768)
        self.hidden_size = self.backbone.config.hidden_sizes[-1]  # Last layer hidden size
        
        # Classification head: global avg pool + FC
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Load encoder weights from Stage 1 if provided
        if encoder_weights is not None:
            self._load_encoder_weights(encoder_weights)
    
    def _load_encoder_weights(self, encoder_weights):
        """Load pretrained Stage 1 encoder weights."""
        encoder_state_dict = torch.load(encoder_weights, map_location='cpu')
        try:
            missing_keys, unexpected_keys = self.backbone.encoder.load_state_dict(
                encoder_state_dict,
                strict=False,
            )
            print(f"✅ Loaded Stage 1 encoder weights from {encoder_weights}")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
        except Exception as e:
            print(f"⚠️ Error loading encoder weights: {e}")
    
    def get_encoder_blocks(self):
        """Get encoder block modules for progressive unfreezing."""
        blocks = []
        for i in range(len(self.backbone.encoder.block)):
            blocks.append(self.backbone.encoder.block[i])
        return blocks
    
    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_block(self, block_idx):
        """Unfreeze specific encoder block by index."""
        try:
            block = self.backbone.encoder.block[block_idx]
            for param in block.parameters():
                param.requires_grad = True
            return True
        except (AttributeError, IndexError):
            pass
        return False
    
    def forward(self, x):
        """Forward pass through encoder and classifier head."""
        # SegFormer encoder output
        encoder_output = self.backbone(pixel_values=x, output_hidden_states=True)
        last_hidden_state = encoder_output.hidden_states[-1]  # (B, C, H, W)
        
        # Global average pooling: (B, C, H, W) -> (B, C)
        pooled = self.pool(last_hidden_state).squeeze(-1).squeeze(-1)
        
        # Classification head: (B, C) -> (B, num_classes)
        logits = self.classifier(pooled)
        
        return logits


def _maybe_init_wandb(config, mode_name):
    """Initialize Weights & Biases tracking if enabled."""
    if not getattr(config, 'USE_WANDB', False):
        return None
    if wandb is None:
        print("⚠️ wandb is not installed; continuing without experiment tracking.")
        return None

    run = wandb.init(
        project=getattr(config, 'WANDB_PROJECT', 'dfu-pipeline'),
        entity=getattr(config, 'WANDB_ENTITY', None),
        name=f"{mode_name}-stage2-{getattr(config, 'WANDB_RUN_NAME', '')}".strip('-'),
        config={
            'stage': 'Stage 2 (Wagner)',
            'model': 'SegFormer-B2',
            'mode': mode_name,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS_STAGE2,
            'lr': config.LR_STAGE2,
            'weight_decay': config.WEIGHT_DECAY,
        },
        reinit=True,
    )
    return run


def compute_class_weights(dataset, num_classes=4):
    """Compute class weights for imbalanced dataset."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Inverse frequency weighting
    total = len(labels)
    weights = total / (num_classes * (class_counts + 1))  # +1 to avoid division by zero
    weights = torch.tensor(weights, dtype=torch.float32)
    
    return weights, class_counts


def seed_everything(seed):
    """Seed the main RNGs for repeatable independent runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _has_split(dpm_root, split):
    split_root = Path(dpm_root) / split
    return split_root.exists() and any(p.is_dir() for p in split_root.iterdir())


def _save_stage2_artifacts(config, history, best_metrics, save_suffix=''):
    if not getattr(config, 'SAVE_METRICS_JSON', True):
        return

    suffix = save_suffix or ''
    history_path = os.path.join(config.OUTPUT_DIR, f'stage2_history{suffix}.json')
    best_metrics_path = os.path.join(config.OUTPUT_DIR, f'stage2_best_metrics{suffix}.json')

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    with open(best_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=2)

    if not getattr(config, 'SAVE_PLOTS', True):
        return

    # Training curves
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(history['epoch'], history['train_loss'], label='Train Loss', color='tab:blue')
    if history['val_epoch']:
        ax.plot(history['val_epoch'], history['val_loss'], label='Val Loss', color='tab:orange', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Stage 2 Training Loss')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, f'stage2_loss_curves{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(history['epoch'], history['train_f1_weighted'], label='Train Weighted F1', color='tab:green')
    if history['val_epoch']:
        ax.plot(history['val_epoch'], history['val_f1_weighted'], label='Val Weighted F1', color='tab:red', marker='o')
        ax.plot(history['val_epoch'], history['val_f1_macro'], label='Val Macro F1', color='tab:purple', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_title('Stage 2 F1 Curves')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, f'stage2_f1_curves{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if best_metrics and best_metrics.get('confusion_matrix') is not None:
        cm = np.array(best_metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title('Best Stage 2 Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels([str(i + 1) for i in range(cm.shape[1])])
        ax.set_yticklabels([str(i + 1) for i in range(cm.shape[0])])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(config.OUTPUT_DIR, f'stage2_confusion_matrix{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def _save_confusion_matrix_plot(config, conf_mat, title, filename):
    if not getattr(config, 'SAVE_PLOTS', True) or conf_mat is None:
        return

    cm = np.array(conf_mat)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels([str(i + 1) for i in range(cm.shape[1])])
    ax.set_yticklabels([str(i + 1) for i in range(cm.shape[0])])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_multi_run_summary_plot(config, summary, mode):
    if not getattr(config, 'SAVE_PLOTS', True):
        return

    labels = []
    means = []
    stds = []

    metric_specs = [
        ('Val F1w', 'val_f1_weighted_mean', 'val_f1_weighted_std'),
        ('Val F1m', 'val_f1_macro_mean', 'val_f1_macro_std'),
        ('Val Acc', 'val_accuracy_mean', 'val_accuracy_std'),
        ('Test F1w', 'test_f1_weighted_mean', 'test_f1_weighted_std'),
        ('Test F1m', 'test_f1_macro_mean', 'test_f1_macro_std'),
        ('Test Acc', 'test_accuracy_mean', 'test_accuracy_std'),
    ]

    for label, mean_key, std_key in metric_specs:
        if summary.get(mean_key) is not None:
            labels.append(label)
            means.append(summary[mean_key])
            stds.append(summary.get(std_key, 0.0) or 0.0)

    if not labels:
        return

    colors = ['tab:blue' if label.startswith('Val') else 'tab:green' for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'Stage 2 Multi-Run Summary ({mode})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    for idx, value in enumerate(means):
        ax.text(idx, min(value + 0.03, 0.98), f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    output_name = f'stage2_{mode.replace("-", "_")}_multi_run_summary.png'
    fig.savefig(os.path.join(config.OUTPUT_DIR, output_name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []

    # Support AMP and gradient accumulation via config passed on model object (or global Config)
    config = getattr(model, '_training_config', None)
    use_amp = getattr(config, 'USE_AMP', False) if config is not None else False
    accum_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1) if config is not None else 1
    scaler = getattr(model, '_amp_scaler', None)

    optimizer.zero_grad()
    for step_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accum_steps

        preds = logits.argmax(dim=1).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Final flush if needed
    if (step_idx + 1) % accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / len(train_loader)
    train_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, train_f1_weighted, train_acc


def validate_epoch(model, val_loader, criterion, device, config=None):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = getattr(config, 'USE_AMP', False) if config is not None else False
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    return (avg_val_loss, val_f1_weighted, val_f1_macro, val_acc, 
            all_preds, all_labels, precision, recall, f1)


def train_wagner_stage2(
    config,
    mode='two-stage',
    encoder_weights_path=None,
    save_suffix='',
    use_class_weights=None,
    seed=None,
    evaluate_test=True
):
    """
    Train Stage 2 Wagner classifier with optional progressive unfreezing.
    
    Args:
        config: Configuration object
        mode: 'scratch' | 'imagenet' | 'two-stage'
        encoder_weights_path: Path to Stage 1 encoder (if mode='two-stage')
        save_suffix: Suffix for checkpoint filenames (for ablations)
        use_class_weights: Optional override for class weighting
    """
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    if seed is not None:
        seed_everything(seed)
    
    print(f"\n{'='*70}")
    print(f"Stage 2: Wagner Grading ({mode.upper()})")
    print(f"{'='*70}")

    if use_class_weights is None:
        use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS_STAGE2', True)
    
    # Build model
    model = SegFormerWagnerClassifier(num_classes=4)
    
    # Load weights based on mode
    if mode == 'two-stage' and encoder_weights_path:
        # Two-stage transfer: load Stage 1 encoder
        encoder_file = (
            encoder_weights_path or 
            f"{config.OUTPUT_DIR}/encoder_pretrained.pth"
        )
        if os.path.exists(encoder_file):
            model._load_encoder_weights(encoder_file)
            model.freeze_encoder()
        else:
            print(f"⚠️ Stage 1 encoder not found at {encoder_file}, training without pretrain")
    elif mode == 'scratch':
        # Scratch: don't load pretrained weights
        print("Training from scratch (no pretrained weights)")
    
    model = model.to(device)
    # Enable gradient checkpointing on backbone if configured
    if getattr(config, 'GRADIENT_CHECKPOINTING', False):
        try:
            model.backbone.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on SegFormer backbone")
        except Exception:
            pass
    
    # Data
    train_dataset = DPMDataset(
        config.DPM_PATH, 
        split='train',
        transform=get_transforms('train')
    )
    val_dataset = DPMDataset(
        config.DPM_PATH, 
        split='valid',
        transform=get_transforms('val')
    )
    test_dataset = None
    if evaluate_test and _has_split(config.DPM_PATH, 'test'):
        test_dataset = DPMDataset(
            config.DPM_PATH,
            split='test',
            transform=get_transforms('val')
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        if test_dataset is not None
        else None
    )
    
    # Compute class weights and distributions from training and validation data
    class_weights, train_class_counts = compute_class_weights(train_dataset, num_classes=4)

    # Validation distribution (for inspection only)
    val_labels = np.array([val_dataset[i][1] for i in range(len(val_dataset))])
    val_class_counts = np.bincount(val_labels, minlength=4)

    print(f"\nClass distribution in training set: {train_class_counts}")
    print(f"Class distribution in validation set: {val_class_counts}")
    if test_dataset is not None:
        test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
        test_class_counts = np.bincount(test_labels, minlength=4)
        print(f"Class distribution in test set: {test_class_counts}")
    print(f"Class weights (used for loss): {class_weights}")

    # Save class distributions so they are visible in Kaggle Outputs / job artifacts
    try:
        dist_path = os.path.join(config.OUTPUT_DIR, 'stage2_class_distribution.json')
        with open(dist_path, 'w', encoding='utf-8') as f:
            json.dump({
                'train_counts': train_class_counts.tolist() if hasattr(train_class_counts, 'tolist') else train_class_counts,
                'val_counts': val_class_counts.tolist() if hasattr(val_class_counts, 'tolist') else val_class_counts,
                'test_counts': test_class_counts.tolist() if test_dataset is not None else None,
                'train_total': int(len(train_dataset)),
                'val_total': int(len(val_dataset)),
                'test_total': int(len(test_dataset)) if test_dataset is not None else 0
            }, f, indent=2)
        print(f"Saved class distributions to {dist_path}")
    except Exception as e:
        print(f"⚠️ Could not save class distribution JSON: {e}")
    
    # Loss with class weighting
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR_STAGE2,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS_STAGE2,
        eta_min=1e-6
    )
    
    # W&B
    wandb_run = _maybe_init_wandb(config, mode)
    
    # Training loop
    best_f1_weighted = -math.inf
    best_f1_macro = -math.inf
    epochs_without_improvement = 0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_f1_weighted': [],
        'train_acc': [],
        'val_epoch': [],
        'val_loss': [],
        'val_f1_weighted': [],
        'val_f1_macro': [],
        'val_acc': [],
    }
    best_metrics = {}
    
    for epoch in range(config.EPOCHS_STAGE2):
        # Progressive unfreezing
        if mode == 'two-stage' and epoch == config.UNFREEZE_ENCODER_EPOCH_STAGE2:
            print(f"\n🔓 Epoch {epoch}: Starting to unfreeze encoder blocks")
            model.freeze_encoder()  # Reset freeze state
        
        if (mode == 'two-stage' and 
            epoch >= config.UNFREEZE_ENCODER_EPOCH_STAGE2 and
            (epoch - config.UNFREEZE_ENCODER_EPOCH_STAGE2) % config.UNFREEZE_INTERVAL_EPOCHS == 0):
            
            block_to_unfreeze = (epoch - config.UNFREEZE_ENCODER_EPOCH_STAGE2) // config.UNFREEZE_INTERVAL_EPOCHS
            if model.unfreeze_block(block_to_unfreeze):
                print(f"   → Unfroze encoder block {block_to_unfreeze}")
        
        # Training
        # Attach config and scaler to model for train_epoch convenience
        # (avoids changing many signatures)
        model._training_config = config
        model._amp_scaler = GradScaler() if getattr(config, 'USE_AMP', False) else None

        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
        )
        
        history['train_loss'].append(train_loss)
        history['train_f1_weighted'].append(train_f1)
        history['train_acc'].append(train_acc)
        history['epoch'].append(epoch + 1)

        should_validate = (
            (epoch + 1) % getattr(config, 'VALIDATE_EVERY_N_EPOCHS', 1) == 0
            or (epoch + 1) == config.EPOCHS_STAGE2
        )

        if should_validate:
            # Validation
            (val_loss, val_f1_weighted, val_f1_macro, val_acc,
             all_preds, all_labels, precision, recall, class_f1) = validate_epoch(
                model, val_loader, criterion, device, config=config
            )

            precision = np.atleast_1d(precision)
            recall = np.atleast_1d(recall)
            class_f1 = np.atleast_1d(class_f1)
            
            history['val_loss'].append(val_loss)
            history['val_f1_weighted'].append(val_f1_weighted)
            history['val_f1_macro'].append(val_f1_macro)
            history['val_acc'].append(val_acc)
            history['val_epoch'].append(epoch + 1)
            
            print(f"\nEpoch {epoch+1}/{config.EPOCHS_STAGE2}")
            print(f"  Train: Loss={train_loss:.4f}, F1_w={train_f1:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, F1_w={val_f1_weighted:.4f}, F1_m={val_f1_macro:.4f}, Acc={val_acc:.4f}")
            print(f"  Per-class F1: {[f'{score:.3f}' for score in class_f1]}")
            
            # Save best model (based on weighted F1)
            if val_f1_weighted > (best_f1_weighted + config.EARLY_STOPPING_MIN_DELTA):
                best_f1_weighted = val_f1_weighted
                best_f1_macro = val_f1_macro
                epochs_without_improvement = 0
                
                checkpoint_name = f"best_wagner_model{save_suffix}.pth"
                torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/{checkpoint_name}")
                print(f"  ✅ Saved best model: F1_w={best_f1_weighted:.4f}, F1_m={best_f1_macro:.4f}")
                
                # Save metrics
                best_metrics = {
                    'epoch': epoch + 1,
                    'val_f1_weighted': float(val_f1_weighted),
                    'val_f1_macro': float(val_f1_macro),
                    'val_acc': float(val_acc),
                    'per_class_f1': [float(x) for x in class_f1],
                    'per_class_precision': [float(x) for x in precision],
                    'per_class_recall': [float(x) for x in recall],
                    'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
                }
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if (config.USE_EARLY_STOPPING and
                epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE_STAGE2):
                print(f"\n⏹️ Early stopping at epoch {epoch+1} "
                      f"(no F1 improvement for {config.EARLY_STOPPING_PATIENCE_STAGE2} epochs)")
                break
        else:
            print(f"\nEpoch {epoch+1}/{config.EPOCHS_STAGE2}")
            print(f"  Train: Loss={train_loss:.4f}, F1_w={train_f1:.4f}, Acc={train_acc:.4f}")
            print(f"  Val: skipped this epoch")
        
        # LR scheduler step
        scheduler.step()
        
        # W&B logging
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'train_f1': train_f1,
                'train_acc': train_acc,
                'val_loss': locals().get('val_loss'),
                'val_f1_weighted': locals().get('val_f1_weighted'),
                'val_f1_macro': locals().get('val_f1_macro'),
                'val_acc': locals().get('val_acc'),
            })
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ Stage 2 ({mode.upper()}) Complete")
    print(f"   Best Weighted F1: {best_f1_weighted:.4f}")
    print(f"   Best Macro F1: {best_f1_macro:.4f}")
    print(f"   Metrics saved with checkpoint")
    print(f"{'='*70}\n")

    _save_stage2_artifacts(config, history, best_metrics, save_suffix=save_suffix)

    if test_loader is not None and best_metrics:
        checkpoint_name = f"best_wagner_model{save_suffix}.pth"
        checkpoint_path = os.path.join(config.OUTPUT_DIR, checkpoint_name)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        (test_loss, test_f1_weighted, test_f1_macro, test_acc,
         test_preds, test_labels, test_precision, test_recall, test_class_f1) = validate_epoch(
            model, test_loader, criterion, device, config=config
        )
        test_metrics = {
            'test_loss': float(test_loss),
            'test_f1_weighted': float(test_f1_weighted),
            'test_f1_macro': float(test_f1_macro),
            'test_acc': float(test_acc),
            'test_per_class_f1': [float(x) for x in np.atleast_1d(test_class_f1)],
            'test_per_class_precision': [float(x) for x in np.atleast_1d(test_precision)],
            'test_per_class_recall': [float(x) for x in np.atleast_1d(test_recall)],
            'test_confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
        }
        best_metrics['test_metrics'] = test_metrics
        test_metrics_path = os.path.join(config.OUTPUT_DIR, f'stage2_test_metrics{save_suffix}.json')
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2)
        _save_confusion_matrix_plot(
            config,
            test_metrics['test_confusion_matrix'],
            'Held-Out Test Confusion Matrix',
            f'stage2_test_confusion_matrix{save_suffix}.png'
        )
        print(
            f"Test: Loss={test_loss:.4f}, F1_w={test_f1_weighted:.4f}, "
            f"F1_m={test_f1_macro:.4f}, Acc={test_acc:.4f}"
        )
        print(f"Saved test metrics to {test_metrics_path}")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    return {
        'best_f1_weighted': best_f1_weighted,
        'best_f1_macro': best_f1_macro,
        'best_metrics': best_metrics,
        'history': history
    }


def run_multiple_stage2(config, mode='two-stage', runs=3, encoder_weights_path=None, use_class_weights=None):
    """Run Stage 2 multiple times and summarize validation/test mean +/- std."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_results = []
    base_seed = getattr(config, 'BASELINE_SEED', 42)

    for run_id in range(runs):
        suffix = f"_{mode.replace('-', '_')}_run{run_id}"
        result = train_wagner_stage2(
            config,
            mode=mode,
            encoder_weights_path=encoder_weights_path,
            save_suffix=suffix,
            use_class_weights=use_class_weights,
            seed=base_seed + run_id,
            evaluate_test=True,
        )
        result['run'] = run_id
        result['seed'] = base_seed + run_id
        all_results.append(result)

    val_f1_weighted = [r['best_metrics']['val_f1_weighted'] for r in all_results if r.get('best_metrics')]
    val_f1_macro = [r['best_metrics']['val_f1_macro'] for r in all_results if r.get('best_metrics')]
    val_acc = [r['best_metrics']['val_acc'] for r in all_results if r.get('best_metrics')]

    summary = {
        'model': f'Stage 2 SegFormer-B2 ({mode})',
        'dataset': 'DPM Wagner grade classification',
        'num_runs': int(runs),
        'runs': [
            {
                'run': int(r['run']),
                'seed': int(r['seed']),
                'best_metrics': r.get('best_metrics', {}),
            }
            for r in all_results
        ],
        'val_f1_weighted_mean': float(np.mean(val_f1_weighted)) if val_f1_weighted else None,
        'val_f1_weighted_std': float(np.std(val_f1_weighted)) if val_f1_weighted else None,
        'val_f1_macro_mean': float(np.mean(val_f1_macro)) if val_f1_macro else None,
        'val_f1_macro_std': float(np.std(val_f1_macro)) if val_f1_macro else None,
        'val_accuracy_mean': float(np.mean(val_acc)) if val_acc else None,
        'val_accuracy_std': float(np.std(val_acc)) if val_acc else None,
    }

    test_metrics = [
        r['best_metrics']['test_metrics']
        for r in all_results
        if r.get('best_metrics', {}).get('test_metrics')
    ]
    if test_metrics:
        test_f1_weighted = [r['test_f1_weighted'] for r in test_metrics]
        test_f1_macro = [r['test_f1_macro'] for r in test_metrics]
        test_acc = [r['test_acc'] for r in test_metrics]
        summary.update({
            'test_f1_weighted_mean': float(np.mean(test_f1_weighted)),
            'test_f1_weighted_std': float(np.std(test_f1_weighted)),
            'test_f1_macro_mean': float(np.mean(test_f1_macro)),
            'test_f1_macro_std': float(np.std(test_f1_macro)),
            'test_accuracy_mean': float(np.mean(test_acc)),
            'test_accuracy_std': float(np.std(test_acc)),
        })

    summary_path = os.path.join(config.OUTPUT_DIR, f'stage2_{mode.replace("-", "_")}_multi_run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    _save_multi_run_summary_plot(config, summary, mode)

    print("\n========== STAGE 2 MULTI-RUN SUMMARY ==========")
    print(f"Val Weighted F1 : {summary['val_f1_weighted_mean']:.4f} +/- {summary['val_f1_weighted_std']:.4f}")
    print(f"Val Macro F1    : {summary['val_f1_macro_mean']:.4f} +/- {summary['val_f1_macro_std']:.4f}")
    print(f"Val Accuracy    : {summary['val_accuracy_mean']:.4f} +/- {summary['val_accuracy_std']:.4f}")
    if test_metrics:
        print("\n========== HELD-OUT TEST SUMMARY ==========")
        print(f"Test Weighted F1 : {summary['test_f1_weighted_mean']:.4f} +/- {summary['test_f1_weighted_std']:.4f}")
        print(f"Test Macro F1    : {summary['test_f1_macro_mean']:.4f} +/- {summary['test_f1_macro_std']:.4f}")
        print(f"Test Accuracy    : {summary['test_accuracy_mean']:.4f} +/- {summary['test_accuracy_std']:.4f}")
    print(f"Saved multi-run summary to {summary_path}")

    return summary


def run_ablation_study(config):
    """Run full ablation study with 3 configurations."""
    
    results = {}
    
    # 1. Scratch: no pretrained weights
    print("\n" + "="*70)
    print("ABLATION 1/3: Training from Scratch")
    print("="*70)
    results['scratch'] = train_wagner_stage2(
        config,
        mode='scratch',
        save_suffix='_scratch'
    )
    
    # 2. ImageNet-only: SegFormer pretrained on ImageNet/ADE but no Stage 1 encoder
    print("\n" + "="*70)
    print("ABLATION 2/3: ImageNet Pretrained (no Stage 1)")
    print("="*70)
    results['imagenet'] = train_wagner_stage2(
        config,
        mode='imagenet',
        save_suffix='_imagenet'
    )
    
    # 3. Two-stage transfer: load Stage 1 encoder
    print("\n" + "="*70)
    print("ABLATION 3/3: Two-Stage Transfer Learning")
    print("="*70)
    encoder_path = f"{config.OUTPUT_DIR}/encoder_pretrained.pth"
    results['two_stage'] = train_wagner_stage2(
        config,
        mode='two-stage',
        encoder_weights_path=encoder_path,
        save_suffix='_two_stage'
    )
    
    # Save ablation results
    ablation_file = f"{config.OUTPUT_DIR}/ablation_results.json"
    ablation_summary = {
        'scratch': {
            'best_f1_weighted': float(results['scratch']['best_f1_weighted']),
            'best_f1_macro': float(results['scratch']['best_f1_macro']),
        },
        'imagenet': {
            'best_f1_weighted': float(results['imagenet']['best_f1_weighted']),
            'best_f1_macro': float(results['imagenet']['best_f1_macro']),
        },
        'two_stage': {
            'best_f1_weighted': float(results['two_stage']['best_f1_weighted']),
            'best_f1_macro': float(results['two_stage']['best_f1_macro']),
        }
    }
    
    with open(ablation_file, 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    for mode, result in ablation_summary.items():
        print(f"{mode:>15}: F1_w={result['best_f1_weighted']:.4f}, F1_m={result['best_f1_macro']:.4f}")
    print(f"Results saved to: {ablation_file}")
    print(f"{'='*70}\n")
    
    return results


def train_wagner_grading(config):
    """
    Backward-compatible wrapper for Stage 2 training.
    
    This function maintains the old calling convention: train_wagner_grading(Config)
    Defaults to two-stage transfer learning (recommended mode).
    
    For more control, use train_wagner_stage2() directly with mode parameter.
    """
    encoder_path = f"{config.OUTPUT_DIR}/encoder_pretrained.pth"
    return train_wagner_stage2(
        config,
        mode='two-stage',
        encoder_weights_path=encoder_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 2 Wagner classifier")
    parser.add_argument(
        '--mode',
        choices=['scratch', 'imagenet', 'two-stage', 'ablation'],
        default='two-stage',
        help='Training mode'
    )
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='Disable class weighting'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of independent runs. Use 3 for paper mean +/- std.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional seed for a single run.'
    )
    
    args = parser.parse_args()
    
    # Override config if needed
    if args.mode == 'ablation':
        run_ablation_study(Config)
    elif args.runs > 1:
        encoder_path = f"{Config.OUTPUT_DIR}/encoder_pretrained.pth"
        run_multiple_stage2(
            Config,
            mode=args.mode,
            runs=args.runs,
            encoder_weights_path=encoder_path,
            use_class_weights=not args.no_class_weights,
        )
    else:
        encoder_path = f"{Config.OUTPUT_DIR}/encoder_pretrained.pth"
        train_wagner_stage2(
            Config,
            mode=args.mode,
            encoder_weights_path=encoder_path,
            use_class_weights=not args.no_class_weights,
            seed=args.seed,
            evaluate_test=True,
        )
