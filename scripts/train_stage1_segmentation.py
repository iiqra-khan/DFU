import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import os
import math

from dataset import FUSegDataset, get_transforms
from config import Config


class SegFormerSeg(nn.Module):
    """SegFormer-B2 wrapper for binary segmentation"""
    def __init__(self, model_id='nvidia/segformer-b2-finetuned-ade-512-512', num_labels=1):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        out = self.model(pixel_values=x).logits
        # Resize output to match input spatial dimensions
        return nn.functional.interpolate(
            out, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss: 0.3*BCE + 0.7*Dice"""
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        # BCE component
        bce_loss = self.bce(pred, target)
        
        # Dice component
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _ensure_channel_dim(mask_tensor):
    if mask_tensor.dim() == 3:
        return mask_tensor.unsqueeze(1)
    return mask_tensor


def _save_stage1_artifacts(config, history, best_val_dice):
    if not getattr(config, 'SAVE_METRICS_JSON', True):
        return

    history_path = os.path.join(config.OUTPUT_DIR, 'stage1_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_val_dice': None if best_val_dice == -math.inf else float(best_val_dice),
            'history': history,
        }, f, indent=2)

    if not getattr(config, 'SAVE_PLOTS', True):
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if history['val_epoch']:
        ax2 = ax1.twinx()
        ax2.plot(history['val_epoch'], history['val_dice'], label='Val Dice', color='tab:green', marker='o')
        ax2.set_ylabel('Val Dice', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
    else:
        ax2 = None

    lines, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='best')
    ax1.set_title('Stage 1 Training Curves')
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, 'stage1_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_segmentation(config):
    """Stage 1: SegFormer-B2 binary segmentation on FUSeg2021"""

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Model
    model = SegFormerSeg(
        model_id=config.SEGFORMER_MODEL,
        num_labels=config.SEGFORMER_NUM_LABELS
    ).to(config.DEVICE)

    # Enable gradient checkpointing if configured (saves memory at cost of compute)
    if getattr(config, 'GRADIENT_CHECKPOINTING', False):
        try:
            model.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on SegFormer model")
        except Exception:
            pass

    # AMP scaler (optional)
    scaler = GradScaler() if getattr(config, 'USE_AMP', False) else None
    accum_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)

    # Data
    train_dataset = FUSegDataset(
        f"{config.FUSEG_PATH}/train/images",
        f"{config.FUSEG_PATH}/train/labels",
        transform=get_transforms('train')
    )
    val_dataset = FUSegDataset(
        f"{config.FUSEG_PATH}/validation/images",
        f"{config.FUSEG_PATH}/validation/labels",
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

    # Loss & optimizer
    criterion = DiceBCELoss(
        bce_weight=config.LOSS_WEIGHTS['bce'],
        dice_weight=config.LOSS_WEIGHTS['dice']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LR_STAGE1,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler: CosineAnnealingLR for 20 epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS_STAGE1,
        eta_min=1e-6
    )

    best_val_dice = -math.inf
    epochs_without_improvement = 0
    history = {
        'epoch': [],
        'train_loss': [],
        'val_epoch': [],
        'val_dice': [],
        'lr': [],
    }

    print(f"🚀 Starting Stage 1 SegFormer-B2 Training")
    print(f"   Model: {config.SEGFORMER_MODEL}")
    print(f"   Loss: 0.3*BCE + 0.7*Dice")
    print(f"   Optimizer: AdamW (lr={config.LR_STAGE1}, weight_decay={config.WEIGHT_DECAY})")
    print(f"   Scheduler: CosineAnnealingLR for {config.EPOCHS_STAGE1} epochs")
    print(f"   Device: {config.DEVICE}")

    # Training loop
    for epoch in range(config.EPOCHS_STAGE1):
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        for step_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_STAGE1} [Train]")):
            images = images.to(config.DEVICE)
            masks = _ensure_channel_dim(masks.to(config.DEVICE).float())

            with autocast(enabled=getattr(config, 'USE_AMP', False)):
                outputs = model(images)
                loss = criterion(outputs, masks) / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer when we've accumulated enough micro-batches
            if (step_idx + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Maintain epoch loss as sum of original (pre-accum) losses
            epoch_loss += loss.item() * accum_steps

        # If number of steps was not divisible by accum_steps, do a final step
        if (step_idx + 1) % accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['lr'].append(float(scheduler.get_last_lr()[0]))

        should_validate = (
            (epoch + 1) % getattr(config, 'VALIDATE_EVERY_N_EPOCHS', 1) == 0
            or (epoch + 1) == config.EPOCHS_STAGE1
        )

        avg_val_dice = None
        if should_validate:
            # Validation
            model.eval()
            val_dice = 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_STAGE1} [Val]"):
                    images = images.to(config.DEVICE)
                    masks = _ensure_channel_dim(masks.to(config.DEVICE).float())
                    masks = (masks > 0.5).float()

                    with autocast(enabled=getattr(config, 'USE_AMP', False)):
                        outputs = model(images)
                    # Dice score for binary segmentation
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    intersection = (preds * masks).sum(dim=(1, 2, 3))
                    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                    dice = (2 * intersection) / (union + 1e-6)
                    val_dice += dice.mean().item()

            avg_val_dice = val_dice / len(val_loader)
            history['val_epoch'].append(epoch + 1)
            history['val_dice'].append(float(avg_val_dice))

            print(f"Epoch {epoch+1}/{config.EPOCHS_STAGE1}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Save best model and track improvement for early stopping
            if avg_val_dice > (best_val_dice + config.EARLY_STOPPING_MIN_DELTA):
                best_val_dice = avg_val_dice
                epochs_without_improvement = 0
                
                # Save full checkpoint
                torch.save(
                    model.state_dict(),
                    f"{config.OUTPUT_DIR}/stage1_segformer_best.pth"
                )
                
                # Save encoder separately for Stage 2 transfer
                torch.save(
                    model.model.segformer.encoder.state_dict(),
                    f"{config.OUTPUT_DIR}/encoder_pretrained.pth"
                )
                
                print(f"  ✅ Saved best model (Dice: {best_val_dice:.4f})")
            else:
                epochs_without_improvement += 1
        else:
            print(f"Epoch {epoch+1}/{config.EPOCHS_STAGE1}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val skipped, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Step scheduler
        scheduler.step()

        if (should_validate and config.USE_EARLY_STOPPING and
                epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE_STAGE1):
            print(
                f"  ⏹️  Early stopping triggered "
                f"(no Dice improvement > {config.EARLY_STOPPING_MIN_DELTA} for "
                f"{config.EARLY_STOPPING_PATIENCE_STAGE1} epochs)"
            )
            break

    best_val_msg = f"{best_val_dice:.4f}" if best_val_dice != -math.inf else "not evaluated"
    _save_stage1_artifacts(config, history, best_val_dice)
    print(f"✅ Stage 1 complete. Best Val Dice: {best_val_msg}")
    print(f"   📦 Checkpoint: {config.OUTPUT_DIR}/stage1_segformer_best.pth")
    print(f"   🔌 Encoder: {config.OUTPUT_DIR}/encoder_pretrained.pth")


if __name__ == "__main__":
    train_segmentation(Config)