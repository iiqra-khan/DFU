import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
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


def train_segmentation(config):
    """Stage 1: SegFormer-B2 binary segmentation on FUSeg2021"""

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Model
    model = SegFormerSeg(
        model_id=config.SEGFORMER_MODEL,
        num_labels=config.SEGFORMER_NUM_LABELS
    ).to(config.DEVICE)

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

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_STAGE1} [Train]"):
            images = images.to(config.DEVICE)
            masks = _ensure_channel_dim(masks.to(config.DEVICE).float())

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_STAGE1} [Val]"):
                images = images.to(config.DEVICE)
                masks = _ensure_channel_dim(masks.to(config.DEVICE).float())
                masks = (masks > 0.5).float()

                outputs = model(images)
                # Dice score for binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dice = (2 * intersection) / (union + 1e-6)
                val_dice += dice.mean().item()

        avg_val_dice = val_dice / len(val_loader)

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

        # Step scheduler
        scheduler.step()

        if (config.USE_EARLY_STOPPING and
                epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE_STAGE1):
            print(
                f"  ⏹️  Early stopping triggered "
                f"(no Dice improvement > {config.EARLY_STOPPING_MIN_DELTA} for "
                f"{config.EARLY_STOPPING_PATIENCE_STAGE1} epochs)"
            )
            break

    print(f"✅ Stage 1 complete. Best Val Dice: {best_val_dice:.4f}")
    print(f"   📦 Checkpoint: {config.OUTPUT_DIR}/stage1_segformer_best.pth")
    print(f"   🔌 Encoder: {config.OUTPUT_DIR}/encoder_pretrained.pth")


if __name__ == "__main__":
    train_segmentation(Config)