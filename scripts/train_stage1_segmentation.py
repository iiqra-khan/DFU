import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math

from dataset import FUSegDataset, get_transforms
from config import Config


def _ensure_channel_dim(mask_tensor):
    if mask_tensor.dim() == 3:
        return mask_tensor.unsqueeze(1)
    return mask_tensor

def train_segmentation(config):
    """Stage 1: Train segmentation on FUSeg"""

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Model
    model = smp.Unet(
        encoder_name=config.BACKBONE,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1
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

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Loss & optimizer
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_STAGE1)

    best_val_dice = -math.inf
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(config.EPOCHS_STAGE1):
        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = images.to(config.DEVICE)
            masks = _ensure_channel_dim(masks.to(config.DEVICE).float())

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(config.DEVICE)
                masks = _ensure_channel_dim(masks.to(config.DEVICE).float())

                # Keep validation strictly binary in case any augmentation or loader
                # preserves grayscale mask values.
                masks = (masks > 0.5).float()

                outputs = model(images)
                # Dice score for binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dice = (2 * intersection) / (union + 1e-6)
                val_dice += dice.mean().item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch {epoch+1}/{config.EPOCHS_STAGE1}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # Save best model and track improvement for early stopping.
        if avg_val_dice > (best_val_dice + config.EARLY_STOPPING_MIN_DELTA):
            best_val_dice = avg_val_dice
            epochs_without_improvement = 0
            torch.save(model.encoder.state_dict(),
                       f"{config.OUTPUT_DIR}/encoder_stage1.pth")
            print(f"  -> Saved best encoder (Dice: {best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1

        if (config.USE_EARLY_STOPPING and
                epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE_STAGE1):
            print(
                "  -> Early stopping triggered "
                f"(no Dice improvement > {config.EARLY_STOPPING_MIN_DELTA} for "
                f"{config.EARLY_STOPPING_PATIENCE_STAGE1} epochs)"
            )
            break

    print(f"✅ Stage 1 complete. Best Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    train_segmentation(Config)