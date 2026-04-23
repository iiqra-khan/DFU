import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os
from tqdm import tqdm

from dataset import DPMDataset, get_transforms
from config import Config

try:
    import wandb
except Exception:
    wandb = None


def _is_resnet_backbone(backbone_name):
    return str(backbone_name).lower().startswith('resnet')


def _apply_generic_freeze_for_non_resnet(model):
    """Fallback freeze policy for timm backbones that don't use ResNet layer names."""
    # Undo earlier name-based freezing if the backbone is not ResNet-like.
    for _, param in model.named_parameters():
        param.requires_grad = True

    # Freeze feature extractor; keep classification heads trainable.
    head_tokens = ('head', 'classifier', 'fc')
    for name, param in model.named_parameters():
        if not any(token in name for token in head_tokens):
            param.requires_grad = False


def _maybe_init_wandb(config):
    if not getattr(config, 'USE_WANDB', False):
        return None
    if wandb is None:
        print("⚠️ wandb is not installed; continuing without experiment tracking.")
        return None

    run = wandb.init(
        project=getattr(config, 'WANDB_PROJECT', 'dfu-pipeline'),
        entity=getattr(config, 'WANDB_ENTITY', None),
        name=getattr(config, 'WANDB_RUN_NAME', None),
        config={
            'backbone': config.BACKBONE,
            'batch_size': config.BATCH_SIZE,
            'epochs_stage2': config.EPOCHS_STAGE2,
            'lr_stage2': config.LR_STAGE2,
        },
        reinit=True,
    )
    return run

def train_wagner_grading(config):
    """Stage 2: Wagner grading with pretrained encoder"""

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load base model
    model = timm.create_model(config.BACKBONE, pretrained=False, num_classes=4)

    # Load encoder weights from Stage 1
    encoder_path = f"{config.OUTPUT_DIR}/encoder_stage1.pth"
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found at {encoder_path}. "
                                "Run Stage 1 training first.")

    pretrained_dict = torch.load(encoder_path, map_location=config.DEVICE)
    model_dict = model.state_dict()

    # Filter and load matching weights (only encoder parts)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(config.DEVICE)

    # Freeze early layers (optional - prevents overfitting on small datasets)
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # For non-ResNet backbones, apply a generic head-only fine-tuning policy.
    if not _is_resnet_backbone(config.BACKBONE):
        _apply_generic_freeze_for_non_resnet(model)

    # Data
    train_dataset = DPMDataset(config.DPM_PATH, split='train',
                               transform=get_transforms('train'))
    val_dataset = DPMDataset(config.DPM_PATH, split='valid',
                             transform=get_transforms('val'))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.LR_STAGE2)

    wandb_run = _maybe_init_wandb(config)

    best_f1 = 0.0

    for epoch in range(config.EPOCHS_STAGE2):
        # Train
        model.train()
        epoch_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(config.DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{config.EPOCHS_STAGE2}: "
              f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        if wandb_run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_acc': acc,
                'val_f1': f1,
            })

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),
                       f"{config.OUTPUT_DIR}/best_wagner_model.pth")
            print(f"  -> Saved best model (F1: {best_f1:.4f})")

            if wandb_run is not None:
                wandb.run.summary['best_val_f1'] = best_f1

    print(f"✅ Stage 2 complete. Best Val F1: {best_f1:.4f}")

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    train_wagner_grading(Config)