from captum.attr import IntegratedGradients
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os

from train_stage2_wagner import SegFormerWagnerClassifier

try:
    from IPython.display import display
except Exception:
    display = None


def load_stage2_model(config, model_path=None):
    """Load the Stage 2 classifier checkpoint for XAI."""
    if model_path is None:
        candidates = [
            os.path.join(config.OUTPUT_DIR, 'best_wagner_model_two_stage_run0.pth'),
            os.path.join(config.OUTPUT_DIR, 'best_wagner_model_two_stage.pth'),
            os.path.join(config.OUTPUT_DIR, 'best_wagner_model.pth'),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = SegFormerWagnerClassifier(num_classes=getattr(config, 'SEGFORMER_NUM_CLASSES_STAGE2', 4))
    state_dict = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE)
    model.eval()
    return model


def _denormalize_image(tensor):
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)


def _normalize_map(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def _last_hidden_state_as_feature(hidden_state):
    """Normalize SegFormer hidden state shape to BCHW for CAM math."""
    if hidden_state.ndim == 4:
        return hidden_state

    if hidden_state.ndim == 3:
        batch_size, seq_len, channels = hidden_state.shape
        side = int(np.sqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(f"Cannot reshape hidden state with sequence length {seq_len} to a square map.")
        return hidden_state.permute(0, 2, 1).reshape(batch_size, channels, side, side)

    raise ValueError(f"Unexpected hidden state shape: {tuple(hidden_state.shape)}")


def compute_gradcampp(model, image, class_idx=None):
    """Compute GradCAM++ from the final SegFormer encoder feature map."""
    model.zero_grad(set_to_none=True)

    encoder_output = model.backbone(pixel_values=image, output_hidden_states=True)
    activation = _last_hidden_state_as_feature(encoder_output.hidden_states[-1])
    activation.retain_grad()

    pooled = model.pool(activation).squeeze(-1).squeeze(-1)
    logits = model.classifier(pooled)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())

    score = logits[:, class_idx].sum()
    gradients = torch.autograd.grad(
        score,
        activation,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_2 = gradients.pow(2)
    grad_3 = gradients.pow(3)
    spatial_sum = (activation * grad_3).sum(dim=(2, 3), keepdim=True)
    alpha = grad_2 / (2.0 * grad_2 + spatial_sum + 1e-8)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = _normalize_map(cam)

    probs = torch.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy()
    return cam, class_idx, probs


def generate_gradcampp_explanations(
    model,
    test_loader,
    config,
    num_samples=12,
    output_subdir='gradcampp',
    use_predicted_class=True,
    display_inline=True,
):
    """Generate GradCAM++ overlays for Stage 2 classifier predictions."""
    model.eval()
    xai_dir = os.path.join(config.OUTPUT_DIR, output_subdir)
    os.makedirs(xai_dir, exist_ok=True)

    samples_processed = 0
    for images, labels in test_loader:
        if samples_processed >= num_samples:
            break

        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        for i in range(images.size(0)):
            if samples_processed >= num_samples:
                break

            image = images[i:i + 1].detach().clone().requires_grad_(True)
            true_label = int(labels[i].item())
            target_class = None if use_predicted_class else true_label
            cam, pred_class, probs = compute_gradcampp(model, image, class_idx=target_class)
            original_img = _denormalize_image(image.squeeze(0))

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title(f"Input\nTrue Grade: {true_label + 1}")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title(f"GradCAM++\nTarget Grade: {pred_class + 1}")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1, 3, 3)
            plt.imshow(original_img)
            plt.imshow(cam, cmap='jet', alpha=0.45)
            plt.title(f"Overlay\nPred: {pred_class + 1} ({probs[pred_class]:.2f})")
            plt.axis('off')

            plt.tight_layout()
            output_path = os.path.join(xai_dir, f'gradcampp_sample_{samples_processed + 1}.png')
            plt.savefig(output_path, dpi=180, bbox_inches='tight')

            if display_inline and display is not None:
                display(Image.open(output_path))

            plt.close()
            samples_processed += 1

    print(f"✅ Generated {samples_processed} GradCAM++ explanations in {xai_dir}")


def _safe_integrated_gradients(
    ig,
    img,
    label,
    device,
    n_steps=50,
    internal_batch_size=4,
    allow_cpu_fallback=True,
):
    """Compute IG with CUDA OOM recovery by reducing work and optional CPU fallback."""
    try:
        return ig.attribute(
            img,
            target=label,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"⚠️ CUDA OOM during IG (n_steps={n_steps}, internal_batch_size={internal_batch_size}). "
            "Retrying with lighter settings..."
        )
        reduced_steps = max(10, n_steps // 2)
        try:
            return ig.attribute(
                img,
                target=label,
                n_steps=reduced_steps,
                internal_batch_size=1,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not allow_cpu_fallback:
                raise

            print("⚠️ Still OOM on CUDA. Falling back to CPU for this sample attribution...")
            model = ig.forward_func
            original_device = next(model.parameters()).device
            model_cpu = model.to('cpu')
            model_cpu.eval()
            ig_cpu = IntegratedGradients(model_cpu)
            cpu_attr = ig_cpu.attribute(
                img.detach().to('cpu'),
                target=label,
                n_steps=reduced_steps,
                internal_batch_size=1,
            )
            model_cpu.to(original_device)
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return cpu_attr.to(original_device)

def generate_xai_explanations(
    model,
    test_loader,
    config,
    num_samples=5,
    display_inline=True,
    n_steps=24,
    internal_batch_size=2,
    allow_cpu_fallback=True,
):
    """Generate Integrated Gradients explanations for model predictions"""

    model.eval()
    ig = IntegratedGradients(model)

    # Create output directory for XAI results
    xai_dir = os.path.join(config.OUTPUT_DIR, 'xai')
    os.makedirs(xai_dir, exist_ok=True)

    samples_processed = 0

    for images, labels in test_loader:
        if samples_processed >= num_samples:
            break

        images = images.to(config.DEVICE).requires_grad_()
        labels = labels.to(config.DEVICE)

        # Generate attributions for each image in batch
        for i in range(images.size(0)):
            if samples_processed >= num_samples:
                break

            img = images[i:i+1]  # Keep batch dimension
            label = int(labels[i].item())

            # Generate attributions with memory-safe fallback.
            attributions = _safe_integrated_gradients(
                ig,
                img,
                label,
                config.DEVICE,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                allow_cpu_fallback=allow_cpu_fallback,
            )

            # Process for visualization
            original_img = img.squeeze(0).cpu().detach().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))  # CHW -> HWC

            # Denormalize image (approximate)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_img = original_img * std + mean
            original_img = np.clip(original_img, 0, 1)

            attr_map = attributions.squeeze().cpu().detach().numpy()
            attr_map = np.transpose(attr_map, (1, 2, 0))  # CHW -> HWC

            # Sum across channels for attribution map
            attr_map = np.sum(np.abs(attr_map), axis=2)

            # Normalize attribution map for visualization
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

            # Create visualization
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title(f"Input Image\nTrue Grade: {label + 1}")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(attr_map, cmap='hot')
            plt.title("Integrated Gradients")
            plt.axis('off')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.imshow(original_img)
            plt.imshow(attr_map, cmap='hot', alpha=0.5)
            plt.title("Overlay")
            plt.axis('off')

            plt.tight_layout()
            output_path = os.path.join(xai_dir, f'xai_sample_{samples_processed+1}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if display_inline and display is not None:
                display(Image.open(output_path))

            plt.close()

            samples_processed += 1

    print(f"✅ Generated {samples_processed} XAI explanations in {xai_dir}")


def run_stage2_xai(
    config,
    split='valid',
    num_samples=5,
    batch_size=1,
    model_path=None,
    display_inline=True,
    n_steps=24,
    internal_batch_size=2,
    allow_cpu_fallback=True,
):
    """One-call helper to load Stage 2 and save XAI outputs."""
    from torch.utils.data import DataLoader
    from dataset import DPMDataset, get_transforms

    model = load_stage2_model(config, model_path=model_path)
    dataset = DPMDataset(config.DPM_PATH, split=split, transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=getattr(config, 'NUM_WORKERS', 0))

    generate_xai_explanations(
        model,
        loader,
        config,
        num_samples=num_samples,
        display_inline=display_inline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def run_stage2_gradcampp(
    config,
    split='test',
    num_samples=12,
    batch_size=1,
    model_path=None,
    display_inline=True,
    use_predicted_class=True,
):
    """One-call helper to load Stage 2 and save GradCAM++ outputs."""
    from torch.utils.data import DataLoader
    from dataset import DPMDataset, get_transforms

    model = load_stage2_model(config, model_path=model_path)
    dataset = DPMDataset(config.DPM_PATH, split=split, transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=getattr(config, 'NUM_WORKERS', 0))

    generate_gradcampp_explanations(
        model,
        loader,
        config,
        num_samples=num_samples,
        output_subdir=f'gradcampp_{split}',
        use_predicted_class=use_predicted_class,
        display_inline=display_inline,
    )

def get_predictions_and_attributions(model, test_loader, config):
    """Get model predictions and attributions for analysis"""

    model.eval()
    ig = IntegratedGradients(model)

    all_preds = []
    all_labels = []
    all_attrs = []  # Store attributions for further analysis

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute attributions for a subset (can be memory intensive)
    model.eval()
    attributions_list = []

    for i, (images, labels) in enumerate(test_loader):
        if i >= 10:  # Limit to first 10 batches for attribution analysis
            break

        images = images.to(config.DEVICE).requires_grad_()
        labels = labels.to(config.DEVICE)

        attributions = ig.attribute(images, target=labels, n_steps=24, internal_batch_size=1)
        attributions_list.append(attributions.cpu())

    if attributions_list:
        all_attrs = torch.cat(attributions_list, dim=0)

    return np.array(all_preds), np.array(all_labels), all_attrs

if __name__ == "__main__":
    from config import Config

    parser = argparse.ArgumentParser(description="Generate Stage 2 XAI visualizations")
    parser.add_argument('--method', choices=['gradcampp', 'ig'], default='gradcampp')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'val', 'test'])
    parser.add_argument('--num-samples', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--target', choices=['predicted', 'true'], default='predicted')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    if args.method == 'gradcampp':
        run_stage2_gradcampp(
            Config,
            split=args.split,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            model_path=args.model_path,
            display_inline=not args.no_display,
            use_predicted_class=args.target == 'predicted',
        )
    else:
        run_stage2_xai(
            Config,
            split=args.split,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            model_path=args.model_path,
            display_inline=not args.no_display,
        )
