from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os

def generate_xai_explanations(model, test_loader, config, num_samples=5):
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
            label = labels[i:i+1]

            # Generate attributions
            attributions = ig.attribute(img, target=label, n_steps=50)

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
            plt.title(f"Input Image\nTrue Grade: {label.item()+1}")
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
            plt.savefig(os.path.join(xai_dir, f'xai_sample_{samples_processed+1}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

            samples_processed += 1

    print(f"✅ Generated {samples_processed} XAI explanations in {xai_dir}")

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

        attributions = ig.attribute(images, target=labels, n_steps=50)
        attributions_list.append(attributions.cpu())

    if attributions_list:
        all_attrs = torch.cat(attributions_list, dim=0)

    return np.array(all_preds), np.array(all_labels), all_attrs

if __name__ == "__main__":
    # This would be called from your Kaggle notebook after loading test data
    print("XAI analysis module ready. Import and use:")
    print("  from xai_analysis import generate_xai_explanations")
    print("  generate_xai_explanations(model, test_loader, config)")