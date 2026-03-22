import torch
import numpy as np
import matplotlib.pyplot as plt


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Dice Score for binary segmentation.

    Args:
        pred: prediction tensor (N, 1, H, W) - raw logits or probabilities
        target: ground truth tensor (N, 1, H, W) - binary {0, 1}
        threshold: threshold for binarizing prediction
        smooth: smoothing factor to avoid division by zero

    Returns:
        mean dice score (scalar)
    """
    # Always apply sigmoid: model outputs raw logits
    pred = torch.sigmoid(pred)

    # Binarize prediction
    pred_binary = (pred > threshold).float()

    # Flatten spatial dimensions
    pred_flat = pred_binary.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    # Compute per-sample dice
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

    return dice.mean().item()


def dice_loss(pred, target, smooth=1e-6):
    """
    Differentiable Dice Loss for training.

    Args:
        pred: prediction logits (N, 1, H, W)
        target: ground truth (N, 1, H, W) binary
    Returns:
        1 - soft dice score (scalar, differentiable)
    """
    pred_prob = torch.sigmoid(pred)

    pred_flat = pred_prob.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

    return (1 - dice).mean()


def visualize_predictions(images, masks, preds, num_samples=4, save_path=None):
    """
    Visualize image, ground truth mask, and predicted mask side by side.

    Args:
        images: (N, 3, H, W) tensor (normalized)
        masks: (N, 1, H, W) tensor (binary ground truth)
        preds: (N, 1, H, W) tensor (model output logits)
        num_samples: number of samples to show
        save_path: if provided, save figure to this path
    """
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    images_denorm = images * std + mean
    images_denorm = images_denorm.clamp(0, 1)

    pred_binary = (torch.sigmoid(preds) > 0.5).float()

    n = min(num_samples, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.unsqueeze(0) if hasattr(axes, 'unsqueeze') else [axes]

    for i in range(n):
        # Image
        axes[i][0].imshow(images_denorm[i].permute(1, 2, 0).numpy())
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        # Ground truth
        axes[i][1].imshow(masks[i, 0].numpy(), cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        # Prediction
        axes[i][2].imshow(pred_binary[i, 0].numpy(), cmap="gray")
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.close()
