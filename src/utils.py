import torch


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


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary segmentation (論文使用).

    Args:
        pred: prediction logits (N, 1, H, W)
        target: ground truth (N, 1, H, W) binary
        alpha: weighting factor for class balance
        gamma: focusing parameter
    Returns:
        focal loss (scalar, differentiable)
    """
    pred_prob = torch.sigmoid(pred)
    # binary cross entropy per pixel
    bce = -target * torch.log(pred_prob + 1e-8) - (1 - target) * torch.log(1 - pred_prob + 1e-8)
    # focal weight
    pt = target * pred_prob + (1 - target) * (1 - pred_prob)
    focal_weight = alpha * (1 - pt) ** gamma
    loss = focal_weight * bce
    return loss.mean()


