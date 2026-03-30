import torch


def dice_score(pred, target, threshold=0.5, epsilon=1e-6):

    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()

    pred_flat = pred_binary.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice.mean().item()


def dice_loss(pred, target, epsilon=1e-6):

    pred_prob = torch.sigmoid(pred)

    pred_flat = pred_prob.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return (1 - dice).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0, epsilon=1e-8):

    pred_prob = torch.sigmoid(pred)

    bce = -target * torch.log(pred_prob + epsilon) - (1 - target) * torch.log(1 - pred_prob + epsilon)

    p = target * pred_prob + (1 - target) * (1 - pred_prob)
    focal_weight = alpha * (1 - p) ** gamma
    
    loss = focal_weight * bce
    return loss.mean()


