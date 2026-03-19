import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(__file__))

from oxford_pet import get_dataloaders
from models.unet import UNet
from utils import dice_score


def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    # ===== 資料 =====
    data_root = os.path.abspath(args.data_root)
    _, val_loader, _ = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ===== 載入模型 =====
    model = UNet(in_channels=3, out_channels=1).to(device)

    # TODO: 用 torch.load 讀取權重，再用 model.load_state_dict 載入
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # ===== 評估 =====
    model.eval()
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += dice_score(outputs, masks)
            num_batches += 1

    avg_dice = total_dice / num_batches
    print(f"Validation Dice Score: {avg_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet on validation set")
    parser.add_argument("--data_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--test_file", type=str, default="test_unet.txt")
    parser.add_argument("--model_path", type=str, default="saved_models/unet_best.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(args)
