import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

from oxford_pet import get_dataloaders
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_score


def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    if args.model == "unet":
        args.test_file = "test_unet.txt"
        args.model_path = "saved_models/unet_best.pth"
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == "resnet34_unet":
        args.test_file = "test_res_unet.txt"
        args.model_path = "saved_models/resnet34_unet_best.pth"
        model = ResNet34UNet(out_channels=1).to(device)
    else:
        raise ValueError(f"wrong model: {args.model}")

    data_root = os.path.abspath(args.data_root)
    _, val_loader, _ = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )


    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Overlap-tile
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
        _out_size = model(_dummy).shape[-1]
    border_pad = (args.img_size - _out_size) // 2
    if border_pad > 0:
        print(f"Overlap-tile")


    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            if border_pad > 0:
                images = F.pad(images, (border_pad,)*4, mode='reflect')
                outputs = model(images)
                crop = (outputs.shape[-1] - args.img_size) // 2
                outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
            else:
                outputs = model(images)

            total_dice += dice_score(outputs, masks)
            num_batches += 1

    avg_dice = total_dice / num_batches
    print(f"Validation Dice Score: {avg_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet on validation set")
    parser.add_argument("--data_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(args)
