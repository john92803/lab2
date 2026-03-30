import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import csv
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from oxford_pet import get_dataloaders
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


def mask2rle(mask):
    flat = mask.flatten(order='F')
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1 
    lengths = np.where(diff == -1)[0] - (starts - 1)

    return " ".join(str(x) for pair in zip(starts, lengths) for x in pair)


def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    if args.model == "unet":
        args.test_file = "test_unet.txt"
        args.model_path = "saved_models/unet_best.pth"
        args.output_csv = "submission_unet.csv"
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == "resnet34_unet":
        args.test_file = "test_res_unet.txt"
        args.model_path = "saved_models/resnet34_unet_best.pth"
        args.output_csv = "submission_resnet34_unet.csv"
        model = ResNet34UNet(out_channels=1).to(device)
    else:
        raise ValueError(f"wrong model: {args.model}")

    data_root = os.path.abspath(args.data_root)
    _, _, test_loader = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
        dummy_out = model(dummy)
        out_size = dummy_out.shape[-1]

    border_pad = (args.img_size - out_size) // 2
    if border_pad > 0:
        padded_size = args.img_size + 2 * border_pad
        padded_out  = model(torch.zeros(1, 3, padded_size, padded_size).to(device)).shape[-1]
        print(f"Overlap-tile")

    results = []

    with torch.no_grad():
        for images, names, orig_ws, orig_hs in test_loader:
            images = images.to(device)

            if border_pad > 0:
                # Overlap-tile：mirror pad 四邊，讓邊界有完整 context
                images_padded = F.pad(images,
                    (border_pad, border_pad, border_pad, border_pad),
                    mode='reflect')
                outputs = model(images_padded)

                crop = (outputs.shape[-1] - args.img_size) // 2
                outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
            else:
                outputs = model(images)                             

            preds = (torch.sigmoid(outputs) > 0.5).float()         

            for i in range(images.size(0)):
                mask = preds[i, 0].cpu().numpy()                  
                orig_w = int(orig_ws[i])
                orig_h = int(orig_hs[i])

                if mask.shape != (orig_h, orig_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                    mask = np.array(mask_img).astype(np.float32)

                rle = mask2rle(mask)
                results.append((names[i], rle))

    # 存成 CSV
    output_path = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        for image_id, rle in results:
            writer.writerow([image_id, rle])

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for binary segmentation")
    parser.add_argument("--data_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
