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


def mask_to_rle(mask):
    """
    將二值 mask 轉成 RLE 字串。

    Args:
        mask: numpy array, shape (H, W), 值為 0 或 1

    Returns:
        RLE 字串, 例如 "3 5 10 2"

    步驟：
    1. 把 2D mask 拉平成 1D（用 Fortran order，即 column-major）
    2. 找出前景 (值=1) 的起始位置和連續長度
    3. 組成 "start1 length1 start2 length2 ..." 的字串

    提示：
    - flat = mask.flatten(order='F')    # column-major 拉平
    - 在 flat 前後各補一個 0，方便找起始和結束
    - 用 np.where(np.diff(...) != 0) 找出變化點
    - RLE 的位置是 1-indexed（從 1 開始算，不是 0）
    """
    # TODO: 實作 RLE 編碼
    flat = mask.flatten(order='F')
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1   # 轉成 1-indexed
    lengths = np.where(diff == -1)[0] - (starts - 1)

    return " ".join(str(x) for pair in zip(starts, lengths) for x in pair)


def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    # ===== 根據模型自動設定預設路徑並載入模型 =====
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
        raise ValueError(f"Unknown model: {args.model}")

    # ===== 資料 =====
    data_root = os.path.abspath(args.data_root)
    _, _, test_loader = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ===== Overlap-tile strategy（UNet 論文 Section 2 / Fig. 2）=====
    # UNet with padding=0：每層 valid conv 縮小 spatial，造成輸出 < 輸入
    # 論文做法："the missing context is extrapolated by mirroring the input image"
    # → 在輸入四邊做 mirror（reflect）padding，使邊界像素也有足夠 context
    # → 輸出涵蓋完整原始圖，再 center crop 取回 img_size × img_size
    #
    # 對應 Fig. 2：
    #   黃色 tile = mirror-padded 輸入（img_size + 2×border_pad）
    #   藍色 tile = 有效輸出區域（img_size），center crop 自輸出取得
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
        dummy_out = model(dummy)
        out_size = dummy_out.shape[-1]

    # border_pad = 每邊需補的像素數，使 mirror-padded 輸出能覆蓋完整 img_size
    border_pad = (args.img_size - out_size) // 2
    if border_pad > 0:
        padded_size = args.img_size + 2 * border_pad
        padded_out  = model(torch.zeros(1, 3, padded_size, padded_size).to(device)).shape[-1]
        print(f"Overlap-tile (UNet padding=0):")
        print(f"  輸入 tile : {padded_size}×{padded_size}（mirror pad {border_pad}px）")
        print(f"  輸出 tile : {padded_out}×{padded_out} → center crop → {args.img_size}×{args.img_size}")
    else:
        print(f"Model output={out_size}×{out_size}，與輸入同尺寸，不需 overlap-tile")

    # ===== 推論 =====
    results = []

    with torch.no_grad():
        for images, names, orig_ws, orig_hs in test_loader:
            images = images.to(device)

            if border_pad > 0:
                # Overlap-tile：mirror pad 四邊，讓邊界像素有完整 context（論文 Fig. 2 黃色 tile）
                images_padded = F.pad(images,
                    (border_pad, border_pad, border_pad, border_pad),
                    mode='reflect')
                outputs = model(images_padded)
                # center crop 取回 img_size × img_size（論文 Fig. 2 藍色 tile）
                crop = (outputs.shape[-1] - args.img_size) // 2
                outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
            else:
                outputs = model(images)                             # ResNet34UNet: 輸出已是 img_size

            preds = (torch.sigmoid(outputs) > 0.5).float()          # [B, 1, img_size, img_size]

            for i in range(images.size(0)):
                mask = preds[i, 0].cpu().numpy()                    # (img_size, img_size)
                orig_w = int(orig_ws[i])
                orig_h = int(orig_hs[i])

                # 還原回原始圖片尺寸
                if mask.shape != (orig_h, orig_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                    mask = np.array(mask_img).astype(np.float32)

                rle = mask_to_rle(mask)
                results.append((names[i], rle))

    # ===== 存成 CSV =====
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
