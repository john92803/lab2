import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from oxford_pet import get_dataloaders
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_score, dice_loss, focal_loss

def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    # ===== 根據模型自動設定預設路徑 =====
    if args.model == "unet":
        args.test_file = "test_unet.txt"
        args.save_path = "saved_models/unet_best.pth"
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == "resnet34_unet":
        args.test_file = "test_res_unet.txt"
        args.save_path = "saved_models/resnet34_unet_best.pth"
        model = ResNet34UNet(out_channels=1).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ===== 資料 =====
    data_root = os.path.abspath(args.data_root)
    train_loader, val_loader, _ = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===== Overlap-tile：計算 border_pad（與 inference.py 一致）=====
    # UNet padding=0 輸出 < 輸入 → 訓練/val 也套用 overlap-tile，使 loss 涵蓋完整圖片
    # ResNet34UNet 輸出 = 輸入 → border_pad=0，不做任何改動
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
        _out_size = model(_dummy).shape[-1]
    border_pad = (args.img_size - _out_size) // 2
    if border_pad > 0:
        print(f"Overlap-tile strategy 啟用（UNet padding=0）: border_pad={border_pad}px")
        print(f"  訓練/val 輸入: {args.img_size} → {args.img_size + 2*border_pad}（reflect pad）→ 輸出 crop 回 {args.img_size}")
    else:
        print(f"Overlap-tile 不需要（輸出與輸入同尺寸）")

    # 根據模型選擇 loss 和 optimizer（論文: ResNet34_UNet 用 Focal+Dice, SGD）
    if args.model == "resnet34_unet":
        use_focal = True
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        use_focal = False
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()  # UNet 用; ResNet34_UNet 改用 focal_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # ===== 訓練 =====
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        # --- 訓練 ---
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_batches = 0

        # 使用 tqdm 包裝 train_loader
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Overlap-tile：reflect pad → model → center crop
            # border_pad=0 時等同直接呼叫 model(images)
            if border_pad > 0:
                images = F.pad(images, (border_pad,)*4, mode='reflect')
                outputs = model(images)
                crop = (outputs.shape[-1] - args.img_size) // 2
                outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
            else:
                outputs = model(images)
            # outputs 現在與 masks 尺寸一致（img_size × img_size），不需 center crop mask

            if use_focal:
                loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)
            else:
                loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # 計算當前 batch 的數值
            current_loss = loss.item()
            current_dice = dice_score(outputs, masks)
            
            train_loss += current_loss
            train_dice += current_dice
            num_batches += 1
            
            # 更新進度條後方的即時資訊
            train_pbar.set_postfix({'loss': f"{current_loss:.4f}", 'dice': f"{current_dice:.4f}"})

        train_loss /= num_batches
        train_dice /= num_batches

        # --- 驗證 ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_batches = 0

        # 使用 tqdm 包裝 val_loader
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  ")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                # Overlap-tile（同訓練，與 inference 一致）
                if border_pad > 0:
                    images = F.pad(images, (border_pad,)*4, mode='reflect')
                    outputs = model(images)
                    crop = (outputs.shape[-1] - args.img_size) // 2
                    outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
                else:
                    outputs = model(images)

                if use_focal:
                    loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)
                else:
                    loss = criterion(outputs, masks)

                # 計算當前 batch 的數值
                current_loss = loss.item()
                current_dice = dice_score(outputs, masks)
                
                val_loss += current_loss
                val_dice += current_dice
                num_batches += 1
                
                # 更新進度條後方的即時資訊
                val_pbar.set_postfix({'loss': f"{current_loss:.4f}", 'dice': f"{current_dice:.4f}"})

        val_loss /= num_batches
        val_dice /= num_batches

        # --- Scheduler 更新 ---
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # --- 印出 Epoch 總結 ---
        print(
            f"-> Epoch [{epoch+1}/{args.epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- 儲存最佳模型 / Early Stopping ---
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nDone. Best Val Dice: {best_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for binary segmentation")
    parser.add_argument("--data_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    main(args)