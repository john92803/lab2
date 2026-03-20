import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm  # 新增 tqdm 引入

sys.path.append(os.path.dirname(__file__))

from oxford_pet import get_dataloaders
from models.unet import UNet
from utils import dice_score, dice_loss

def main(args):
    device = torch.accelerator.current_accelerator()
    print(f"device: {device}")

    # ===== 資料 =====
    data_root = os.path.abspath(args.data_root)
    train_loader, val_loader, _ = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ===== 模型、損失函數、優化器 =====
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            outputs = model(images)
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
                outputs = model(images)
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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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
    parser.add_argument("--test_file", type=str, default="test_unet.txt")
    parser.add_argument("--save_path", type=str, default="saved_models/unet_best.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    main(args)