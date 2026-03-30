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

    if args.model == "unet":
        args.test_file = "test_unet.txt"
        args.save_path = "saved_models/unet_best.pth"
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == "resnet34_unet":
        args.test_file = "test_res_unet.txt"
        args.save_path = "saved_models/resnet34_unet_best.pth"
        model = ResNet34UNet(out_channels=1).to(device)
    else:
        raise ValueError(f"wrong model: {args.model}")

    print(f"model: {args.model}")
    data_root = os.path.abspath(args.data_root)
    train_loader, val_loader, _ = get_dataloaders(
        data_root, test_file=args.test_file,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Overlap-tile

    with torch.no_grad():
        _dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
        _out_size = model(_dummy).shape[-1]
    border_pad = (args.img_size - _out_size) // 2
    if border_pad > 0:
        print(f"Overlap-tile strategy")

    if args.model == "resnet34_unet":
        focal = True
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        focal = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    bce = nn.BCEWithLogitsLoss() 

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # training
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_batches = 0

        # train by tqdm
        train_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        
        for images, masks in train_tqdm:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Overlap-tile
            if border_pad > 0:
                images = F.pad(images, (border_pad,)*4, mode='reflect')
                outputs = model(images)
                crop = (outputs.shape[-1] - args.img_size) // 2
                outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
            else:
                outputs = model(images)

            if focal:
                loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)
            else:
                loss = bce(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            current_dice = dice_score(outputs, masks)
            
            train_loss += current_loss
            train_dice += current_dice
            num_batches += 1
            
            train_tqdm.set_postfix({'loss': f"{current_loss:.4f}", 'dice': f"{current_dice:.4f}"})

        train_loss /= num_batches
        train_dice /= num_batches

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_batches = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  ")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                if border_pad > 0:
                    images = F.pad(images, (border_pad,)*4, mode='reflect')
                    outputs = model(images)
                    crop = (outputs.shape[-1] - args.img_size) // 2
                    outputs = outputs[:, :, crop:crop+args.img_size, crop:crop+args.img_size]
                else:
                    outputs = model(images)

                if focal:
                    loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)
                else:
                    loss = bce(outputs, masks) + dice_loss(outputs, masks)

                current_loss = loss.item()
                current_dice = dice_score(outputs, masks)
                
                val_loss += current_loss
                val_dice += current_dice
                num_batches += 1

                val_pbar.set_postfix({'loss': f"{current_loss:.4f}", 'dice': f"{current_dice:.4f}"})

        val_loss /= num_batches
        val_dice /= num_batches

        # scheduler update
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"-> Epoch [{epoch+1}/{args.epochs}]| "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
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