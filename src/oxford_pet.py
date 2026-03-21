import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch


class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet Dataset for binary segmentation."""

    def __init__(self, root, split_file, mode="train", img_size=256):
        """
        Args:
            root: path to dataset root (contains images/ and annotations/)
            split_file: path to a txt file listing image names
            mode: "train"/"val" (has mask) or "test" (no mask)
            img_size: resize images to this size
        """
        self.root = root
        self.img_size = img_size
        self.mode = mode

        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        # Read split file (each line: image_name or "image_name col2 col3 ...")
        self.ids = []
        with open(split_file, "r") as f:
            for line in f:
                name = line.strip().split()[0]
                img_path = os.path.join(self.img_dir, f"{name}.jpg")
                if os.path.exists(img_path):
                    self.ids.append(name)

        print(f"[{mode}] Loaded {len(self.ids)} samples from {os.path.basename(split_file)}")

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 訓練時才用的顏色擴增（只套用在 image，不套用在 mask）
        self.color_jitter = T.ColorJitter(
            brightness=0.5,   # 亮度變化 ±50%
            contrast=0.5,     # 對比度變化 ±50%
            saturation=0.4,   # 飽和度變化 ±40%
            hue=0.1           # 色調偏移 ±10%
        )

    def __len__(self):
        return len(self.ids)

    def _trimap_to_binary(self, trimap):
        trimap_np = np.array(trimap)
        binary = (trimap_np == 1).astype(np.float32)
        return binary

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{name}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.mode == "test":
            # 紀錄原始尺寸 (寬, 高)
            orig_w, orig_h = image.size
            
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            image = TF.to_tensor(image)
            image = self.normalize(image)
            # 回傳原始寬高
            return image, name, orig_w, orig_h
            

        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        trimap = Image.open(mask_path)
        binary_np = self._trimap_to_binary(trimap)
        mask = Image.fromarray(binary_np)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # ===== 資料擴增（只在 train mode 執行）=====
        if self.mode == "train":
            # 水平翻轉：image 和 mask 使用相同的隨機決定
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 顏色抖動：調整亮度、對比、飽和度、色調（只套用在 image）
            image = self.color_jitter(image)

            # 隨機灰階：10% 機率轉成灰階，讓模型不過度依賴顏色
            if random.random() < 0.1:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        image = self.normalize(image)

        return image, mask


def get_dataloaders(data_root, test_file="test_unet.txt", img_size=256, batch_size=16, num_workers=2):
    """
    Create train, val, test dataloaders using Kaggle splits in annotations/.

    Split files used:
        annotations/kaggle_train.txt   -> train (5173)
        annotations/kaggle_val.txt     -> val   (739)
        annotations/{test_file}        -> test  (739)

    Args:
        data_root: path to dataset root (e.g. dataset/oxford-iiit-pet)
        test_file: "test_unet.txt" or "test_res_unet.txt"
    """
    ann_dir = os.path.join(data_root, "annotations")

    train_dataset = OxfordPetDataset(data_root, os.path.join(ann_dir, "kaggle_train.txt"), mode="train", img_size=img_size)
    val_dataset   = OxfordPetDataset(data_root, os.path.join(ann_dir, "kaggle_val.txt"),   mode="val",   img_size=img_size)
    test_dataset  = OxfordPetDataset(data_root, os.path.join(ann_dir, test_file),           mode="test",  img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
