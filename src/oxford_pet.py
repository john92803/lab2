import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class OxfordPetDataset(Dataset):

    def __init__(self, root, split_file, mode="train", img_size=256):

        self.root = root
        self.img_size = img_size
        self.mode = mode

        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        self.ids = []
        with open(split_file, "r") as f:
            for line in f:
                name = line.strip().split()[0]
                img_path = os.path.join(self.img_dir, f"{name}.jpg")
                if os.path.exists(img_path):
                    self.ids.append(name)

        print(f"Loading from {os.path.basename(split_file)}")

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 顏色擴增
        self.color_jitter = T.ColorJitter(
            brightness=0.5,  
            contrast=0.5,     
            saturation=0.4,   
            hue=0.1          
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

        if self.mode == "train":
            # 隨機裁切
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(3/4, 4/3)
            )
            image = TF.resized_crop(image, i, j, h, w, (self.img_size, self.img_size), Image.BILINEAR)
            mask  = TF.resized_crop(mask,  i, j, h, w, (self.img_size, self.img_size), Image.NEAREST)

            # 水平翻轉
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 隨機旋轉
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

            # 顏色抖動
            image = self.color_jitter(image)

            # 隨機灰階
            if random.random() < 0.1:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
        else:
            # 不做任何增強
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        image = self.normalize(image)

        return image, mask


def get_dataloaders(data_root, test_file="test_unet.txt", img_size=256, batch_size=16, num_workers=2):

    ann_dir = os.path.join(data_root, "annotations")

    train_dataset = OxfordPetDataset(data_root, os.path.join(ann_dir, "kaggle_train.txt"), mode="train", img_size=img_size)
    val_dataset   = OxfordPetDataset(data_root, os.path.join(ann_dir, "kaggle_val.txt"),   mode="val",   img_size=img_size)
    test_dataset  = OxfordPetDataset(data_root, os.path.join(ann_dir, test_file),           mode="test",  img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
