import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    基本構建塊：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU

    提示：
    - 兩次 Conv2d 都是 kernel_size=3, padding=1（維持空間尺寸不變）
    - 每次 Conv 後接 BatchNorm2d 和 ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: 定義兩層 Conv + BN + ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # TODO: 前向傳播
        return self.conv(x)


class Encoder(nn.Module):
    """
    UNet 編碼器的單一層：DoubleConv + MaxPool2d

    提示：
    - MaxPool2d(kernel_size=2, stride=2) 讓空間尺寸減半
    - forward 需要回傳兩個東西：
      1. pool 後的特徵（傳給下一層 encoder）
      2. pool 前的特徵（用於 skip connection，傳給 decoder）
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: 定義 DoubleConv 和 MaxPool2d
        self.Dconv = DoubleConv(in_channels, out_channels)
        self.Mpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # TODO: 回傳 (pooled, skip)
        skip = self.Dconv(x)
        pooled = self.Mpool(skip)
        return pooled, skip


class Decoder(nn.Module):
    """
    UNet 解碼器的單一層：上採樣 + 拼接 skip connection + DoubleConv

    提示：
    - 上採樣用 ConvTranspose2d(kernel_size=2, stride=2) 讓尺寸加倍
    - 上採樣後的特徵和 skip 特徵在 channel 維度拼接 (torch.cat)
    - 拼接後通過 DoubleConv
    - ConvTranspose2d 的 in_channels = in_channels, out_channels = out_channels
    - DoubleConv 的 in_channels = in_channels（因為 cat 後是 out_channels + out_channels = in_channels）
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: 定義 ConvTranspose2d 和 DoubleConv
        self.UPconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Dconv = DoubleConv(in_channels, out_channels) 

    def forward(self, x, skip):
        # TODO: 上採樣 -> cat(x, skip) -> DoubleConv
        x = self.UPconv(x)
        x = torch.cat([x, skip], dim=1)  
        x = self.Dconv(x)
        return x


class UNet(nn.Module):
    """
    完整的 UNet 架構

    結構概覽（以預設 channel 數為例）：

    輸入: [B, 3, 256, 256]

    Encoder:
      enc1: 3   -> 64,   skip1: [B, 64,  256, 256],  pool: [B, 64,  128, 128]
      enc2: 64  -> 128,  skip2: [B, 128, 128, 128],  pool: [B, 128,  64,  64]
      enc3: 128 -> 256,  skip3: [B, 256,  64,  64],  pool: [B, 256,  32,  32]
      enc4: 256 -> 512,  skip4: [B, 512,  32,  32],  pool: [B, 512,  16,  16]

    Bottleneck:
      DoubleConv: 512 -> 1024,  [B, 1024, 16, 16]

    Decoder:
      dec4: 1024 -> 512,  cat(up, skip4) -> [B, 1024, 32, 32] -> [B, 512, 32, 32]
      dec3: 512  -> 256,  cat(up, skip3) -> [B, 512,  64, 64] -> [B, 256, 64, 64]
      dec2: 256  -> 128,  cat(up, skip2) -> [B, 256, 128,128] -> [B, 128,128,128]
      dec1: 128  -> 64,   cat(up, skip1) -> [B, 128, 256,256] -> [B, 64, 256,256]

    Output:
      Conv2d(64, 1, kernel_size=1)  ->  [B, 1, 256, 256]  (raw logits，不加 sigmoid)
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # TODO: 定義 4 層 Encoder, 1 個 Bottleneck, 4 層 Decoder, 1 個 Output Conv
        self.enc1 = Encoder(in_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)   
        self.enc4 = Encoder(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 輸入圖片
        Returns:
            [B, 1, H, W] 輸出 logits（不經過 sigmoid）
        """
        # TODO: encoder -> bottleneck -> decoder -> output
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        p4, s4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.output(d1)
        return out



