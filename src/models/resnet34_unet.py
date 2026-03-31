import torch
import torch.nn as nn
import torch.nn.functional as F

#   CBAM

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

#   resnet34_unet
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.st1 = nn.Sequential(Block(64, 64), Block(64, 64), Block(64, 64))
        self.st2 = nn.Sequential(Block(64, 128, 2), Block(128, 128),
                                  Block(128, 128), Block(128, 128))
        self.st3 = nn.Sequential(Block(128, 256, 2), Block(256, 256),
                                  Block(256, 256), Block(256, 256),
                                  Block(256, 256), Block(256, 256))
        self.st4 = nn.Sequential(Block(256, 512, 2), Block(512, 512), Block(512, 512))

    def forward(self, x):
        s0 = self.init(x)
        s1 = self.st1(s0)  
        s2 = self.st2(s1)  
        s3 = self.st3(s2)  
        s4 = self.st4(s3)  
        return s1, s2, s3, s4

class Decoder(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels=32):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip):
        x = self.up(x)               
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)   
        return self.cbam(x)

class ResNet34UNet(nn.Module):

    def __init__(self, out_channels=1):
        super().__init__()
        self.en = Encoder()

        # s3 (256ch) -> 512ch projection
        self.s3_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec1 = Decoder(512, 256, 32)
        self.dec2 = Decoder(32, 512, 32)
        self.dec3 = Decoder(32, 128, 32)
        self.dec4 = Decoder(32, 64, 32)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output   = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.en(x)

        d1 = self.dec1(s4, s3)

        s3_proj = self.s3_proj(s3)                                              # [512, 16, 16]
        s3_proj_32 = F.interpolate(s3_proj, scale_factor=2,
                                   mode='bilinear', align_corners=True)         # [512, 32, 32]
        d2 = self.dec2(d1, s3_proj_32)

        s2_64 = F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(d2, s2_64)

        s1_128 = F.interpolate(s1, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(d3, s1_128)

        return self.output(self.final_up(d4))
