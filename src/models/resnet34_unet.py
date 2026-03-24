import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  CBAM
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  Encoder（ResNet-34，隨機初始化）
# ─────────────────────────────────────────────

class BasicBlock(nn.Module):
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
    """
    ResNet-34 Encoder（不含 avgpool / fc）

    256×256 輸入各層輸出：
        s1 [B,  64, 64, 64]
        s2 [B, 128, 32, 32]
        s3 [B, 256, 16, 16]
        s4 [B, 512,  8,  8]  ← bottleneck
    """
    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.st1 = nn.Sequential(BasicBlock(64,64), BasicBlock(64,64), BasicBlock(64,64))
        self.st2 = nn.Sequential(BasicBlock(64,128,2), BasicBlock(128,128),
                                  BasicBlock(128,128), BasicBlock(128,128))
        self.st3 = nn.Sequential(BasicBlock(128,256,2), BasicBlock(256,256),
                                  BasicBlock(256,256), BasicBlock(256,256),
                                  BasicBlock(256,256), BasicBlock(256,256))
        self.st4 = nn.Sequential(BasicBlock(256,512,2), BasicBlock(512,512), BasicBlock(512,512))

    def forward(self, x):
        s0 = self.init(x)
        s1 = self.st1(s0)   # [B,  64, 64, 64]
        s2 = self.st2(s1)   # [B, 128, 32, 32]
        s3 = self.st3(s2)   # [B, 256, 16, 16]
        s4 = self.st4(s3)   # [B, 512,  8,  8]
        return s1, s2, s3, s4


# ─────────────────────────────────────────────
#  Decoder Block（論文 Fig. 2a）
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    論文順序：Upsample(2×) → Concat skip → Conv→ReLU→BN → CBAM → 32ch

    skip 必須在呼叫前已 resize 到與 upsample 後相同的空間尺寸。
    """
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
        x = self.up(x)                        # 2× upsample，保留原通道數
        x = torch.cat([x, skip], dim=1)       # concat skip（空間尺寸需相同）
        x = self.conv(x)                      # 壓縮至 32ch
        return self.cbam(x)


# ─────────────────────────────────────────────
#  ResNet34UNet（遵照論文 Fig. 2a）
# ─────────────────────────────────────────────

class ResNet34UNet(nn.Module):
    """
    架構對照 Fig. 2a：

    四條 Copy & Concatenate 箭頭（Fig. 2a 使用者讀法）:
        512(s4, 8×8)   ──────────────────────────> dec1  主路徑入口
        256(s3, 16×16) ──────────────────────────> dec1  skip（原始 256ch）
        256(s3, 16×16) ── proj(256→512) + up2× ──> dec2  skip（投影後 512ch）
        128(s2, 32×32) ── upsample 2× ───────────> dec3  skip
         64(s1, 64×64) ── upsample 2× ───────────> dec4  skip

    Decoder concat labels（論文圖示）:
        dec1: 512+256 = 768ch → 32ch  [16×16]
        dec2:  32+512 = 544ch → 32ch  [32×32]   ← s3 投影 256→512
        dec3:  32+128 = 160ch → 32ch  [64×64]
        dec4:   32+64 =  96ch → 32ch [128×128]

    Final: 128×128 ─ upsample 2× ─> 256×256 ─ Conv1×1 ─> 1ch
    """

    def __init__(self, out_channels=1):
        super().__init__()
        self.en = Encoder()

        # s3 (256ch) → 512ch projection，使其能接入「32+512」的 skip
        self.s3_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec1 = DecoderBlock(512, 256, 32)   # 256+512  ← s4(512) up + s3(256) skip
        self.dec2 = DecoderBlock(32,  512, 32)   # 32+512   ← d1(32)  up + s3_proj(512) skip
        self.dec3 = DecoderBlock(32,  128, 32)   # 32+128   ← d2(32)  up + s2(128) skip
        self.dec4 = DecoderBlock(32,   64, 32)   # 32+64    ← d3(32)  up + s1(64)  skip

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output   = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.en(x)

        # dec1: up(s4, 8→16) + s3(16×16) → 512+256=768ch → 32ch
        d1 = self.dec1(s4, s3)

        # dec2: up(d1, 16→32) + s3_proj(256→512, 16→32, 2×) → 32+512=544ch → 32ch
        # 四條 copy & concat 箭頭之一：256(s3) → 32+512
        # s3 本身是 256ch；先用 1×1 Conv 投影到 512ch，再雙線性上採樣 2× 到 32×32
        s3_proj = self.s3_proj(s3)                                              # [B, 512, 16, 16]
        s3_proj_32 = F.interpolate(s3_proj, scale_factor=2,
                                   mode='bilinear', align_corners=True)         # [B, 512, 32, 32]
        d2 = self.dec2(d1, s3_proj_32)

        # dec3: up(d2, 32→64) + s2 upsampled(32→64, 2×) → 32+128=160ch → 32ch
        s2_64 = F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(d2, s2_64)

        # dec4: up(d3, 64→128) + s1 upsampled(64→128, 2×) → 32+64=96ch → 32ch
        s1_128 = F.interpolate(s1, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(d3, s1_128)

        # final: 128×128 → 256×256 → 1ch
        return self.output(self.final_up(d4))
