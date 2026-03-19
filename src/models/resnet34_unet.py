import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock（用於 ResNet-18 / ResNet-34）

    結構：
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU

    提示：
    - 當 stride=2 或 in_channels != out_channels 時，
      shortcut 需要用 1x1 Conv + BN 來匹配維度
    - 否則 shortcut 就是 identity（直接加）
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: 定義兩層 Conv + BN，以及 shortcut（如果需要的話）
        raise NotImplementedError

    def forward(self, x):
        # TODO: 主路徑 + shortcut -> ReLU
        raise NotImplementedError


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 編碼器（不含最後的 avgpool 和 fc）

    結構：
        Initial:  Conv7x7(stride=2) -> BN -> ReLU -> MaxPool(stride=2)
                  輸入 [B,3,256,256] -> 輸出 [B,64,64,64]

        Stage 1:  3 個 BasicBlock, 64 ch,  stride=1  -> [B, 64,  64, 64]
        Stage 2:  4 個 BasicBlock, 128 ch, 首個 stride=2 -> [B, 128, 32, 32]
        Stage 3:  6 個 BasicBlock, 256 ch, 首個 stride=2 -> [B, 256, 16, 16]
        Stage 4:  3 個 BasicBlock, 512 ch, 首個 stride=2 -> [B, 512,  8,  8]

    提示：
    - _make_stage 是一個輔助方法，用來建立多個 BasicBlock
    - 每個 stage 的第一個 block 可能需要 stride=2 來降維
    - forward 要回傳每個 stage 的輸出，供 decoder 做 skip connection
    """

    def __init__(self):
        super().__init__()
        # TODO: 定義 initial 層 (conv7x7 + bn + relu + maxpool)
        # TODO: 定義 4 個 stage
        raise NotImplementedError

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        """
        建立一個 stage，包含 num_blocks 個 BasicBlock

        提示：
        - 第一個 block 用指定的 stride（可能是 2）
        - 之後的 block stride 都是 1
        - 第一個 block 的 in_channels 可能和 out_channels 不同
        """
        # TODO: 回傳 nn.Sequential(block1, block2, ...)
        raise NotImplementedError

    def forward(self, x):
        """
        Returns:
            features: list of feature maps [init_feat, s1, s2, s3, s4]
            各自的形狀（以 256x256 輸入為例）：
            - init_feat: [B, 64,  64, 64]   (initial conv + maxpool 後)
            - s1:        [B, 64,  64, 64]
            - s2:        [B, 128, 32, 32]
            - s3:        [B, 256, 16, 16]
            - s4:        [B, 512,  8,  8]
        """
        # TODO
        raise NotImplementedError


class DecoderBlock(nn.Module):
    """
    UNet 風格的解碼塊（用於 ResNet34_UNet 的 decoder）

    結構：上採樣 -> 拼接 skip -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU

    提示：
    - 上採樣用 ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
    - 拼接後的 channels = in_channels + skip_channels
    - 兩層 Conv 把 channels 降到 out_channels
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # TODO: 定義上採樣層和兩層 Conv + BN + ReLU
        raise NotImplementedError

    def forward(self, x, skip):
        # TODO: 上採樣 -> cat -> conv -> conv
        raise NotImplementedError


class ResNet34UNet(nn.Module):
    """
    ResNet34 (Encoder) + UNet (Decoder)

    Encoder 輸出的 feature maps 和 Decoder 的對應關係（256x256 輸入）：

    Encoder:
      init:   [B, 64,  64, 64]  ──skip──> dec4
      stage1: [B, 64,  64, 64]  ──skip──> dec3
      stage2: [B, 128, 32, 32]  ──skip──> dec2
      stage3: [B, 256, 16, 16]  ──skip──> dec1
      stage4: [B, 512,  8,  8]  (bottleneck，不做 skip)

    Decoder:
      dec1: in=512,  skip=256, out=256  -> [B, 256, 16, 16]
      dec2: in=256,  skip=128, out=128  -> [B, 128, 32, 32]
      dec3: in=128,  skip=64,  out=64   -> [B, 64,  64, 64]
      dec4: in=64,   skip=64,  out=32   -> [B, 32,  64, 64]

    Final:
      上採樣 64->128, 再上採樣 128->256
      Conv2d(32, 1, kernel_size=1)  ->  [B, 1, 256, 256]

    注意：因為 ResNet 的 initial conv+maxpool 把 256 降到了 64，
    decoder 最後需要額外上採樣回 256。你可以用多種方式處理這件事，
    例如 Upsample + Conv 或 ConvTranspose2d。
    """

    def __init__(self, out_channels=1):
        super().__init__()
        # TODO: 定義 encoder, decoder blocks, final upsample + output conv
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, 1, H, W] logits
        """
        # TODO: encoder -> decoder with skips -> upsample -> output
        raise NotImplementedError


if __name__ == "__main__":
    # 測試模型能否正常跑
    model = ResNet34UNet(out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # 預期: [2, 1, 256, 256]
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
