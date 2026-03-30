import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # no padding no bn
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Dconv = DoubleConv(in_channels, out_channels)
        self.Mpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.Dconv(x)
        pool = self.Mpool(skip)
        return pool, skip


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.UPconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Dconv = DoubleConv(in_channels, out_channels) 

    def forward(self, x, skip):
        x = self.UPconv(x)

        # center crop skip [0 batch, 1 channel, 2 height, 3 width]
        difh = skip.size(2) - x.size(2)
        difw = skip.size(3) - x.size(3)
        skip = skip[:, :, difh // 2 : difh // 2 + x.size(2),
                         difw // 2 : difw // 2 + x.size(3)]
        
        x = torch.cat([x, skip], dim=1)
        x = self.Dconv(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.en1 = Encoder(in_channels, 64)
        self.en2 = Encoder(64, 128)
        self.en3 = Encoder(128, 256)   
        self.en4 = Encoder(256, 512)

        self.bt = DoubleConv(512, 1024)

        self.de4 = Decoder(1024, 512)
        self.de3 = Decoder(512, 256)
        self.de2 = Decoder(256, 128)
        self.de1 = Decoder(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        p1, s1 = self.en1(x)
        p2, s2 = self.en2(p1)
        p3, s3 = self.en3(p2)
        p4, s4 = self.en4(p3)

        b = self.bt(p4)

        d4 = self.de4(b, s4)
        d3 = self.de3(d4, s3)
        d2 = self.de2(d3, s2)
        d1 = self.de1(d2, s1)
        out = self.output(d1)
        return out



