import torch 
import torch.nn as nn 

from competitiveconfusionblock import CFB 
from opticdiscgradientadjustmentalgorithm import ODGA
from trumpetattention import TrumpetAttention

class TAOD_CFNet(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(TAOD_CFNet, self).__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.odga = ODGA() 

        # Encoder 
        self.cfb0_0 = CFB(self.in_channels, 4)
        self.tam1 = TrumpetAttention(4)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)

        self.cfb0_1 = CFB(4, 8)
        self.tam2 = TrumpetAttention(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cfb0_2 = CFB(8, 16)
        self.tam3 = TrumpetAttention(16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.cfb0_3 = CFB(16, 32)
        self.tam4 = TrumpetAttention(32)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.cfb0_4 = CFB(32, 64)
        self.cfb0_5 = CFB(64, 64)

        # Decoder: Cat the TAM before conduct the CFB
        self.cfb1_0 = CFB(64, 64)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=1)
        )

        self.cfb1_1 = CFB(64, 32)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.cfb1_2 = CFB(32, 16)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=1)
        )

        self.cfb1_3 = CFB(16, 8)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 4, kernel_size=1)
        )

        self.cfb1_4 = CFB(8, 4)
        self.out_conv = nn.Conv2d(4, self.out_channels, kernel_size=1) 



    def forward(self, x): 
        # x = self.odga(x)

        # ======== Encoder ========
        x0 = self.cfb0_0(x)  # 64
        t0 = self.tam1(x0)
        x1 = self.maxpool0(t0)

        x1 = self.cfb0_1(x1)  # 128
        t1 = self.tam2(x1)
        x2 = self.maxpool1(t1)

        x2 = self.cfb0_2(x2)  # 256
        t2 = self.tam3(x2)
        x3 = self.maxpool3(t2)

        x3 = self.cfb0_3(x3)  # 512
        t3 = self.tam4(x3)
        x4 = self.maxpool4(t3)

        x4 = self.cfb0_4(x4)  # bottleneck 1024
        x5 = self.cfb0_5(x4) 

        # ======== Decoder ========
        d0 = self.cfb1_0(x5)   # 1024
        up1 = self.up1(d0)     # upsample to 512
        d1 = self.cfb1_1(torch.cat([up1, t3], dim=1))  # concat với TAM từ encoder

        up2 = self.up2(d1)     # 256
        d2 = self.cfb1_2(torch.cat([up2, t2], dim=1))

        up3 = self.up3(d2)     # 128
        d3 = self.cfb1_3(torch.cat([up3, t1], dim=1))

        up4 = self.up4(d3)     # 64
        d4 = self.cfb1_4(torch.cat([up4, t0], dim=1))

        out = self.out_conv(d4)
        return out 
