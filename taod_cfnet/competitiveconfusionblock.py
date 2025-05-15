import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_function import compute_partial 

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.main = VGGBlock(in_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)  
        self.relu = nn.ReLU() 

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x)) 


class CFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFB, self).__init__()
        self.vgg_block = VGGBlock(in_channels, out_channels)
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.omega = 0.5

    def forward(self, x):
        B, C_in, H, W = x.shape

        # Branch 1 - VGG
        V = self.vgg_block(x)

        # Branch 2 - Residual
        R = self.res_block(x)

        # Initial fusion
        C = (V + R) / 2
        
        # Activation
        M1 = self.omega * (torch.sigmoid(V) + F.relu(R))
        M2 = F.relu(R)

        # Compute ∂ (partial)
        partial = compute_partial(M2)  # shape (B, C, 1, 1)

        # M3 - remove low-activated values
        M3 = torch.where(M1 > partial, M1, torch.zeros_like(M1))

        # β calculation
        maxM1 = torch.amax(M1, dim=(2, 3))                # shape (B, C)
        minM1 = torch.amin(M1, dim=(2, 3))                # shape (B, C)
        partial_M1 = compute_partial(M1).squeeze(-1).squeeze(-1)  # shape (B, C)

        beta = (maxM1 + partial_M1) / (partial_M1 - minM1 + 1e-6)  # shape (B, C)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Final output
        out = torch.where(M1 > partial, beta * M3 * C * M2, M3 * C * M2)
        return C 
