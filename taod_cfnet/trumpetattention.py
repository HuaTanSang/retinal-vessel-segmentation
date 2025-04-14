import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class TrumpetAttention(nn.Module): 
    def __init__(self, in_channels, threshold=0.9, coef=0.1): 
        super(TrumpetAttention, self).__init__()
        self.doubleconv = DoubleConv(in_channels, in_channels)
        
        # Convolution for each branch 
        self.conv_horizon = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.conv_vertical = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.conv_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
    
        self.threshold = threshold
        self.coef = coef

    def forward(self, x):  # x : (B, C, H, W)
        B, C, H, W = x.shape

        # Horizontal branch 
        x_horizon0 = x.permute(0, 3, 2, 1) # B, W, H, C
        x_horizon_1 = torch.sigmoid(torch.sigmoid(x_horizon0) + 
                                     torch.relu(x.permute(0, 3, 1, 2)).permute(0, 1, 3, 2))
        
        x_horizon_2 = x_horizon_1.permute(0, 3, 2, 1)  #(B, C, H, W)
        x_horizon_21 = self.doubleconv(x_horizon_2)

        x_horizon_3 = torch.cat([torch.sigmoid(x), x_horizon_21], dim=1)
        x_horizon_4 = self.conv_horizon(torch.sigmoid(x_horizon_3)) 

        # Vertical Branch:
        x_vertical0 = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x_vertical_1 = torch.relu(torch.relu(x_vertical0) + 
                                  torch.sigmoid(x.permute(0, 3, 2, 1)).permute(0, 1, 3, 2))
        
        x_vertical_2 = x_vertical_1.permute(0, 2, 1, 3)  # (B, C, W, H)
        x_vertical_21 = self.doubleconv(x_vertical_2)
        x_vertical_3 = torch.cat([torch.relu(x), x_vertical_21], dim=1)
        x_vertical_4 = self.conv_vertical(torch.relu(x_vertical_3))
        
        # Combination 
        x_combined = torch.cat([x_vertical_4, x_horizon_4], dim=1)  # (B, 2C, H, W)
        x_combined = self.conv_fusion(torch.sigmoid(x_combined))    # (B, C, H, W)

        # Last stage
        x_combined = torch.where(x_combined < self.threshold, torch.tensor(0.0, device=x.device), x_combined * self.coef)
        out = x + x_combined    
        return out
