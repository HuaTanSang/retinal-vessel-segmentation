import torch
import torch.nn as nn

from helper_function import * 

class ODGA(nn.Module):
    def __init__(self):
        super(ODGA, self).__init__()

    def forward(self, x):
        """
        x: tensor đầu vào shape (B, 3, H, W) - ảnh RGB (3 kênh giống nhau).
        Output: tensor sau khi điều chỉnh độ sáng vùng đĩa thị.
        """
        B, C, H, W = x.shape
        assert C == 3, "ODGA yeu cau 3 kenh mau gray-scale"

        I = x[:, 0:1, :, :].clone()  # shape (B, 1, H, W)
        MaxI = torch.amax(I, dim=(2, 3), keepdim=True)
        MinI = torch.amin(I, dim=(2, 3), keepdim=True)

        Eqs1 = MaxI / 2 

        Bound = (3/4) * (1 - (MinI/MaxI))
        Step = (7/8) * (1 - (MinI/MaxI))

        I = torch.where(I > Eqs1, I * Step, I)

        Eqs2 = compute_eqs2(I)

        odga = torch.where(I > Eqs2, I * Bound, I)  # shape: (B, 1, H, W)
        odga = odga.repeat(1, 3, 1, 1)  # shape: (B, 3, H, W)
        return odga 
