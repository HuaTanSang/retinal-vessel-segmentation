import sys 
sys.path.append("/home/huatansang/Documents/Research/retinal-vessel-segmentation")

from PIL import Image
import os
import numpy as np 

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Định nghĩa hàm compute_eqs2
def compute_eqs2(Light):
    """
    Light: tensor shape (C, H, W)
    Output: Eqs2, shape (C, 1, 1)
    """
    C, H, W = Light.shape
    device = Light.device

    i_idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
    j_idx = torch.arange(W, device=device).view(1, W).expand(H, W)

    pos = (i_idx * W + j_idx).float() + 1e-6
    weighted = Light / pos
    eqs2 = weighted.sum(dim=(1, 2))
    eqs2 = eqs2.view(C, 1, 1)
    return eqs2

# Định nghĩa class ODGA sửa đổi để giữ màu
class ODGA(nn.Module):
    def __init__(self):
        super(ODGA, self).__init__()

    def forward(self, x):
        """
        x: tensor đầu vào shape (C, H, W) - ảnh RGB (3 kênh màu).
        Output: tensor sau khi điều chỉnh độ sáng vùng đĩa thị cho từng kênh, shape (C, H, W).
        """
        C, H, W = x.shape
        assert C == 3, "ODGA yêu cầu 3 kênh màu RGB"

        # Xử lý từng kênh màu riêng biệt
        output_channels = []
        for c in range(C):
            I = x[c:c+1, :, :].clone()  # shape (1, H, W) cho kênh c
            MaxI = torch.amax(I, dim=(1, 2), keepdim=True)  # shape (1, 1, 1)
            MinI = torch.amin(I, dim=(1, 2), keepdim=True)  # shape (1, 1, 1)

            Eqs1 = MaxI / 2  # shape (1, 1, 1)

            Bound = (3/4) * (1 - (MinI / MaxI))  # shape (1, 1, 1)
            Step = (7/8) * (1 - (MinI / MaxI))  # shape (1, 1, 1)

            I = torch.where(I > Eqs1, I * Step, I)  # shape (1, H, W)

            Eqs2 = compute_eqs2(I)  # shape (1, 1, 1)

            odga_channel = torch.where(I > Eqs2, I * Bound, I)  # shape (1, H, W)
            output_channels.append(odga_channel)

        # Kết hợp các kênh
        odga = torch.cat(output_channels, dim=0)  # shape (3, H, W)
        return odga

# Hàm cắt ảnh tích hợp ODGA
def split_image_with_odga(input_folder, output_folder):
    """
    Đọc ảnh từ input_folder, áp dụng ODGA, cắt thành 256 patch và lưu vào output_folder.
    
    Parameters:
    - input_folder: Đường dẫn đến folder chứa ảnh gốc (3504x2336)
    - output_folder: Đường dẫn đến folder lưu patch (219x146)
    """
    # Tạo folder output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Kích thước patch
    patch_width, patch_height = 219, 146
    patches_per_row = 16  # 3504 / 219 = 16
    patches_per_col = 16  # 2336 / 146 = 16

    # Khởi tạo ODGA
    odga_model = ODGA()

    # Biến đổi để chuyển PIL Image thành tensor và ngược lại
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Đọc tất cả file ảnh trong folder input
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.tif')):
            # Đọc ảnh
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Đảm bảo ảnh là RGB

            # Kiểm tra kích thước ảnh
            if img.size == (3504, 2336):
                # Chuyển ảnh thành tensor
                img_tensor = to_tensor(img)  # shape (3, 2336, 3504)

                if (input_folder == "/home/huatansang/Documents/Research/retinal-vessel-segmentation/Dataset/hrf/images"): 
                    # Áp dụng ODGA
                    with torch.no_grad():
                        odga_output = odga_model(img_tensor)  # shape (3, 2336, 3504)

                    # Chuyển tensor về PIL Image
                    odga_output = odga_output.clamp(0, 1)  # Đảm bảo giá trị trong [0, 1]
                    img_processed = to_pil(odga_output)  # PIL Image

                else: 
                    # Nếu không phải là folder HRF, chỉ chuyển đổi về PIL Image
                    img_processed = to_pil(img_tensor)

                
                # Cắt ảnh thành 256 patch
                patch_count = 0
                for i in range(patches_per_col):
                    for j in range(patches_per_row):
                        # Tính toán vị trí cắt
                        left = j * patch_width
                        top = i * patch_height
                        right = left + patch_width
                        bottom = top + patch_height

                        # Cắt patch
                        patch = img_processed.crop((left, top, right, bottom))

                        # Tạo tên file mới: tên gốc + số thứ tự patch
                        base_name = os.path.splitext(filename)[0]
                        patch_filename = f"{base_name}_{patch_count:03d}.jpg"
                        patch_path = os.path.join(output_folder, patch_filename)

                        # Lưu patch
                        patch.save(patch_path, quality=95)
                        patch_count += 1

                print(f"Đã xử lý {filename}: {patch_count} patch được lưu.")
            else:
                print(f"Ảnh {filename} không có kích thước 3504x2336, bỏ qua.")

    print("Hoàn tất!")

# Đường dẫn đến folder
input_folder = "/home/huatansang/Documents/Research/retinal-vessel-segmentation/Dataset/hrf/images"
output_folder = "/home/huatansang/Documents/Research/retinal-vessel-segmentation/Dataset/preprocessed_hrf/images"

# Chạy hàm
split_image_with_odga(input_folder, output_folder)
mask_inp = "/home/huatansang/Documents/Research/retinal-vessel-segmentation/Dataset/hrf/manual1"
mask_out = "/home/huatansang/Documents/Research/retinal-vessel-segmentation/Dataset/preprocessed_hrf/manual1"

split_image_with_odga(mask_inp, mask_out)