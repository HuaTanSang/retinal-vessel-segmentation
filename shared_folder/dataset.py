import os
import glob
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class HRF_Dataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder

        self.images = sorted(
            glob.glob(os.path.join(self.root_folder, 'images', '**.jpg')) +
            glob.glob(os.path.join(self.root_folder, 'images', '**.JPG'))
        )
        self.masks = sorted(
            glob.glob(os.path.join(self.root_folder, 'manual1', '**.jpg')) + 
            glob.glob(os.path.join(self.root_folder, 'manual1', '**.tif'))
        )

        assert len(self.images) == len(self.masks), "Number of mask and image mis-match!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load và xử lý image
        image = Image.open(self.images[index]).convert('RGB')  # Đảm bảo ảnh là RGB
        # image = image.convert('L')  # Chuyển sang grayscale
        # image = image.resize((224, 144), Image.BICUBIC)  # Resize
        
        # Load và xử lý mask
        mask = Image.open(self.masks[index]).convert('L')  # Grayscale
        # mask = mask.resize((224, 144), Image.NEAREST)  # Resize
        
        # Chuyển thành tensor
        image = TF.to_tensor(image)*255  # Shape: (1, 224, 144), giá trị [0, 1]
        # image = image.repeat(3, 1, 1)  # Shape: (3, 224, 144)
        mask = TF.to_tensor(mask)*255  # Shape: (1, 224, 144), giá trị [0, 1]
        
        # Binarize mask: giá trị > 0 thành 1, giá trị = 0 giữ nguyên
        # mask = (mask > 0.0).float()  # Shape: (1, 224, 144), giá trị {0.0, 1.0}

        return {
            "image": image,
            "mask": mask
        }