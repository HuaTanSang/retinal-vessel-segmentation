# import cv2 
# import os
# import glob
# import torchvision.transforms.functional as TF
# from torch.utils.data import Dataset

import os
import glob
from PIL import Image
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
        # Đọc ảnh
        image = Image.open(self.images[index]).convert('RGB')  # Đảm bảo ảnh là RGB
        # Chuyển sang grayscale
        image = image.convert('L')  # Grayscale
        # Resize ảnh về (width=224, height=144)
        image = image.resize((224, 144), Image.BICUBIC)
        
        # Đọc mask
        mask = Image.open(self.masks[index]).convert('L')  # Grayscale
        # Resize mask về (width=224, height=144)
        mask = mask.resize((224, 144), Image.BICUBIC)
        
        # Chuyển thành tensor
        image = TF.to_tensor(image) * 255  # shape (1, 224, 144), giá trị [0, 255]
        image = image.repeat(3, 1, 1)  # shape (3, 224, 144)
        mask = TF.to_tensor(mask)  # shape (1, 224, 144)
        
        return {
            "image": image,
            "mask": mask
        }