import os
import glob
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms 

class HRF_Dataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((512, 512))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((512, 512))
        ])

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
        image = Image.open(self.images[index])
        mask = Image.open(self.masks[index])

        image = self.img_transforms(img=image) 
        mask = self.mask_transform(img=mask) 

        return {
            'image': image*255, 
            'mask': mask 
        }

