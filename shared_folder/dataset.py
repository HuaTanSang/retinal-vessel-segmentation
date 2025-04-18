import cv2 
import os
import glob
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
            glob.glob(os.path.join(root_folder, 'manual1', '**.jpg')) + 
            glob.glob(os.path.join(self.root_folder, 'manual1', '**.tif'))
        )

        assert len(self.images) == len(self.masks), "Number of mask and image mis-match!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize((144, 256))
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize((144, 256))

        image = TF.to_tensor(image)*255
        image = image.repeat(3, 1, 1)
        mask = TF.to_tensor(mask)

        return {
            "image": image,
            "mask": mask
        }
