from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import glob

class HRF_Dataset(Dataset): 
    def __init__(self, root_folder): 
        self.root_folder = root_folder 

        self.images = sorted(
            glob.glob(os.path.join(self.root_folder, 'images/**.jpg')) +
            glob.glob(os.path.join(self.root_folder, 'images/**.JPG'))
        )
        self.masks = sorted(glob.glob(self.root_folder + 'manual1/**.tif'))

        assert len(self.images == self.images), "Number of mask and image mis-match!"

        self.transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ])

        self.transforms_masks = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.transforms_img(Image.open(self.images[index]).convert('RGB'))
        mask = self.transforms_masks(Image.open(self.masks[index]).convert('L'))
        
        return {
            "image" : image, 
            "mask" : mask
        }