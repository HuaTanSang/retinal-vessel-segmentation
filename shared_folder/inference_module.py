import torch
import matplotlib.pyplot as plt
# from utils import * 
from taod_cfnet.cfnet_model import TAOD_CFNet
from shared_folder.compute_score import * 

def load_model(checkpoint_dir: str, device: torch.device, model=TAOD_CFNet(3, 1)) -> torch.nn.Module: 
    
    checkpoint = torch.load(checkpoint_dir, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("Model has been loaded successfully!")

    return model 


def inference(image: torch.Tensor, mask: torch.Tensor, model: torch.nn.Module, device: torch.device): 
    image = image.to(device)
    mask = mask.to(device)

    image = image.unsqueeze(0)

    with torch.no_grad(): 
        logits = model(image)
        pred = torch.sigmoid(logits)
       

        pred = (pred > 0.5) # .float().cpu().numpy().squeeze() 
        print(pred.max(), pred.min())
        print(torch.unique(pred, return_counts=True))
        raise 
    image = image.squeeze(0).cpu().permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)
    # image = (image * 0.5) + 0.5 # Revert Normalize 
    
    mask = mask.cpu().numpy().squeeze() 

    return image, mask, pred



def plot_results(image, mask, pred): 

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1) 
    plt.title("Original image")
    plt.imshow(image)
    plt.axis("off")

    # True mask 
    plt.subplot(1, 3, 2)
    plt.title("True mask")
    plt.imshow(mask)
    plt.axis("off")

    # Prediction mask 
    plt.subplot(1, 3, 3)
    plt.title("Prediction mask")
    plt.imshow(pred)

    plt.show() 

    # Calculating score for each tuple of results 
    print(f"Dice Score: {compute_dice(pred, mask)}")
    print(f"IOU Scores: {compute_iou(pred, mask)}")



