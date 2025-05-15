import sys
sys.path.append("/home/huatansang/Documents/Research")

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from shared_folder.compute_score import *
from shared_folder.loss import *
from cfnet_model import TAOD_CFNet
from shared_folder.utils import * 
from shared_folder.dataset import HRF_Dataset
from helper_function import * 

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from shutil import copyfile

import os
import torch
import torch.nn as nn

import torch
torch.cuda.empty_cache()

def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, criterion: Dice_Loss):
    model.train()
    running_loss = .0
    with tqdm(desc=f'Epoch {epoch} - Training', unit='it', total=len(dataloader)) as pb:
        for it, batch in enumerate(dataloader):
            images, masks = batch['image'], batch['mask']

            images = images.to(device)
            masks = masks.to(device)
            
            masked = model(images)
            loss = criterion(masked, masks)
            
            # Back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            
            # Update training status
            pb.set_postfix(loss=running_loss / (it + 1))
            pb.update()

    return running_loss / len(dataloader)

def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_predictions = []
    all_masks = []
    
    with tqdm(desc=f'Epoch {epoch} - Evaluating', unit='it', total=len(dataloader)) as pb:
        for batch in dataloader:
            images, masks = batch['image'], batch['mask']

            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                logits = model(images)
            
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long().cpu().numpy()
            masks = masks.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_masks.extend(masks)
            
            pb.update()

    scores = compute_scores(all_predictions, all_masks)
    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(dict_to_save, os.path.join(checkpoint_dir, "last_model.pth"))


def main(folder_dir, checkpoint_dir):
    set_seed(42)
    
    # Load data
    data = HRF_Dataset(folder_dir)
    train_indices, val_indices = train_test_split(range(len(data)), test_size=0.3, random_state=42)
    
    train_dataset = torch.utils.data.Subset(data, train_indices)
    val_dataset = torch.utils.data.Subset(data, val_indices)
    

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # for batch in train_loader: 
    #     image = batch['image']
    #     print(image.shape)
    #     raise 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pos_weight = compute_pos_weight(train_dataset, device)

    # Define model
    model = TAOD_CFNet(3, 1)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    criterion = CustomLoss(alpha=0.9)

    epoch = 0
    allowed_patience = 5
    best_score = 0
    compared_score = "iou"
    patience = 0
    exit_train = False
    
    # Training model
    while not exit_train :
        train_loss = train_model(epoch, model, train_loader, optimizer, device, criterion)
        scores = evaluate_model(epoch, model, eval_loader, device)
        print(f"Epoch {epoch}: IOU = {scores['iou']}, Dice = {scores['dice']}, F1 = {scores['f1']}, Jaccard = {scores['jaccard']}, Precision = {scores['precision']}, Recall = {scores['recall']}, Train Loss = {train_loss:.4f}")
        
        score = scores[compared_score]
        # scheduler.step(score)  # Cập nhật learning rate dựa trên Dice Score
        
        is_best_model = False
        if score > best_score:
            best_score = score
            patience = 0
            is_best_model = True
        else:
            patience += 1
        
        if  epoch == 30:
            exit_train = True
        
        save_checkpoint({
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_dir)
        
        if is_best_model:
            copyfile(
                os.path.join(checkpoint_dir, "last_model.pth"),
                os.path.join(checkpoint_dir, "best_model.pth")
            )
        
        epoch += 1
    
if __name__ == "__main__":
    main(
        folder_dir='/home/huatansang/Documents/Research/Dataset/hrf',
        checkpoint_dir='/home/huatansang/Documents/Research/taod_cfnet/checkpoint'
    )