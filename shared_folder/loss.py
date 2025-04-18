import torch 
import torch.nn as nn 

class Dice_Loss(nn.Module): 
    def __init__(self, smooth=1e-6): 
        super(Dice_Loss, self).__init__() 
        self.smooth = smooth 

    def forward(self, logits, target):
        logits = torch.sigmoid(logits)  
        intersection = (logits * target).sum(dim=(2,3))  
        union = logits.sum(dim=(2,3)) + target.sum(dim=(2,3))  
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

