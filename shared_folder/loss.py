import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 

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

class GeneralizedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Generalized Focal Loss implementation.
        Args:
            gamma: focusing parameter (higher = more focus on hard examples).
            alpha: balancing parameter (can be a float or a list/tensor for class-wise weights).
            reduction: 'none' | 'mean' | 'sum'
        """
        super(GeneralizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: raw output from model, shape (B, C)
            targets: ground truth class indices, shape (B)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (B)
        pt = torch.exp(-ce_loss)  # pt = softmax prob of correct class
        focal_term = (1 - pt) ** self.gamma

        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.ones_like(targets, dtype=torch.float32) * self.alpha
        elif isinstance(self.alpha, (list, torch.Tensor)):
            alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
        else:
            raise ValueError("Unsupported alpha format")

        loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



def compute_pos_weight(dataset, device):  # for BCEWithLogitsLoss 
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_pos = 0.0
    total_neg = 0.0

    for batch in loader:
        mask = batch['mask'].to(device)          # shape (1,1,H,W), giá trị 0.0 hoặc 1.0
        total_pos += mask.sum().item()           # cộng tất cả pixel =1
        total_neg += (1.0 - mask).sum().item()    # pixel =0 là 1-mask

    # tránh chia cho 0
    if total_pos == 0:
        raise ValueError("Không tìm thấy pixel positive nào trong dataset!")
    pos_weight = torch.tensor([total_neg / total_pos], device=device)
    return pos_weight