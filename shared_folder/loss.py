import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (B, 1, H, W), target: (B, 1, H, W)
        probs = torch.sigmoid(logits)

        # Tính Dice coefficient theo từng ảnh
        intersection = (probs * target).sum(dim=(1, 2, 3))  # tổng theo H, W, C
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, y_pred, y):
        """
        input: [N, C]
        target: [N, 1]
        """
        y = y.squeeze(1)
        log_pt = F.log_softmax(y_pred, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(log_pt, y, self.weight)

        return loss

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




class CustomLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()  # dùng BCE với logits

    def forward(self, logits, truth):
        # Chuyển logits thành xác suất
        pred = torch.sigmoid(logits)

        # Flatten từng ảnh (batch_size, H*W)
        pred_flat = pred.view(pred.size(0), -1)
        truth_flat = truth.view(truth.size(0), -1).type(torch.float32)

        # BCE Loss (sử dụng BCEWithLogitsLoss trực tiếp trên logits)
        bce_loss = self.bce(logits.view(logits.size(0), -1), truth_flat)

        # Abe Dice Loss
        erc = torch.pow(pred_flat, 2 * (1 - (pred_flat ** 3)))
        numerator = (2 * erc * truth_flat).sum(dim=1)
        denominator = (erc ** 2 + truth_flat).sum(dim=1)
        abe_dice = 1 - (numerator / (denominator + 1e-6)).mean()

        return self.alpha * abe_dice + (1 - self.alpha) * bce_loss
