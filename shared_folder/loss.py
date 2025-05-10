import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # Ánh xạ logits về xác suất [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten từng ảnh (batch_size, H*W)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        # Tính Dice coefficient
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
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



class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multi-class classification.

    Parameters:
    - alpha: balancing factor (float or tensor). If float, applies to class 1 only (binary).
    - gamma: focusing parameter (default: 2.0).
    - reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: model outputs (logits), shape: (N, C) or (N,) for binary
        - targets: ground truth labels, shape: (N,) or (N, 1)
        """
        if inputs.dim() > 1 and inputs.size(1) > 1:
            # Multi-class case
            logpt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(logpt)
            logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Binary case
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            logpt = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-logpt)

        alpha_t = self.alpha if isinstance(self.alpha, float) else self.alpha[targets]
        loss = -alpha_t * ((1 - pt) ** self.gamma) * logpt

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
