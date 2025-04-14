import torch 
import numpy as np 

def compute_dice(predicted_mask, mask, threshold=0.5):
    iou = compute_iou(predicted_mask, mask)
    return 2 * iou / (1 + iou) if iou != 0 else torch.tensor(0.0)


def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def compute_f1(pred_mask, true_mask):
    tp = ((pred_mask == true_mask) & (true_mask == 1)).sum()
    fp = ((pred_mask != true_mask) & (true_mask == 1)).sum()
    fn = ((pred_mask != true_mask) & (true_mask == 0)).sum()

    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return (2 * pre * recall) / (pre + recall)


def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "iou": compute_iou,
        "dice": compute_dice,
        "f1" : compute_f1
    }

    scores = {metric_name: [] for metric_name in metrics}

    for predicted_mask, mask in zip(predicted_masks, masks):  
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(predicted_mask, mask)) 

    return {metric_name: np.mean(values) for metric_name, values in scores.items()}

