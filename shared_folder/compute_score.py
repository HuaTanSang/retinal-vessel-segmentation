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

def f1_score(pred_mask, true_mask):
    f1 = 0.0

    for pred, true in pred_mask, true_mask:
        TP = ((pred == true) & (true == 1)).sum()
        FP = ((pred != true) & (true == 1)).sum()
        FN = ((pred != true) & (true == 0)).sum()

        pre = TP/(TP+FP)
        recall = TP/(TP+FN)

        f1 += (2 * pre * recall) / (pre + recall)
        
    return f1 / len(pred_mask) 


def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "iou": compute_iou,
        "dice": compute_dice
    }
    scores = {metric_name: [] for metric_name in metrics}

    for predicted_mask, mask in zip(predicted_masks, masks):  
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(predicted_mask, mask)) 

    return {metric_name: np.mean(values) for metric_name, values in scores.items()}

