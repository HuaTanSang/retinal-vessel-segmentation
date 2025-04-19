import torch 
import numpy as np 

def compute_iou(pred, true):
    pred = pred.astype(np.float32)
    true = true.astype(np.float32)
    inter = (pred * true).sum()
    union = pred.sum() + true.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

def compute_dice(pred, true):
    pred = pred.astype(np.float32)
    true = true.astype(np.float32)
    inter = (pred * true).sum()
    return (2 * inter + 1e-6) / (pred.sum() + true.sum() + 1e-6)



# def compute_f1(pred_mask, true_mask):
#     tp = ((pred_mask == true_mask) & (true_mask == 1)).sum()
#     fp = ((pred_mask != true_mask) & (true_mask == 1)).sum()
#     fn = ((pred_mask != true_mask) & (true_mask == 0)).sum()

#     pre = tp / (tp + fp)
#     recall = tp / (tp + fn)
    
#     return (2 * pre * recall) / (pre + recall)


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

