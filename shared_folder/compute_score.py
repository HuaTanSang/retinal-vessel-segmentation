import numpy as np

def compute_jaccard(pred, true):
    """
    Jaccard index = IoU = |pred ∩ true| / |pred ∪ true|
    """
    pred = pred.astype(np.float32)
    true = true.astype(np.float32)
    inter = (pred * true).sum()
    union = pred.sum() + true.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

def compute_dice(predicted_mask, mask, threshold=0.5):
    iou = compute_iou(predicted_mask, mask)
    return 2 * iou / (1 + iou) if iou != 0 else 0


def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def compute_f1(pred, true):
    pred = pred.astype(np.bool_)
    true = true.astype(np.bool_)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return (2 * precision * recall) / (precision + recall + 1e-6)

def compute_precision(pred, true):
    pred = pred.astype(np.bool_)
    true = true.astype(np.bool_)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    return tp / (tp + fp + 1e-6)

def compute_recall(pred, true):
    pred = pred.astype(np.bool_)
    true = true.astype(np.bool_)
    tp = np.logical_and(pred, true).sum()
    fn = np.logical_and(~pred, true).sum()
    return tp / (tp + fn + 1e-6)


def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "iou": compute_iou,
        "jaccard": compute_jaccard,
        "dice":    compute_dice,
        "f1":      compute_f1,
        "precision": compute_precision,
        "recall":    compute_recall
    }

    scores = {name: [] for name in metrics}
    for pred, true in zip(predicted_masks, masks):
        for name, fn in metrics.items():
            scores[name].append(fn(pred, true))

    return {name: np.mean(vals) for name, vals in scores.items()}
