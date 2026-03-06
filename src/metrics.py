"""
Shared detection evaluation helpers used across training callbacks,
visualization utilities, and the standalone evaluation script.
"""
import inspect

import torchvision.ops as ops


def accepts_pixel_mask(model):
    """Return True if the model's forward() accepts a pixel_mask argument."""
    return "pixel_mask" in inspect.signature(model.forward).parameters


def get_model_probs(logits, num_target_classes):
    """
    Apply sigmoid or softmax based on model architecture.
    - DETR-style (with background class):  num_pred == num_target + 1  -> softmax
    - RT-DETR-style (no background class): num_pred == num_target      -> sigmoid
    """
    num_pred_classes = logits.shape[-1]
    if num_pred_classes == num_target_classes + 1:
        return logits.softmax(-1)
    else:
        return logits.sigmoid()


def extract_scores_and_labels(probs_i, num_target_classes):
    """
    Extract max object scores and predicted class labels from a per-image
    probability tensor. Excludes the trailing background class for DETR-style
    models where num_pred > num_target.
    """
    num_pred_classes = probs_i.shape[-1]
    if num_pred_classes > num_target_classes:
        # DETR-style: last dimension is background — exclude it
        return probs_i[:, :-1].max(dim=-1)
    else:
        # RT-DETR-style: all dimensions are object classes
        return probs_i[:, :].max(dim=-1)


def compute_precision_recall_f1(cached_predictions, iou_threshold=0.5, conf_threshold=0.5):
    """
    Compute global Precision, Recall, and F1-score from single-pass cached predictions.

    Predictions are filtered by *conf_threshold* before IoU-based matching.
    Each entry in *cached_predictions* must contain:
        obj_scores, obj_labels, p_boxes_xyxy, t_boxes_xyxy, t_labels  (all CPU tensors)
    """
    import torch  # local import – metrics.py has no top-level torch dependency

    tp = fp = fn = 0
    for entry in cached_predictions:
        scores  = entry["obj_scores"]
        labels  = entry["obj_labels"]
        p_boxes = entry["p_boxes_xyxy"]
        t_boxes = entry["t_boxes_xyxy"]
        t_labels = entry["t_labels"]

        keep = scores >= conf_threshold
        scores, labels, p_boxes = scores[keep], labels[keep], p_boxes[keep]

        order = scores.argsort(descending=True)
        labels, p_boxes = labels[order], p_boxes[order]

        n_gt = len(t_labels)
        if len(labels) == 0:
            fn += n_gt
            continue
        if n_gt == 0:
            fp += len(labels)
            continue

        _, matched_gt = match_boxes(p_boxes, labels, t_boxes, t_labels, iou_threshold)
        image_tp = len(matched_gt)
        tp += image_tp
        fp += len(labels) - image_tp
        fn += n_gt - image_tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def match_boxes(p_boxes, p_labels, t_boxes, t_labels, iou_threshold):
    """
    Greedy IoU-based matching of predicted boxes to ground-truth boxes.
    Predictions must already be sorted by confidence (descending) by the caller.

    Args:
        p_boxes:       (N, 4) predicted boxes in xyxy format
        p_labels:      (N,)   predicted class labels
        t_boxes:       (M, 4) ground-truth boxes in xyxy format
        t_labels:      (M,)   ground-truth class labels
        iou_threshold: minimum IoU to count as a true positive

    Returns:
        pred_results: list of (best_iou, best_gt_idx) for each prediction.
                      best_gt_idx is -1 when no match was found.
        matched_gt:   set of GT indices that were successfully matched.
    """
    matched_gt = set()
    pred_results = []

    for p_box, p_label in zip(p_boxes, p_labels):
        best_iou, best_gt_idx = 0, -1
        for gt_idx, (t_box, t_label) in enumerate(zip(t_boxes, t_labels)):
            if gt_idx in matched_gt or p_label != t_label:
                continue
            iou = ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)

        pred_results.append((best_iou, best_gt_idx))

    return pred_results, matched_gt
