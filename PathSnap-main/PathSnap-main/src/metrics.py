"""
metrics.py - Evaluation metrics for PathSnap Map-Matching.
Implements Top-1 observation accuracy, macro F1, log loss, candidate miss rate,
and route continuity.
"""

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def compute_observation_metrics(meta, y_true, y_prob, arcs, top_k=10):
    """
    Computes observation-level Top-1 accuracy, candidate miss rate,
    and route continuity from candidate prediction probabilities.

    Args:
        meta: list of (point_idx, candidate_arc_id)
        y_true: ground truth binary labels (1 if true arc, 0 otherwise)
        y_prob: model predicted probabilities for class 1
        arcs: numpy array of road network arcs (from_node, to_node)

    Returns:
        dict with top1_accuracy, candidate_miss_rate, route_continuity,
                  pointwise_predictions, true_segments
    """
    # Group candidates by point_idx
    from collections import defaultdict
    point_cands = defaultdict(list)

    for idx, (pt_idx, cid) in enumerate(meta):
        point_cands[pt_idx].append({
            "cid": cid,
            "y": y_true[idx] if y_true is not None else 0,
            "prob": y_prob[idx],
        })

    total_points = len(point_cands)
    correct_top1 = 0
    missed_cands = 0
    pred_sequence = []
    true_sequence = []

    for pt_idx in range(total_points):
        cands = point_cands[pt_idx]
        # Sort by model predicted probability descending
        cands.sort(key=lambda x: x["prob"], reverse=True)
        top_cand = cands[0]["cid"]
        pred_sequence.append(top_cand)

        if y_true is not None:
            # Check if any candidate was positive
            has_pos = any(c["y"] == 1 for c in cands)
            if not has_pos:
                missed_cands += 1
                true_sequence.append(-1)
            else:
                true_cid = next(c["cid"] for c in cands if c["y"] == 1)
                true_sequence.append(true_cid)
                if top_cand == true_cid:
                    correct_top1 += 1

    top1_acc = (correct_top1 / total_points) if (y_true is not None and total_points > 0) else 0.0
    miss_rate = (missed_cands / total_points) if (y_true is not None and total_points > 0) else 0.0

    # Route continuity: fraction of consecutive segments that are connected
    continuity = calculate_route_continuity(pred_sequence, arcs)

    return {
        "top1_accuracy": float(top1_acc),
        "candidate_miss_rate": float(miss_rate),
        "route_continuity": float(continuity),
        "total_points": total_points,
        "pred_sequence": pred_sequence,
        "true_sequence": true_sequence,
    }


def calculate_route_continuity(pred_sequence, arcs):
    """
    Measures route continuity: percentage of consecutive steps where
    pred[t] == pred[t-1] OR arcs[pred[t-1]][1] == arcs[pred[t]][0].
    """
    if len(pred_sequence) <= 1:
        return 1.0

    connected_count = 0
    total_transitions = len(pred_sequence) - 1

    for i in range(total_transitions):
        a1 = pred_sequence[i]
        a2 = pred_sequence[i + 1]

        if a1 == a2:
            connected_count += 1
        elif 0 <= a1 < len(arcs) and 0 <= a2 < len(arcs):
            # Arcs: [from_node, to_node]
            v1 = arcs[a1, 1]
            u2 = arcs[a2, 0]
            if v1 == u2:
                connected_count += 1

    return float(connected_count / total_transitions)


def compute_classification_metrics(y_true, y_pred, y_prob):
    """
    Computes standard sample-level classification metrics:
    Macro F1, Precision, Recall, and Log Loss.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        # Clip probabilities to avoid log loss numerical issues
        eps = 1e-15
        clipped_prob = np.clip(y_prob, eps, 1 - eps)
        # Log loss expects shape (N, 2) or (N,)
        loss = float(log_loss(y_true, clipped_prob))
    except Exception:
        loss = float("nan")

    return {
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "log_loss": loss,
    }

