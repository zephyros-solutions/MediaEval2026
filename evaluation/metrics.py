"""Unified classification metrics for MediaEval Task 1.

All methods MUST use these functions for F1 and cross-entropy calculation
so that scores are comparable across approaches.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
)


EPS = 1e-15


def compute_metrics(y_true, y_pred, prob_vectors, ann_labels=None, method_name=""):
    """Compute all classification metrics identically for all methods.

    Args:
        y_true: ground-truth label IDs (array-like, class IDs from config.LABEL_TO_ID)
        y_pred: predicted label IDs (array-like)
        prob_vectors: list of prob dicts per instance, e.g. {"premise": 0.4, ...}
        ann_labels: optional list of soft label dicts (class_id -> prob) per instance
        method_name: display name for the report

    Returns:
        metrics: dict with f1_macro_3class, f1_macro_2class, cross_entropy, etc.
        per_class: dict with per-class f1, precision, recall, support
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 3-class
    f1_3macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_3weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 2-class: none vs implicit (class 2 maps to 0, others to 1)
    y_true_2 = np.where(y_true == 2, 0, 1)
    y_pred_2 = np.where(y_pred == 2, 0, 1)
    f1_2macro = float(f1_score(y_true_2, y_pred_2, average="macro", zero_division=0))
    f1_2weighted = float(f1_score(y_true_2, y_pred_2, average="weighted", zero_division=0))

    # Cross-entropy against soft labels
    if ann_labels is not None:
        ce_values = [
            -sum(ann_labels[i].get(j, 0) * np.log(max(prob_vectors[i].get(j, 0), EPS)) for j in range(3))
            for i in range(len(y_true))
        ]
        ce = float(np.mean(ce_values))
    else:
        ce = float(np.mean([-np.log(max(prob_vectors[i].get(y_true[i], 0), EPS)) for i in range(len(y_true))]))

    # Per-class
    prec, rec, f1_each, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    per_class = {
        "none": {"f1": float(f1_each[0]), "precision": float(prec[0]), "recall": float(rec[0]), "support": int(supp[0])},
        "premise": {"f1": float(f1_each[1]), "precision": float(prec[1]), "recall": float(rec[1]), "support": int(supp[1])},
        "conclusion": {"f1": float(f1_each[2]), "precision": float(prec[2]), "recall": float(rec[2]), "support": int(supp[2])},
    }

    metrics = {
        "method": method_name,
        "f1_macro_3class": f1_3macro,
        "f1_weighted_3class": f1_3weighted,
        "f1_macro_2class": f1_2macro,
        "f1_weighted_2class": f1_2weighted,
        "cross_entropy": ce,
        "per_class": per_class,
    }

    return metrics, per_class
