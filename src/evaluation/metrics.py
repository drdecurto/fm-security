"""
Evaluation metrics tailored for ICS intrusion detection.

Includes standard classification metrics plus ICS-specific measures
such as False Alarm Rate (FAR), Detection Rate (DR), and per-attack
class analysis.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    normal_label: str = "Normal",
) -> Dict:
    """
    Compute comprehensive ICS intrusion detection metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC).
        normal_label: Label used for normal/benign traffic.

    Returns:
        Dictionary of metric names and values.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(np.unique(y_true))
    is_binary = n_classes <= 2

    metrics = {}

    # Standard metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ICS-specific: False Alarm Rate (normal misclassified as attack)
    normal_mask = y_true == normal_label
    if normal_mask.sum() > 0:
        far = (y_pred[normal_mask] != normal_label).sum() / normal_mask.sum()
        metrics["false_alarm_rate"] = far
    else:
        metrics["false_alarm_rate"] = np.nan

    # ICS-specific: Detection Rate (attack correctly identified)
    attack_mask = y_true != normal_label
    if attack_mask.sum() > 0:
        dr = (y_pred[attack_mask] != normal_label).sum() / attack_mask.sum()
        metrics["detection_rate"] = dr
    else:
        metrics["detection_rate"] = np.nan

    # Per-class metrics
    per_class = {}
    for cls in np.unique(y_true):
        cls_mask = y_true == cls
        cls_pred_mask = y_pred[cls_mask]
        per_class[str(cls)] = {
            "precision": precision_score(y_true == cls, y_pred == cls, zero_division=0),
            "recall": recall_score(y_true == cls, y_pred == cls, zero_division=0),
            "f1": f1_score(y_true == cls, y_pred == cls, zero_division=0),
            "support": int(cls_mask.sum()),
        }
    metrics["per_class"] = per_class

    # AUC (if probabilities provided)
    if y_proba is not None:
        try:
            if is_binary:
                # For binary, use column for positive class
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    pos_idx = list(classes).index(
                        [c for c in classes if c != normal_label][0]
                    )
                    metrics["auc_roc"] = roc_auc_score(
                        (y_true != normal_label).astype(int),
                        y_proba[:, pos_idx],
                    )
                else:
                    metrics["auc_roc"] = np.nan
            else:
                metrics["auc_roc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro",
                )
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["auc_roc"] = np.nan

    # Classification report (string)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0,
    )

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_labels"] = classes.tolist()

    return metrics


def compute_rare_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: int = 50,
    normal_label: str = "Normal",
) -> Dict:
    """
    Compute metrics specifically for rare attack classes.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        threshold: Classes with fewer samples than this are 'rare'.
        normal_label: Normal class label.

    Returns:
        Dict with rare class analysis.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    class_counts = pd.Series(y_true).value_counts()
    rare_classes = class_counts[
        (class_counts < threshold) & (class_counts.index != normal_label)
    ].index.tolist()

    if not rare_classes:
        return {"rare_classes": [], "message": f"No classes with <{threshold} samples"}

    rare_mask = np.isin(y_true, rare_classes)
    rare_metrics = compute_metrics(
        y_true[rare_mask], y_pred[rare_mask], normal_label=normal_label
    )

    return {
        "rare_classes": rare_classes,
        "n_rare_samples": int(rare_mask.sum()),
        "rare_accuracy": rare_metrics["accuracy"],
        "rare_f1_macro": rare_metrics["f1_macro"],
        "rare_detection_rate": rare_metrics.get("detection_rate", np.nan),
        "rare_per_class": rare_metrics["per_class"],
    }


def aggregate_runs(results_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple runs (mean ± std).

    Args:
        results_list: List of metric dictionaries from individual runs.

    Returns:
        Dictionary with mean and std for each scalar metric.
    """
    scalar_keys = [k for k in results_list[0].keys()
                   if isinstance(results_list[0][k], (int, float, np.floating))
                   and not np.isnan(results_list[0][k])]

    aggregated = {}
    for key in scalar_keys:
        values = [r[key] for r in results_list if key in r and not np.isnan(r[key])]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    aggregated["n_runs"] = len(results_list)
    return aggregated
