"""
Visualization utilities for OT/ICS intrusion detection results.

Generates publication-quality plots for model comparisons,
few-shot curves, confusion matrices, and class-level analysis.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

logger = logging.getLogger(__name__)


def plot_model_comparison(
    df: pd.DataFrame,
    metric: str = "Accuracy",
    title: str = "Model Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing models on a given metric.

    Args:
        df: DataFrame with 'Model' column and metric columns as 'X ± Y' strings.
        metric: Column name to plot.
        title: Plot title.
        output_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["Model"].tolist()
    means = []
    stds = []

    for val in df[metric]:
        if isinstance(val, str) and "±" in val:
            parts = val.split("±")
            means.append(float(parts[0].strip()))
            stds.append(float(parts[1].strip()))
        else:
            means.append(float(val) if val else 0)
            stds.append(0)

    colors = sns.color_palette("Set2", len(models))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_few_shot_curves(
    df: pd.DataFrame,
    metric_mean: str = "F1_macro_mean",
    metric_std: str = "F1_macro_std",
    title: str = "Few-Shot Performance",
    ylabel: str = "F1-macro",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot few-shot learning curves (performance vs k).

    Args:
        df: DataFrame with columns: Model, k, metric_mean, metric_std.
        metric_mean: Column name for mean values.
        metric_std: Column name for std values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["Model"].unique()
    colors = sns.color_palette("Set1", len(models))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, model in enumerate(models):
        model_df = df[df["Model"] == model].sort_values("k")
        k_vals = model_df["k"].values
        means = model_df[metric_mean].values
        stds = model_df[metric_std].values

        ax.errorbar(
            k_vals, means, yerr=stds,
            label=model, color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=8, capsize=4, linewidth=2,
        )

    ax.set_xlabel("k (samples per class)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot class distribution as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(y.unique()) * 0.5)))

    counts = y.value_counts().sort_values()
    colors = ["#2ecc71" if c == "Normal" else "#e74c3c" for c in counts.index]

    counts.plot(kind="barh", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_title(title)

    # Add count labels
    for i, (v, label) in enumerate(zip(counts.values, counts.index)):
        ax.text(v + counts.max() * 0.01, i, f"{v:,}", va="center", fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def plot_per_class_f1(
    per_class_metrics: Dict,
    title: str = "Per-Class F1 Score",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot F1 score for each class."""
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[c]["f1"] for c in classes]
    supports = [per_class_metrics[c]["support"] for c in classes]

    fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))

    colors = ["#2ecc71" if c == "Normal" else
              "#e74c3c" if f < 0.5 else "#f39c12" if f < 0.8 else "#3498db"
              for c, f in zip(classes, f1_scores)]

    y_pos = range(len(classes))
    bars = ax.barh(y_pos, f1_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{c} (n={s})" for c, s in zip(classes, supports)])
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="F1=0.5")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    return fig
