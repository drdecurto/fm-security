"""
OT/ICS-specific preprocessing utilities.

Functions for temporal feature engineering, data augmentation,
and class-balancing strategies tailored to industrial datasets.
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

from src.data.config import SMOTE_CONFIG

logger = logging.getLogger(__name__)


def add_delta_features(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add rate-of-change (delta) features for continuous sensor readings.

    Args:
        df: Feature DataFrame.
        columns: Columns to compute deltas for. If None, uses all numeric.

    Returns:
        DataFrame with appended delta columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    for col in columns:
        result[f"{col}_delta"] = df[col].diff().fillna(0)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add rolling window statistics for sensor columns.

    Args:
        df: Feature DataFrame.
        columns: Columns to aggregate. If None, uses all numeric.
        window: Rolling window size.

    Returns:
        DataFrame with appended rolling features.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    for col in columns:
        result[f"{col}_rmean"] = df[col].rolling(window, min_periods=1).mean()
        result[f"{col}_rstd"] = df[col].rolling(window, min_periods=1).std().fillna(0)
    return result


def balance_dataset_smote(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance dataset using SMOTE or its variants.

    Args:
        X: Feature matrix.
        y: Label vector.
        method: 'smote', 'borderline', or 'adasyn'.
        random_state: Random seed.

    Returns:
        Balanced (X, y) tuple.
    """
    logger.info(f"Balancing dataset with {method}...")
    logger.info(f"Before: {y.value_counts().to_dict()}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    k_neighbors = min(SMOTE_CONFIG["k_neighbors"],
                      pd.Series(y_encoded).value_counts().min() - 1)
    k_neighbors = max(1, k_neighbors)

    if method == "smote":
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    elif method == "borderline":
        sampler = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=random_state)
    elif method == "adasyn":
        sampler = ADASYN(n_neighbors=k_neighbors, random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X.values, y_encoded)

    X_result = pd.DataFrame(X_resampled, columns=X.columns)
    y_result = pd.Series(le.inverse_transform(y_resampled), name=y.name)

    logger.info(f"After: {y_result.value_counts().to_dict()}")
    return X_result, y_result


def subsample_per_class(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subsample exactly k instances per class for few-shot experiments.

    For classes with fewer than k samples, all available samples are used.

    Args:
        X: Feature matrix.
        y: Label vector.
        k: Number of samples per class.
        random_state: Random seed.

    Returns:
        Subsampled (X, y) tuple.
    """
    rng = np.random.RandomState(random_state)
    indices = []

    for label in sorted(y.unique()):
        label_idx = y[y == label].index.tolist()
        n_sample = min(k, len(label_idx))
        chosen = rng.choice(label_idx, size=n_sample, replace=False)
        indices.extend(chosen)

    rng.shuffle(indices)
    return X.loc[indices].reset_index(drop=True), y.loc[indices].reset_index(drop=True)


def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute inverse-frequency class weights for imbalanced ICS datasets.

    Args:
        y: Label vector.

    Returns:
        Dictionary mapping class labels to weights.
    """
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)
    weights = {label: total / (n_classes * count) for label, count in counts.items()}
    return weights
