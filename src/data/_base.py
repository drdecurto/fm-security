"""
Base dataset class for OT/ICS intrusion detection datasets.

Provides common interface for loading, preprocessing, splitting,
and analyzing industrial control system datasets.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from src.data.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for all OT/ICS datasets."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config.copy()
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.pca_transformer: Optional[PCA] = None
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load raw data from disk and apply preprocessing."""
        pass

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Load raw CSV files and return combined DataFrame."""
        pass

    @abstractmethod
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific cleaning (type casting, label mapping, etc.)."""
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply common preprocessing pipeline.

        Steps:
            1. Drop constant columns (if configured)
            2. Remove duplicate rows (if configured)
            3. Handle missing values
            4. Apply windowing (if window_size > 1)
            5. Normalize features (if configured)
            6. Apply PCA (if configured)
        """
        target = df[self.target_column].copy()
        features = df.drop(columns=[self.target_column])

        # Drop constant columns
        if self.config.get("drop_constant", True):
            constant_cols = [c for c in features.columns
                            if features[c].nunique() <= 1]
            if constant_cols:
                logger.info(f"Dropping {len(constant_cols)} constant columns: "
                            f"{constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}")
                features = features.drop(columns=constant_cols)

        # Remove duplicates
        if self.config.get("remove_duplicates", False):
            before = len(features)
            combined = pd.concat([features, target], axis=1)
            combined = combined.drop_duplicates()
            features = combined.drop(columns=[self.target_column])
            target = combined[self.target_column]
            logger.info(f"Removed {before - len(features)} duplicate rows")

        # Handle missing values
        n_missing = features.isnull().sum().sum()
        if n_missing > 0:
            logger.warning(f"Found {n_missing} missing values. Filling with column median.")
            features = features.fillna(features.median())
            # If still NaN (entire column was NaN), fill with 0
            features = features.fillna(0)

        # Ensure all numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        features = features.fillna(0)

        # Windowing (for time-series to tabular conversion)
        window_size = self.config.get("window_size", 1)
        if window_size > 1:
            features, target = self._apply_windowing(
                features, target,
                window_size=window_size,
                stride=self.config.get("window_stride", 1),
                agg=self.config.get("window_agg", "stats"),
            )

        # Normalize
        if self.config.get("normalize", True):
            self.scaler = StandardScaler()
            feature_values = self.scaler.fit_transform(features.values)
            features = pd.DataFrame(feature_values, columns=features.columns,
                                    index=features.index)

        # PCA
        if self.config.get("pca", False):
            n_components = self.config.get("pca_components", 25)
            n_components = min(n_components, features.shape[1], features.shape[0])
            self.pca_transformer = PCA(n_components=n_components, random_state=42)
            feature_values = self.pca_transformer.fit_transform(features.values)
            pca_cols = [f"PC{i+1}" for i in range(n_components)]
            features = pd.DataFrame(feature_values, columns=pca_cols,
                                    index=features.index)
            explained = self.pca_transformer.explained_variance_ratio_.sum()
            logger.info(f"PCA: {n_components} components, "
                        f"{explained:.2%} variance explained")

        # Recombine
        result = features.copy()
        result[self.target_column] = target.values[:len(result)]
        return result

    def _apply_windowing(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        window_size: int,
        stride: int,
        agg: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert time-series features into tabular instances via sliding windows.

        Args:
            features: Raw feature DataFrame.
            target: Label Series.
            window_size: Number of timesteps per window.
            stride: Step between consecutive windows.
            agg: Aggregation method — 'stats', 'flatten', or 'last'.

        Returns:
            Aggregated features DataFrame and corresponding labels.
        """
        logger.info(f"Applying windowing: size={window_size}, stride={stride}, agg={agg}")

        n_samples = len(features)
        indices = range(0, n_samples - window_size + 1, stride)

        if agg == "flatten":
            # Flatten all timesteps into a single row
            rows = []
            labels = []
            cols = []
            for i, idx in enumerate(indices):
                window = features.iloc[idx:idx + window_size]
                row = window.values.flatten()
                rows.append(row)
                # Label: majority vote within window
                window_labels = target.iloc[idx:idx + window_size]
                labels.append(window_labels.mode().iloc[0])
                if i == 0:
                    cols = [f"{c}_t{t}" for t in range(window_size)
                            for c in features.columns]
            result_features = pd.DataFrame(rows, columns=cols)
            result_labels = pd.Series(labels, name=self.target_column)

        elif agg == "stats":
            # Statistical aggregation per feature
            rows = []
            labels = []
            for idx in indices:
                window = features.iloc[idx:idx + window_size]
                stats = {}
                for col in features.columns:
                    vals = window[col].values.astype(float)
                    stats[f"{col}_mean"] = np.mean(vals)
                    stats[f"{col}_std"] = np.std(vals)
                    stats[f"{col}_min"] = np.min(vals)
                    stats[f"{col}_max"] = np.max(vals)
                    stats[f"{col}_delta"] = vals[-1] - vals[0]
                rows.append(stats)
                window_labels = target.iloc[idx:idx + window_size]
                # If any attack in window, label as attack (conservative)
                if (window_labels != self.config.get("_normal_label", "Normal")).any():
                    labels.append(window_labels[window_labels != self.config.get("_normal_label", "Normal")].iloc[0])
                else:
                    labels.append(window_labels.iloc[0])
            result_features = pd.DataFrame(rows)
            result_labels = pd.Series(labels, name=self.target_column)

        elif agg == "last":
            # Just take the last timestep (baseline, no aggregation benefit)
            rows = []
            labels = []
            for idx in indices:
                rows.append(features.iloc[idx + window_size - 1].values)
                labels.append(target.iloc[idx + window_size - 1])
            result_features = pd.DataFrame(rows, columns=features.columns)
            result_labels = pd.Series(labels, name=self.target_column)

        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        logger.info(f"Windowing: {n_samples} → {len(result_features)} samples, "
                     f"{result_features.shape[1]} features")
        return result_features, result_labels

    def get_features_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) from loaded data."""
        self._check_loaded()
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def train_test_split(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Fraction for test set.
            random_state: Random seed.
            stratify: Whether to stratify by label.

        Returns:
            X_train, X_test, y_train, y_test
        """
        self._check_loaded()
        X, y = self.get_features_and_labels()

        stratify_col = y if stratify else None

        # Check minimum samples per class for stratification
        if stratify:
            min_count = y.value_counts().min()
            if min_count < 2:
                logger.warning("Some classes have <2 samples. Disabling stratification.")
                stratify_col = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )

        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def subsample(
        self,
        k: int,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Subsample k instances per class (few-shot).

        Args:
            k: Number of samples per class.
            random_state: Random seed.

        Returns:
            X_sub, y_sub
        """
        self._check_loaded()
        X, y = self.get_features_and_labels()

        rng = np.random.RandomState(random_state)
        indices = []

        for label in y.unique():
            label_indices = y[y == label].index.tolist()
            n_available = len(label_indices)
            n_sample = min(k, n_available)
            if n_sample < k:
                logger.warning(f"Class '{label}' has only {n_available} samples "
                               f"(requested {k}). Using all available.")
            chosen = rng.choice(label_indices, size=n_sample, replace=False)
            indices.extend(chosen)

        rng.shuffle(indices)
        return X.loc[indices], y.loc[indices]

    def get_class_distribution(self) -> pd.Series:
        """Return value counts of target column."""
        self._check_loaded()
        return self.data[self.target_column].value_counts()

    def get_info(self) -> Dict:
        """Return dataset metadata."""
        self._check_loaded()
        X, y = self.get_features_and_labels()
        return {
            "name": self.name,
            "n_samples": len(self.data),
            "n_features": X.shape[1],
            "n_classes": y.nunique(),
            "classes": y.unique().tolist(),
            "class_distribution": y.value_counts().to_dict(),
            "imbalance_ratio": y.value_counts().max() / y.value_counts().min(),
            "config": self.config,
        }

    def save_processed(self, path: Optional[str] = None) -> str:
        """Save processed dataset to CSV."""
        self._check_loaded()
        if path is None:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            path = os.path.join(PROCESSED_DATA_DIR, f"{self.name}_processed.csv")
        self.data.to_csv(path, index=False)
        logger.info(f"Saved processed data to {path}")
        return path

    def _check_loaded(self):
        if not self._is_loaded or self.data is None:
            raise RuntimeError(f"Dataset '{self.name}' not loaded. Call .load() first.")

    def __repr__(self):
        if self._is_loaded:
            return (f"{self.__class__.__name__}(name='{self.name}', "
                    f"samples={len(self.data)}, "
                    f"features={self.data.shape[1]-1})")
        return f"{self.__class__.__name__}(name='{self.name}', loaded=False)"
