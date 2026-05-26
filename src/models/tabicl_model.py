"""
TabICL (Tabular In-Context Learning) model wrapper for ICS intrusion detection.

TabICL performs classification via in-context learning on tabular data.
No gradient-based training required — it uses a pretrained transformer
that processes (X_train, y_train, X_test) in a single forward pass.

Reference:
    Qu, X. et al., 2024. TabICL: A Tabular Foundation Model for
    In-Context Learning on Novel Table Schemas.

Install: pip install tabicl
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import os

from src.models._base import BaseModel
from src.models.config import TABICL_CONFIG, TABICL_PARAMS, TABICL_SAVING_PATH

logger = logging.getLogger(__name__)


class TabICLModel(BaseModel):
    """
    TabICL classifier for OT/ICS intrusion detection.

    Key properties for ICS use cases:
        - No training required: ideal for few-shot scenarios
        - Handles class imbalance via hierarchical classification
        - Ensembling with normalization diversity
        - Robust to feature scale differences (industrial sensors)

    Args:
        n_estimators: Number of ensemble members.
        use_hierarchical: Enable hierarchical classification (for >10 classes).
        device: Compute device ('auto', 'cpu', 'cuda').
        config_overrides: Dict to override default TABICL_CONFIG values.
    """

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        use_hierarchical: Optional[bool] = None,
        device: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        name: str = "TabICL",
    ):
        super().__init__(name=name)

        self.config = TABICL_CONFIG.copy()
        if n_estimators is not None:
            self.config["n_estimators"] = n_estimators
        if use_hierarchical is not None:
            self.config["use_hierarchical"] = use_hierarchical
        if device is not None:
            self.config["device"] = device
        if config_overrides:
            self.config.update(config_overrides)

        self.batch_size = TABICL_PARAMS["predicting_batch_size"]
        self.model = None
        self._label_classes = None

    def _init_model(self):
        """Lazily initialize the TabICL model."""
        if self.model is None:
            from tabicl import TabICLClassifier
            self.model = TabICLClassifier(**self.config)
            logger.info(f"Initialized TabICLClassifier with config: "
                        f"n_estimators={self.config['n_estimators']}, "
                        f"hierarchical={self.config['use_hierarchical']}, "
                        f"device={self.config['device']}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit TabICL on training data.

        Note: TabICL stores training data for in-context learning
        rather than performing gradient-based optimization.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
        """
        self._init_model()

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self._label_classes = np.unique(y_train)
        n_classes = len(self._label_classes)
        n_samples, n_features = X_train.shape

        logger.info(f"[{self.name}] Fitting: {n_samples} samples, "
                     f"{n_features} features, {n_classes} classes")

        if n_classes > 10 and not self.config.get("use_hierarchical", True):
            logger.warning(f"Dataset has {n_classes} classes but hierarchical mode "
                           f"is disabled. Consider enabling for better performance.")

        self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Fit complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using TabICL's in-context inference.

        Uses batch processing for large test sets to manage memory.

        Args:
            X: Test features (n_samples, n_features).

        Returns:
            Predicted labels array.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size
        batch_size = min(batch_size, n_samples)

        logger.info(f"[{self.name}] Predicting {n_samples} samples "
                     f"(batch_size={batch_size})...")

        preds = []
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            batch_preds = self.model.predict(batch)
            preds.append(batch_preds)
            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= n_samples:
                logger.info(f"  {min(i + batch_size, n_samples)}/{n_samples} predicted")

        predictions = np.concatenate(preds)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test features.

        Returns:
            Probability matrix (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size

        proba_list = []
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            proba = self.model.predict_proba(batch)
            proba_list.append(proba)

        return np.concatenate(proba_list)

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk."""
        if path is None:
            os.makedirs(TABICL_SAVING_PATH, exist_ok=True)
            path = os.path.join(TABICL_SAVING_PATH, f"{self.name}.joblib")
        joblib.dump({"model": self.model, "config": self.config}, path)
        logger.info(f"Saved {self.name} to {path}")
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk."""
        if path is None:
            path = os.path.join(TABICL_SAVING_PATH, f"{self.name}.joblib")
        data = joblib.load(path)
        self.model = data["model"]
        self.config = data["config"]
        logger.info(f"Loaded {self.name} from {path}")
