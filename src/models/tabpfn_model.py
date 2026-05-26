"""
TabPFN (Tabular Prior-Data Fitted Network) model wrapper.

TabPFN is a transformer pretrained on synthetic tabular datasets that
performs Bayesian inference in a single forward pass.

Reference:
    Hollmann, N. et al., 2023. TabPFN: A Transformer That Solves Small
    Tabular Classification Problems in a Second. ICLR 2023.

Install: pip install tabpfn
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import os

from src.models._base import BaseModel
from src.models.config import (
    TABPFN_CONFIG, TABPFN_PARAMS, TABPFN_SAVING_PATH,
    TABPFN_LIMITS, TABPFN_MANY_CLASS_CONFIG,
)

logger = logging.getLogger(__name__)


class TabPFNModel(BaseModel):
    """
    TabPFN classifier for OT/ICS intrusion detection.

    Important constraints:
        - Default max 10,000 training samples (can be overridden)
        - Default max 100 features (use PCA for high-dim ICS data)
        - Default max 10 classes (use ManyClassClassifier wrapper)

    The model automatically handles many-class scenarios via the
    error-correcting tournament strategy when n_classes > 10.

    Args:
        n_estimators: Number of ensemble members.
        device: Compute device.
        config_overrides: Dict to override TABPFN_CONFIG.
    """

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        device: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        name: str = "TabPFN",
    ):
        super().__init__(name=name)

        self.config = TABPFN_CONFIG.copy()
        if n_estimators is not None:
            self.config["n_estimators"] = n_estimators
        if device is not None:
            self.config["device"] = device
        if config_overrides:
            self.config.update(config_overrides)

        self.batch_size = TABPFN_PARAMS["predicting_batch_size"]
        self.model = None
        self._is_many_class = False

    def _init_model(self, n_classes: int):
        """Initialize TabPFN, using ManyClassClassifier if needed."""
        from tabpfn import TabPFNClassifier

        if n_classes > TABPFN_LIMITS["max_classes"]:
            logger.info(f"Dataset has {n_classes} classes (> {TABPFN_LIMITS['max_classes']}). "
                        f"Using ManyClassClassifier wrapper.")
            try:
                from tabpfn.others.many_class_classifier import ManyClassClassifier
                base_estimator = TabPFNClassifier(**self.config)
                self.model = ManyClassClassifier(
                    estimator=base_estimator,
                    **TABPFN_MANY_CLASS_CONFIG,
                )
                self._is_many_class = True
            except ImportError:
                logger.warning("ManyClassClassifier not available. "
                               "Using standard TabPFN (may fail with >10 classes).")
                self.model = TabPFNClassifier(**self.config)
        else:
            self.model = TabPFNClassifier(**self.config)

        logger.info(f"Initialized TabPFN: n_estimators={self.config['n_estimators']}, "
                     f"many_class={self._is_many_class}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit TabPFN on training data.

        Handles sample/feature limits and automatic many-class wrapping.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        logger.info(f"[{self.name}] Fitting: {n_samples} samples, "
                     f"{n_features} features, {n_classes} classes")

        # Warn about limits
        if n_samples > TABPFN_LIMITS["max_samples"]:
            logger.warning(f"TabPFN pretrained with max {TABPFN_LIMITS['max_samples']} "
                           f"samples. Current: {n_samples}. Performance may degrade.")
        if n_features > TABPFN_LIMITS["max_features"]:
            logger.warning(f"TabPFN pretrained with max {TABPFN_LIMITS['max_features']} "
                           f"features. Current: {n_features}. Consider PCA reduction.")

        self._init_model(n_classes)
        self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Fit complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels with batch processing."""
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

        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
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
        if path is None:
            os.makedirs(TABPFN_SAVING_PATH, exist_ok=True)
            path = os.path.join(TABPFN_SAVING_PATH, f"{self.name}.joblib")
        joblib.dump({"model": self.model, "config": self.config}, path)
        logger.info(f"Saved {self.name} to {path}")
        return path

    def load(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(TABPFN_SAVING_PATH, f"{self.name}.joblib")
        data = joblib.load(path)
        self.model = data["model"]
        self.config = data["config"]
        logger.info(f"Loaded {self.name} from {path}")
