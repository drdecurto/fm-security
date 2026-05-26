"""
TabICL trainable-wrapper module.

Provides the SklearnTrainableModel-compatible TabICLModel that experiments
import from `src.models.trainable.tabicl`. The wrapper accepts a config
dictionary at construction time and exposes the standard fit / predict /
predict_proba API expected by the evaluation pipeline.

Reference:
    Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025).
    TabICL: A Tabular Foundation Model for In-Context Learning on
    Large Data. ICML 2025.
"""

import os
import logging

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier

from src.models._base import SklearnTrainableModel
from src.models.config import TABICL_CONFIG, TABICL_PARAMS, TABICL_SAVING_PATH

logger = logging.getLogger(__name__)


class TabICLModel(SklearnTrainableModel):
    """
    TabICL classifier wrapped for the OT/ICS benchmarking package.

    Hyper-parameters are drawn from `TABICL_CONFIG` in `src.models.config`.
    To override per-experiment, pass a custom config dict at construction.

    Key properties:
        - No gradient-based training (in-context inference at predict time).
        - Robust to feature-scale heterogeneity typical of industrial sensors.
        - Multi-class support via class_shuffle_method="shift" and
          support_many_classes=True (TabICL v2.x).
    """

    model: TabICLClassifier

    def __init__(self, name: str = "tabicl", config: dict = None):
        cfg = dict(TABICL_CONFIG)
        if config:
            cfg.update(config)
        model = TabICLClassifier(**cfg)
        super().__init__(name=name, model=model)
        self.label_encoder = LabelEncoder()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit TabICL on (x_train, y_train).

        Note: TabICL stores the context internally; no gradient updates occur.
        """
        logger.info("Fitting TabICL model...")
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels for x in batches of `TABICL_PARAMS['predicting_batch_size']`.

        Returns:
            np.ndarray of predicted class labels.
        """
        batch_size = TABICL_PARAMS.get("predicting_batch_size", 50000)
        if batch_size is None or batch_size < 0 or len(x) <= batch_size:
            return self.model.predict(x)

        preds = []
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            preds.append(self.model.predict(x[start:stop]))
        return np.concatenate(preds, axis=0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for x (batched)."""
        batch_size = TABICL_PARAMS.get("predicting_batch_size", 50000)
        if batch_size is None or batch_size < 0 or len(x) <= batch_size:
            return self.model.predict_proba(x)

        proba_chunks = []
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            proba_chunks.append(self.model.predict_proba(x[start:stop]))
        return np.concatenate(proba_chunks, axis=0)

    def save(self, path: str = None) -> str:
        """Persist the fitted TabICL classifier to disk."""
        if path is None:
            os.makedirs(TABICL_SAVING_PATH, exist_ok=True)
            path = os.path.join(TABICL_SAVING_PATH, f"{self.name}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"TabICL model saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Restore a TabICL classifier from disk."""
        self.model = joblib.load(path)
        logger.info(f"TabICL model loaded from {path}")
