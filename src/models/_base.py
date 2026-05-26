"""
Base model class for all classifiers.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models in the framework."""

    def __init__(self, name: str):
        self.name = name
        self._fit_time: Optional[float] = None
        self._predict_time: Optional[float] = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        pass

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit on training data, then predict on test data. Returns predictions."""
        t0 = time.time()
        self.fit(X_train, y_train)
        self._fit_time = time.time() - t0
        logger.info(f"[{self.name}] Fit time: {self._fit_time:.2f}s")

        t0 = time.time()
        predictions = self.predict(X_test)
        self._predict_time = time.time() - t0
        logger.info(f"[{self.name}] Predict time: {self._predict_time:.2f}s")

        return predictions

    def get_timing(self) -> Dict[str, Optional[float]]:
        """Return fit and predict times."""
        return {
            "fit_time": self._fit_time,
            "predict_time": self._predict_time,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
