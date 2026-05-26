"""
Baseline ML classifiers for comparison with foundation models.

Includes Random Forest, XGBoost, LightGBM, KNN, and Decision Tree.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models._base import BaseModel
from src.models.config import (
    RANDOM_FOREST_CONFIG, XGBOOST_CONFIG, LIGHTGBM_CONFIG,
    KNN_CONFIG, DECISION_TREE_CONFIG,
)

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest baseline."""

    def __init__(self, name: str = "RandomForest", **kwargs):
        super().__init__(name=name)
        config = RANDOM_FOREST_CONFIG.copy()
        config.update(kwargs)
        self.model = RandomForestClassifier(**config)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        logger.info(f"[{self.name}] Fitting on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost baseline."""

    def __init__(self, name: str = "XGBoost", **kwargs):
        super().__init__(name=name)
        config = XGBOOST_CONFIG.copy()
        config.update(kwargs)
        self._config = config
        self.model = None

    def fit(self, X_train, y_train):
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)

        logger.info(f"[{self.name}] Fitting on {X_train.shape[0]} samples...")
        self.model = XGBClassifier(**self._config)
        self.model.fit(X_train, y_encoded)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = self.model.predict(X)
        return self._le.inverse_transform(y_pred)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """LightGBM baseline."""

    def __init__(self, name: str = "LightGBM", **kwargs):
        super().__init__(name=name)
        config = LIGHTGBM_CONFIG.copy()
        config.update(kwargs)
        self._config = config
        self.model = None

    def fit(self, X_train, y_train):
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import LabelEncoder

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)

        logger.info(f"[{self.name}] Fitting on {X_train.shape[0]} samples...")
        self.model = LGBMClassifier(**self._config)
        self.model.fit(X_train, y_encoded)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = self.model.predict(X)
        return self._le.inverse_transform(y_pred)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)


class KNNModel(BaseModel):
    """K-Nearest Neighbors baseline."""

    def __init__(self, name: str = "KNN", **kwargs):
        super().__init__(name=name)
        config = KNN_CONFIG.copy()
        config.update(kwargs)
        self.model = KNeighborsClassifier(**config)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        logger.info(f"[{self.name}] Fitting on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


class DecisionTreeModel(BaseModel):
    """Decision Tree baseline."""

    def __init__(self, name: str = "DecisionTree", **kwargs):
        super().__init__(name=name)
        config = DECISION_TREE_CONFIG.copy()
        config.update(kwargs)
        self.model = DecisionTreeClassifier(**config)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        logger.info(f"[{self.name}] Fitting on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


def get_all_models() -> dict:
    """Return dictionary of all available models."""
    return {
        "TabICL": lambda: __import__("src.models.tabicl_model", fromlist=["TabICLModel"]).TabICLModel(),
        "TabPFN": lambda: __import__("src.models.tabpfn_model", fromlist=["TabPFNModel"]).TabPFNModel(),
        "RandomForest": lambda: RandomForestModel(),
        "XGBoost": lambda: XGBoostModel(),
        "LightGBM": lambda: LightGBMModel(),
        "KNN": lambda: KNNModel(),
        "DecisionTree": lambda: DecisionTreeModel(),
    }
