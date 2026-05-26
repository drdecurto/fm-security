"""
SWaT (Secure Water Treatment) dataset loader.

Dataset from iTrust, SUTD Singapore. Contains 11 days of continuous operation
of a scaled-down water treatment plant with 41 attack scenarios.

Reference:
    Mathur, A.P. and Tippenhauer, N.O., 2016. SWaT: a water treatment testbed
    for research and training on ICS security. CySWATER@CPSWeek.

Access: https://itrust.sutd.edu.sg/itrust-labs_datasets/
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data._base import BaseDataset
from src.data.config import (
    SWAT_RAW_DIR, SWAT_NORMAL_FILE, SWAT_ATTACK_FILE,
    SWAT_TARGET_COLUMN, SWAT_DROP_COLUMNS, SWAT_LABEL_MAP,
    SWAT_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)


class SWaTDataset(BaseDataset):
    """
    SWaT dataset for water treatment ICS intrusion detection.

    The dataset contains 51 features from sensors and actuators monitoring
    a 6-stage water treatment process: raw water intake, pre-treatment,
    ultrafiltration, dechlorination, reverse osmosis, and backwash.

    Features include:
        - LIT (Level Indicator Transmitter): Tank levels
        - FIT (Flow Indicator Transmitter): Flow rates
        - AIT (Analyzer Indicator Transmitter): Water quality
        - PIT (Pressure Indicator Transmitter): Pressures
        - DPIT (Differential Pressure): Filter pressures
        - MV (Motorized Valve): Valve states
        - P (Pump): Pump states

    Args:
        window_size: Number of timesteps per window (1 = raw samples).
        window_stride: Step between consecutive windows.
        window_agg: Aggregation method ('stats', 'flatten', 'last').
        pca: Whether to apply PCA dimensionality reduction.
        pca_components: Number of PCA components.
        normalize: Whether to StandardScale features.
        binary: If True, binary classification (Normal/Attack).
    """

    def __init__(
        self,
        window_size: int = 1,
        window_stride: int = 1,
        window_agg: str = "stats",
        pca: bool = False,
        pca_components: int = 25,
        normalize: bool = True,
        binary: bool = True,
        **kwargs,
    ):
        config = SWAT_DEFAULT_CONFIG.copy()
        config.update({
            "window_size": window_size,
            "window_stride": window_stride,
            "window_agg": window_agg,
            "pca": pca,
            "pca_components": pca_components,
            "normalize": normalize,
            "binary": binary,
            "_normal_label": "Normal",
        })
        config.update(kwargs)
        super().__init__(name="swat", config=config)
        self.target_column = SWAT_TARGET_COLUMN

    def load(self) -> None:
        """Load and preprocess SWaT dataset."""
        logger.info("Loading SWaT dataset...")

        df = self._load_raw()
        logger.info(f"Raw data shape: {df.shape}")

        df = self._clean(df)
        logger.info(f"Cleaned data shape: {df.shape}")
        logger.info(f"Class distribution:\n{df[self.target_column].value_counts()}")

        self.data = self.preprocess(df)
        self._is_loaded = True

        logger.info(f"Final dataset: {self.data.shape[0]} samples, "
                     f"{self.data.shape[1]-1} features")

    def _load_raw(self) -> pd.DataFrame:
        """Load normal and attack CSV files."""
        normal_path = os.path.join(SWAT_RAW_DIR, SWAT_NORMAL_FILE)
        attack_path = os.path.join(SWAT_RAW_DIR, SWAT_ATTACK_FILE)

        if not os.path.exists(normal_path):
            raise FileNotFoundError(
                f"SWaT normal data not found at {normal_path}. "
                f"Download from https://itrust.sutd.edu.sg/itrust-labs_datasets/ "
                f"and place in {SWAT_RAW_DIR}/"
            )
        if not os.path.exists(attack_path):
            raise FileNotFoundError(
                f"SWaT attack data not found at {attack_path}. "
                f"Download from https://itrust.sutd.edu.sg/itrust-labs_datasets/ "
                f"and place in {SWAT_RAW_DIR}/"
            )

        logger.info(f"Reading normal data: {normal_path}")
        df_normal = pd.read_csv(normal_path, sep=";", decimal=",")

        logger.info(f"Reading attack data: {attack_path}")
        df_attack = pd.read_csv(attack_path, sep=";", decimal=",")

        # SWaT CSVs sometimes use semicolons and commas for decimals
        # Try standard CSV if semicolon parse fails
        if df_normal.shape[1] <= 2:
            logger.info("Retrying with standard CSV format...")
            df_normal = pd.read_csv(normal_path)
            df_attack = pd.read_csv(attack_path)

        df = pd.concat([df_normal, df_attack], ignore_index=True)
        logger.info(f"Combined: {len(df_normal)} normal + {len(df_attack)} attack "
                     f"= {len(df)} total")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean SWaT data: strip whitespace, map labels, drop non-features."""
        # Strip column name whitespace
        df.columns = df.columns.str.strip()

        # Ensure target column exists
        if self.target_column not in df.columns:
            # Try common alternative names
            alternatives = ["Normal/Attack", "label", "Label", "Attack"]
            for alt in alternatives:
                alt_stripped = [c for c in df.columns if c.strip() == alt]
                if alt_stripped:
                    self.target_column = alt_stripped[0]
                    break
            else:
                raise KeyError(
                    f"Target column '{SWAT_TARGET_COLUMN}' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )

        # Strip label whitespace and map
        df[self.target_column] = df[self.target_column].astype(str).str.strip()
        df[self.target_column] = df[self.target_column].map(
            lambda x: SWAT_LABEL_MAP.get(x, x)
        )

        # Drop non-feature columns
        drop_cols = [c for c in SWAT_DROP_COLUMNS if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info(f"Dropped columns: {drop_cols}")

        # Convert feature columns to numeric
        feature_cols = [c for c in df.columns if c != self.target_column]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in features
        n_before = len(df)
        df = df.dropna(subset=feature_cols)
        if len(df) < n_before:
            logger.info(f"Dropped {n_before - len(df)} rows with NaN features")

        return df

    def get_sensor_groups(self) -> dict:
        """Return feature columns grouped by sensor type."""
        if not self._is_loaded:
            return {}

        features = [c for c in self.data.columns if c != self.target_column]
        groups = {
            "LIT": [c for c in features if "LIT" in c.upper()],
            "FIT": [c for c in features if "FIT" in c.upper()],
            "AIT": [c for c in features if "AIT" in c.upper()],
            "PIT": [c for c in features if "PIT" in c.upper()],
            "DPIT": [c for c in features if "DPIT" in c.upper()],
            "MV": [c for c in features if "MV" in c.upper()],
            "P": [c for c in features if c.upper().startswith("P") and
                  not c.upper().startswith("PI")],
        }
        return {k: v for k, v in groups.items() if v}
