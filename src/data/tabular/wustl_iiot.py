"""
WUSTL-IIoT-2021 dataset loader.

IIoT dataset with Modbus/TCP protocol traffic bridging IT and OT layers.

Reference:
    Zolanvari, M. et al., 2021. WUSTL-IIoT-2021 Dataset for IIoT
    Threat Detection. Washington University in St. Louis.

Access: https://www.cse.wustl.edu/~jain/iiot2/index.html
"""

import os
import glob
import logging

import numpy as np
import pandas as pd

from src.data._base import BaseDataset
from src.data.config import (
    WUSTL_RAW_DIR, WUSTL_TARGET_COLUMN, WUSTL_DROP_COLUMNS,
    WUSTL_LABEL_MAP, WUSTL_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)


class WUSTLIIoTDataset(BaseDataset):
    """
    WUSTL-IIoT-2021 dataset for IIoT intrusion detection.

    Contains both IT network layer and OT process layer features
    with Modbus/TCP protocol traffic. Supports multi-class classification
    with attack types: DoS, Reconnaissance, MitM, Injection, Backdoor.

    Args:
        pca: Whether to apply PCA.
        normalize: Whether to normalize features.
        binary: Binary or multi-class classification.
    """

    def __init__(
        self,
        pca: bool = False,
        pca_components: int = 20,
        normalize: bool = True,
        binary: bool = False,
        **kwargs,
    ):
        config = WUSTL_DEFAULT_CONFIG.copy()
        config.update({
            "pca": pca,
            "pca_components": pca_components,
            "normalize": normalize,
            "binary": binary,
            "_normal_label": "Normal",
        })
        config.update(kwargs)
        super().__init__(name="wustl_iiot", config=config)
        self.target_column = WUSTL_TARGET_COLUMN

    def load(self) -> None:
        """Load and preprocess WUSTL-IIoT dataset."""
        logger.info("Loading WUSTL-IIoT-2021 dataset...")

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
        """Load WUSTL CSV files."""
        if not os.path.exists(WUSTL_RAW_DIR):
            raise FileNotFoundError(
                f"WUSTL-IIoT data not found at {WUSTL_RAW_DIR}. "
                f"Download from https://www.cse.wustl.edu/~jain/iiot2/index.html "
                f"and place in {WUSTL_RAW_DIR}/"
            )

        csv_files = sorted(glob.glob(os.path.join(WUSTL_RAW_DIR, "*.csv")))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(WUSTL_RAW_DIR, "**", "*.csv"),
                                         recursive=True))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {WUSTL_RAW_DIR}")

        logger.info(f"Found {len(csv_files)} CSV files")
        dfs = []
        for f in csv_files:
            logger.info(f"  Loading {os.path.basename(f)}...")
            df = pd.read_csv(f)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean WUSTL data."""
        df.columns = df.columns.str.strip()

        # Find target column
        if self.target_column not in df.columns:
            candidates = ["type", "Type", "label", "Label", "class", "Class"]
            for c in candidates:
                if c in df.columns:
                    self.target_column = c
                    break

        # Map labels
        df[self.target_column] = df[self.target_column].astype(str).str.strip().str.lower()
        df[self.target_column] = df[self.target_column].map(
            lambda x: WUSTL_LABEL_MAP.get(x, x.title())
        )

        if self.config.get("binary", False):
            df[self.target_column] = df[self.target_column].apply(
                lambda x: "Normal" if x == "Normal" else "Attack"
            )

        # Drop specified columns
        drop_cols = [c for c in WUSTL_DROP_COLUMNS if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Convert to numeric
        feature_cols = [c for c in df.columns if c != self.target_column]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=feature_cols, how="all")

        return df
