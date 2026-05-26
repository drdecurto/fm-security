"""
HAI (HIL-based Augmented ICS Security) dataset loader.

Hardware-in-the-loop ICS testbed combining GE turbine, Emerson boiler,
and FESTO water treatment systems. Contains 50+ attack scenarios.

Reference:
    Shin, H.K. et al., 2020. HAI 1.0: HIL-based Augmented ICS Security
    Dataset. CSET@USENIX Security.

Access: https://github.com/icsdataset/hai
"""

import os
import glob
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from src.data._base import BaseDataset
from src.data.config import (
    HAI_RAW_DIR, HAI_TARGET_COLUMN, HAI_DROP_COLUMNS,
    HAI_BINARY_MAP, HAI_VERSIONS, HAI_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)


class HAIDataset(BaseDataset):
    """
    HAI dataset for multi-process ICS intrusion detection.

    Covers three industrial processes centered on a HIL simulator:
        - P1: GE turbine (power generation)
        - P2: Emerson boiler (heat generation)
        - P3: FESTO water treatment

    Features include process variables (temperatures, pressures, flows,
    levels, valve positions) and control signals from PLCs.

    Args:
        version: HAI version ('1.0', '2.0', '3.0').
        window_size: Timesteps per window.
        pca: Whether to apply PCA.
        binary: Binary (Normal/Attack) or multi-class.
    """

    def __init__(
        self,
        version: str = "2.0",
        window_size: int = 1,
        window_stride: int = 1,
        window_agg: str = "stats",
        pca: bool = False,
        pca_components: int = 30,
        normalize: bool = True,
        binary: bool = True,
        **kwargs,
    ):
        config = HAI_DEFAULT_CONFIG.copy()
        config.update({
            "version": version,
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
        super().__init__(name=f"hai_v{version}", config=config)
        self.target_column = HAI_TARGET_COLUMN
        self.version = version

    def load(self) -> None:
        """Load and preprocess HAI dataset."""
        logger.info(f"Loading HAI v{self.version} dataset...")

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
        """Load HAI CSV files based on version."""
        version_dir = os.path.join(HAI_RAW_DIR, f"hai-{self.version}")

        # Try flat directory structure too
        if not os.path.exists(version_dir):
            version_dir = HAI_RAW_DIR

        if not os.path.exists(version_dir):
            raise FileNotFoundError(
                f"HAI data directory not found at {version_dir}. "
                f"Download from https://github.com/icsdataset/hai "
                f"and place in {HAI_RAW_DIR}/"
            )

        # Find all CSV files
        csv_files = sorted(glob.glob(os.path.join(version_dir, "*.csv")))
        if not csv_files:
            # Try subdirectories
            csv_files = sorted(glob.glob(os.path.join(version_dir, "**", "*.csv"),
                                         recursive=True))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {version_dir}. "
                f"Expected train*.csv and test*.csv files."
            )

        logger.info(f"Found {len(csv_files)} CSV files in {version_dir}")

        dfs = []
        for f in csv_files:
            logger.info(f"  Loading {os.path.basename(f)}...")
            df = pd.read_csv(f)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean HAI data: handle labels, drop timestamps."""
        df.columns = df.columns.str.strip()

        # Find the attack/label column
        target_candidates = ["Attack", "attack", "label", "Label",
                             "attack_label", "Attack_Label"]
        found_target = None
        for cand in target_candidates:
            matches = [c for c in df.columns if c.strip().lower() == cand.lower()]
            if matches:
                found_target = matches[0]
                break

        if found_target is None:
            # HAI uses multiple attack columns (Attack_P1, Attack_P2, Attack_P3)
            attack_cols = [c for c in df.columns if "attack" in c.lower()]
            if attack_cols:
                logger.info(f"Found attack indicator columns: {attack_cols}")
                # Create combined attack label: attack if any subprocess is attacked
                df[HAI_TARGET_COLUMN] = "Normal"
                for col in attack_cols:
                    df.loc[df[col].astype(str).str.strip() != "0", HAI_TARGET_COLUMN] = "Attack"
                    df.loc[df[col].astype(str).str.strip().str.lower() == "true", HAI_TARGET_COLUMN] = "Attack"
                found_target = HAI_TARGET_COLUMN
                # Drop the individual attack columns from features
                df = df.drop(columns=attack_cols)
            else:
                raise KeyError(
                    f"No attack/label column found. "
                    f"Available columns: {df.columns.tolist()}"
                )

        if found_target != self.target_column:
            df = df.rename(columns={found_target: self.target_column})

        # Map labels
        if self.config.get("binary", True):
            # Convert to binary: 0/Normal → Normal, anything else → Attack
            def map_label(x):
                x_str = str(x).strip().lower()
                if x_str in ("0", "normal", "false", "nan", ""):
                    return "Normal"
                return "Attack"
            df[self.target_column] = df[self.target_column].apply(map_label)

        # Drop timestamp columns
        drop_cols = [c for c in df.columns
                     if c.lower() in [d.lower() for d in HAI_DROP_COLUMNS]
                     or "time" in c.lower()]
        feature_drop = [c for c in drop_cols if c != self.target_column]
        if feature_drop:
            df = df.drop(columns=feature_drop)
            logger.info(f"Dropped columns: {feature_drop}")

        # Convert features to numeric
        feature_cols = [c for c in df.columns if c != self.target_column]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=feature_cols, how="all")

        return df

    def get_process_features(self) -> dict:
        """Return features grouped by industrial process (P1, P2, P3)."""
        if not self._is_loaded:
            return {}
        features = [c for c in self.data.columns if c != self.target_column]
        return {
            "P1_Turbine": [c for c in features if c.startswith("P1")],
            "P2_Boiler": [c for c in features if c.startswith("P2")],
            "P3_Water": [c for c in features if c.startswith("P3")],
            "Control": [c for c in features
                        if not any(c.startswith(p) for p in ["P1", "P2", "P3"])],
        }
