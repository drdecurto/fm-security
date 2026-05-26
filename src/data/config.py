"""
Configuration for OT/ICS datasets.

Each dataset has its own configuration dictionary controlling preprocessing,
class mappings, feature selection, and windowing parameters.
"""

import os

# ============================================================================
# Global paths
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")

# ============================================================================
# SWaT Configuration
# ============================================================================

SWAT_RAW_DIR = os.path.join(RAW_DATA_DIR, "swat")

SWAT_NORMAL_FILE = "SWaT_Dataset_Normal_v1.csv"
SWAT_ATTACK_FILE = "SWaT_Dataset_Attack_v0.csv"

SWAT_TARGET_COLUMN = "Normal/Attack"

# Columns to drop (non-feature columns)
SWAT_DROP_COLUMNS = ["Timestamp"]

# Map raw labels to standardized format
SWAT_LABEL_MAP = {
    "Normal": "Normal",
    "Attack": "Attack",
    "A ttack": "Attack",  # known typo in some versions
}

# Binary classification config
SWAT_BINARY_CONFIG = {
    "normal_label": "Normal",
    "attack_label": "Attack",
}

# Multi-class: map specific attack scenarios (from metadata)
# Attack IDs from SWaT documentation — 41 distinct attack types
SWAT_MULTICLASS_ENABLED = True

SWAT_DEFAULT_CONFIG = {
    "pca": False,
    "pca_components": 25,
    "window_size": 1,        # 1 = no windowing (raw samples)
    "window_stride": 1,
    "window_agg": "stats",   # "stats" | "flatten" | "last"
    "normalize": True,
    "binary": True,           # True = binary, False = multiclass (if available)
    "drop_constant": True,
    "remove_duplicates": True,
}

# ============================================================================
# HAI Configuration
# ============================================================================

HAI_RAW_DIR = os.path.join(RAW_DATA_DIR, "hai")

HAI_TARGET_COLUMN = "Attack"

# HAI dataset versions and their file patterns
HAI_VERSIONS = {
    "1.0": {
        "train_files": ["train1.csv"],
        "test_files": ["test1.csv", "test2.csv"],
    },
    "2.0": {
        "train_files": ["train1.csv", "train2.csv"],
        "test_files": ["test1.csv", "test2.csv", "test3.csv"],
    },
    "3.0": {
        "train_files": ["train1.csv", "train2.csv", "train3.csv"],
        "test_files": ["test1.csv", "test2.csv", "test3.csv", "test4.csv"],
    },
}

HAI_DROP_COLUMNS = ["timestamp"]

# HAI attack labels: 0 = normal, >0 = attack type
HAI_BINARY_MAP = {
    0: "Normal",
}  # all non-zero mapped to "Attack"

HAI_DEFAULT_CONFIG = {
    "version": "2.0",
    "pca": False,
    "pca_components": 30,
    "window_size": 1,
    "window_stride": 1,
    "window_agg": "stats",
    "normalize": True,
    "binary": True,
    "drop_constant": True,
    "remove_duplicates": False,  # HAI has natural temporal ordering
}

# ============================================================================
# WUSTL-IIoT-2021 Configuration
# ============================================================================

WUSTL_RAW_DIR = os.path.join(RAW_DATA_DIR, "wustl_iiot")

WUSTL_TARGET_COLUMN = "type"

WUSTL_LABEL_MAP = {
    "normal": "Normal",
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
    "mitm": "MitM",
    "injection": "Injection",
    "backdoor": "Backdoor",
}

WUSTL_DROP_COLUMNS = []

WUSTL_DEFAULT_CONFIG = {
    "pca": False,
    "pca_components": 20,
    "window_size": 1,
    "window_stride": 1,
    "window_agg": "stats",
    "normalize": True,
    "binary": False,
    "drop_constant": True,
    "remove_duplicates": True,
}

# ============================================================================
# Preprocessing / Augmentation
# ============================================================================

SMOTE_CONFIG = {
    "class_samples_threshold": 0.1,  # augment classes with <10% of majority
    "k_neighbors": 5,
    "random_state": 42,
}

WINDOW_AGG_FEATURES = ["mean", "std", "min", "max", "median"]

# ============================================================================
# Few-shot configuration
# ============================================================================

FEW_SHOT_K_VALUES = [5, 10, 25, 50, 100, 500]
FEW_SHOT_N_RUNS = 10  # repetitions per k for statistical robustness
