"""
Model configurations for TabICL, TabPFN, and baseline classifiers.

This file consolidates per-model hyper-parameters used by the OT/ICS
benchmarking package. Hyper-parameters reflect the values reported in
the accompanying paper:

    de Curtò, J., de Zarzà, I., Cano, J. C., Calafate, C. T.
    A Comparative Study of Large Language Models for Industrial
    Cyber-Physical Security. Electronics (2026).

TabICL parameter naming follows the v2.x API. If you must use tabicl<2.0,
the four renamed parameters listed in the TABICL_CONFIG block must be
reverted to their legacy names (see comments inline).
"""

import os

# ============================================================================
# Paths
# ============================================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
TABICL_SAVING_PATH = os.path.join(MODELS_DIR, "tabicl")
TABPFN_SAVING_PATH = os.path.join(MODELS_DIR, "tabpfn")

# ============================================================================
# TabICL Configuration
# ----------------------------------------------------------------------------
# tabicl v2.x renamed several parameters silently between v0.x and v2.x.
# The mapping below documents the four renames; the dict that follows uses
# the v2.x names throughout. If you must use tabicl<2.0, revert as noted.
#
#   v0.x parameter           v2.x parameter
#   ----------------         -------------------------------
#   class_shift: True        class_shuffle_method: "shift"
#   use_hierarchical: True   support_many_classes: True
#   use_amp: True            use_amp: "auto"
#   checkpoint_version: ".../v1.1-0506.ckpt"
#                            checkpoint_version: ".../v1.1-20250506.ckpt"
# ============================================================================

TABICL_CONFIG = {
    "n_estimators": 16,
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",        # v2.x rename (was class_shift: True)
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "support_many_classes": True,            # v2.x rename (was use_hierarchical: True)
    "batch_size": 8,
    "use_amp": "auto",                       # v2.x rename (was True)
    "model_path": None,
    "allow_auto_download": True,
    "checkpoint_version": "tabicl-classifier-v1.1-20250506.ckpt",
    "device": None,
    "random_state": 42,
    "n_jobs": None,
    "verbose": False,
    "inference_config": None,
}

TABICL_PARAMS = {
    "predicting_batch_size": 50000,
}

# ============================================================================
# TabPFN Configuration
# ============================================================================

TABPFN_CONFIG = {
    "n_estimators": 4,
    "balance_probabilities": False,
    "average_before_softmax": False,
    "device": "auto",
    "ignore_pretraining_limits": False,
    "inference_precision": "auto",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": True,
    "random_state": 0,
    "n_jobs": -1,
}

# Hard limits enforced by TabPFN (per the official docs). These constants
# are referenced by `src.models.tabpfn_model` to decide when to subsample.
TABPFN_LIMITS = {
    "max_samples": 10000,
    "max_features": 500,
    "max_classes": 10,
}

# Many-class configuration used for the multi-class WUSTL evaluation.
TABPFN_MANY_CLASS_CONFIG = {
    "alphabet_size": 5,                # must be <= TABPFN_LIMITS["max_classes"]
    "n_estimators": None,              # None = automatic
    "n_estimators_redundancy": 4,
    "random_state": 0,
}

TABPFN_PARAMS = {
    "predicting_batch_size": 40000,
}

# ============================================================================
# Classical baselines
# ----------------------------------------------------------------------------
# RandomForest is the headline anchor used in the paper. XGBoost / LightGBM
# / KNN / DecisionTree are retained as reference baselines but are not part
# of the paper's primary protocol.
# ============================================================================

RANDOM_FOREST_CONFIG = {
    "n_estimators": 200,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_CONFIG = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

LIGHTGBM_CONFIG = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

KNN_CONFIG = {
    "n_neighbors": 8,
    "n_jobs": -1,
}

# Aliases for backward compatibility with earlier package versions
KNEIGHBORS_CONFIG = KNN_CONFIG

DECISION_TREE_CONFIG = {
    "max_depth": 8,
    "class_weight": "balanced",
    "random_state": 42,
}

LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 5000,
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

LINEAR_SVC_CONFIG = {
    "C": 1.0,
    "max_iter": 5000,
    "class_weight": "balanced",
    "random_state": 42,
}

# ============================================================================
# Misc
# ============================================================================

BACKOFF_MAX_TRIES = 15
BACKOFF_FACTOR = 2
