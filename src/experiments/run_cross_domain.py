"""
Experiment 4: Cross-Domain Transfer (IT → OT).

Tests whether foundation models trained on IT IDS datasets can
generalize to OT/ICS datasets, and vice versa.

This experiment requires the IT datasets (CIC-IDS2017, N-BaIoT) from
Pablo's original framework alongside the OT datasets.

Usage:
    python -m src.experiments.run_cross_domain --source swat --target hai
    python -m src.experiments.run_cross_domain --source swat --target wustl --n_runs 5
"""

import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import SWaTDataset, HAIDataset, WUSTLIIoTDataset
from src.models import TabICLModel, TabPFNModel, RandomForestModel, XGBoostModel
from src.evaluation import EvaluationPipeline, compute_metrics, aggregate_runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "swat": SWaTDataset,
    "hai": HAIDataset,
    "wustl": WUSTLIIoTDataset,
}


def align_feature_spaces(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    n_components: int = 25,
) -> tuple:
    """
    Align source and target feature spaces via PCA projection.

    Both datasets are independently PCA-transformed to the same
    dimensionality, enabling cross-domain comparison despite
    different original feature sets.

    Args:
        X_source: Source domain features.
        X_target: Target domain features.
        n_components: Shared PCA dimensionality.

    Returns:
        (X_source_aligned, X_target_aligned) as numpy arrays.
    """
    n_components = min(n_components, X_source.shape[1], X_target.shape[1])

    # Source PCA
    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(X_source.values)
    pca_s = PCA(n_components=n_components, random_state=42)
    X_s_pca = pca_s.fit_transform(X_s)

    # Target PCA
    scaler_t = StandardScaler()
    X_t = scaler_t.fit_transform(X_target.values)
    pca_t = PCA(n_components=n_components, random_state=42)
    X_t_pca = pca_t.fit_transform(X_t)

    logger.info(f"Source PCA: {X_source.shape[1]} → {n_components} "
                f"({pca_s.explained_variance_ratio_.sum():.2%} var)")
    logger.info(f"Target PCA: {X_target.shape[1]} → {n_components} "
                f"({pca_t.explained_variance_ratio_.sum():.2%} var)")

    # Column names
    cols = [f"PC{i+1}" for i in range(n_components)]

    return (
        pd.DataFrame(X_s_pca, columns=cols),
        pd.DataFrame(X_t_pca, columns=cols),
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-domain transfer evaluation")
    parser.add_argument("--source", type=str, required=True,
                        choices=list(DATASET_MAP.keys()),
                        help="Source domain dataset (train)")
    parser.add_argument("--target", type=str, required=True,
                        choices=list(DATASET_MAP.keys()),
                        help="Target domain dataset (test)")
    parser.add_argument("--pca_components", type=int, default=25)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    # Load both datasets in binary mode
    logger.info(f"Source: {args.source}, Target: {args.target}")

    SourceClass = DATASET_MAP[args.source]
    TargetClass = DATASET_MAP[args.target]

    source_ds = SourceClass(binary=True)
    source_ds.load()

    target_ds = TargetClass(binary=True)
    target_ds.load()

    X_source, y_source = source_ds.get_features_and_labels()
    X_target, y_target = target_ds.get_features_and_labels()

    logger.info(f"Source: {X_source.shape}, classes: {y_source.value_counts().to_dict()}")
    logger.info(f"Target: {X_target.shape}, classes: {y_target.value_counts().to_dict()}")

    # Align feature spaces
    X_source_aligned, X_target_aligned = align_feature_spaces(
        X_source, X_target, n_components=args.pca_components,
    )

    # Evaluate models: train on source, test on target
    models = {
        "TabICL": lambda: TabICLModel(),
        "TabPFN": lambda: TabPFNModel(),
        "RandomForest": lambda: RandomForestModel(),
        "XGBoost": lambda: XGBoostModel(),
    }

    all_results = []
    for name, factory in models.items():
        logger.info(f"\n--- {name}: {args.source} → {args.target} ---")
        run_results = []

        for run in range(args.n_runs):
            # Subsample source for training (use different seeds)
            rng = np.random.RandomState(42 + run)
            n_train = min(5000, len(X_source_aligned))
            idx = rng.choice(len(X_source_aligned), n_train, replace=False)

            X_train = X_source_aligned.iloc[idx]
            y_train = y_source.iloc[idx]

            # Full target as test
            X_test = X_target_aligned
            y_test = y_target

            try:
                model = factory()
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)

                metrics = compute_metrics(y_test.values, y_pred)
                metrics["model"] = name
                metrics["run"] = run
                run_results.append(metrics)

            except Exception as e:
                logger.warning(f"Run {run} failed: {e}")

        if run_results:
            agg = aggregate_runs(run_results)
            agg["model"] = name
            all_results.append(agg)

            logger.info(f"{name}: Accuracy={agg.get('accuracy_mean',0):.4f}, "
                         f"F1={agg.get('f1_macro_mean',0):.4f}, "
                         f"DR={agg.get('detection_rate_mean',0):.4f}")

    # Summary
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model"],
            "Accuracy": f"{r.get('accuracy_mean',0):.4f} ± {r.get('accuracy_std',0):.4f}",
            "F1_macro": f"{r.get('f1_macro_mean',0):.4f} ± {r.get('f1_macro_std',0):.4f}",
            "FAR": f"{r.get('false_alarm_rate_mean',0):.4f} ± {r.get('false_alarm_rate_std',0):.4f}",
            "DR": f"{r.get('detection_rate_mean',0):.4f} ± {r.get('detection_rate_std',0):.4f}",
        })

    summary = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir,
                        f"cross_domain_{args.source}_to_{args.target}.csv")
    summary.to_csv(path, index=False)

    logger.info(f"\n{summary.to_string(index=False)}")
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
