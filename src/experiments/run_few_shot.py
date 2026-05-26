"""
Experiment 2: Few-Shot Intrusion Detection.

Evaluates how TabICL and TabPFN perform with very limited labeled data,
the critical scenario for real-world OT/ICS deployments.

Usage:
    python -m src.experiments.run_few_shot --dataset swat --shots 5 10 25 50 100 500
    python -m src.experiments.run_few_shot --dataset hai --n_runs 10
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import SWaTDataset, HAIDataset, WUSTLIIoTDataset
from src.models import (
    TabICLModel, TabPFNModel,
    RandomForestModel, XGBoostModel, LightGBMModel,
)
from src.evaluation import FewShotComparison, plot_few_shot_curves

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


def main():
    parser = argparse.ArgumentParser(description="Few-shot evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_MAP.keys()))
    parser.add_argument("--shots", nargs="+", type=int,
                        default=[5, 10, 25, 50, 100, 500],
                        help="k values (samples per class)")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--window_size", type=int, default=1)

    args = parser.parse_args()

    # Load dataset
    DatasetClass = DATASET_MAP[args.dataset]
    dataset = DatasetClass(window_size=args.window_size, binary=args.binary)
    dataset.load()

    X_train, X_test, y_train, y_test = dataset.train_test_split(
        test_size=args.test_size, random_state=42,
    )

    logger.info(f"Train pool: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"k values: {args.shots}")

    # Define models
    model_factories = {
        "TabICL": lambda: TabICLModel(),
        "TabPFN": lambda: TabPFNModel(),
        "RandomForest": lambda: RandomForestModel(),
        "XGBoost": lambda: XGBoostModel(),
        "LightGBM": lambda: LightGBMModel(),
    }

    # Run few-shot comparison
    comparison = FewShotComparison(
        model_factories=model_factories,
        k_values=args.shots,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
    )

    results_df = comparison.run(
        X_train, y_train, X_test, y_test,
        dataset_name=args.dataset,
    )

    logger.info(f"\n{results_df.to_string(index=False)}")

    # Plot learning curves
    try:
        for metric, ylabel in [
            ("F1_macro", "F1-macro"),
            ("Accuracy", "Accuracy"),
            ("DR", "Detection Rate"),
        ]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in results_df.columns:
                fig = plot_few_shot_curves(
                    results_df,
                    metric_mean=mean_col,
                    metric_std=std_col,
                    title=f"Few-Shot {ylabel} — {args.dataset.upper()}",
                    ylabel=ylabel,
                    output_path=os.path.join(args.output_dir,
                                             f"few_shot_{args.dataset}_{metric}.png"),
                )
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info("Few-shot evaluation complete.")


if __name__ == "__main__":
    main()
