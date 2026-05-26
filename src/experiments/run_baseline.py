"""
Experiment 1: Baseline Model Comparison on OT/ICS Datasets.

Compares TabICL, TabPFN, and classical ML baselines on full training data.

Usage:
    python -m src.experiments.run_baseline --dataset swat --models all
    python -m src.experiments.run_baseline --dataset hai --models tabicl tabpfn rf
    python -m src.experiments.run_baseline --dataset wustl --models all --n_runs 10
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
    KNNModel, DecisionTreeModel,
)
from src.evaluation import ModelComparisonPipeline, plot_model_comparison

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

MODEL_MAP = {
    "tabicl": ("TabICL", lambda: TabICLModel()),
    "tabpfn": ("TabPFN", lambda: TabPFNModel()),
    "rf": ("RandomForest", lambda: RandomForestModel()),
    "xgb": ("XGBoost", lambda: XGBoostModel()),
    "lgbm": ("LightGBM", lambda: LightGBMModel()),
    "knn": ("KNN", lambda: KNNModel()),
    "dt": ("DecisionTree", lambda: DecisionTreeModel()),
}


def main():
    parser = argparse.ArgumentParser(description="Baseline model comparison")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_MAP.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to evaluate (e.g., tabicl tabpfn rf xgb) or 'all'")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of evaluation runs")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Use binary classification")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--pca", action="store_true", default=False)
    parser.add_argument("--pca_components", type=int, default=25)

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    DatasetClass = DATASET_MAP[args.dataset]
    dataset = DatasetClass(
        window_size=args.window_size,
        pca=args.pca,
        pca_components=args.pca_components,
        binary=args.binary,
    )
    dataset.load()

    info = dataset.get_info()
    logger.info(f"Dataset: {info['name']}")
    logger.info(f"Samples: {info['n_samples']}, Features: {info['n_features']}")
    logger.info(f"Classes: {info['n_classes']} — {info['class_distribution']}")
    logger.info(f"Imbalance ratio: {info['imbalance_ratio']:.1f}:1")

    # Select models
    if "all" in args.models:
        model_factories = {name: factory for key, (name, factory) in MODEL_MAP.items()}
    else:
        model_factories = {}
        for m in args.models:
            if m in MODEL_MAP:
                name, factory = MODEL_MAP[m]
                model_factories[name] = factory
            else:
                logger.warning(f"Unknown model: {m}. Skipping.")

    logger.info(f"Models: {list(model_factories.keys())}")

    # Run comparison
    pipeline = ModelComparisonPipeline(
        dataset=dataset,
        model_factories=model_factories,
        n_runs=args.n_runs,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )

    results_df = pipeline.run()

    # Plot
    try:
        fig = plot_model_comparison(
            results_df, metric="F1-macro",
            title=f"Model Comparison — {args.dataset.upper()} (F1-macro)",
            output_path=os.path.join(args.output_dir,
                                     f"comparison_{args.dataset}_f1.png"),
        )
        fig = plot_model_comparison(
            results_df, metric="DR",
            title=f"Model Comparison — {args.dataset.upper()} (Detection Rate)",
            output_path=os.path.join(args.output_dir,
                                     f"comparison_{args.dataset}_dr.png"),
        )
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info("Baseline comparison complete.")


if __name__ == "__main__":
    main()
