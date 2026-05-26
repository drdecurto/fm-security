"""
Experiment 3: Rare Attack Class Detection.

Evaluates model performance on minority attack classes, testing whether
foundation models outperform traditional methods on long-tail distributions
typical of ICS datasets.

Usage:
    python -m src.experiments.run_rare_attack --dataset swat --threshold 50
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
from src.evaluation import (
    EvaluationPipeline, compute_rare_class_metrics,
    plot_per_class_f1,
)

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
    parser = argparse.ArgumentParser(description="Rare attack class evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_MAP.keys()))
    parser.add_argument("--threshold", type=int, default=50,
                        help="Classes with fewer samples are considered 'rare'")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--binary", action="store_true", default=False,
                        help="Must be False for rare class analysis")

    args = parser.parse_args()

    if args.binary:
        logger.warning("Rare attack analysis requires multi-class labels. "
                        "Setting binary=False.")
        args.binary = False

    # Load dataset in multi-class mode
    DatasetClass = DATASET_MAP[args.dataset]
    dataset = DatasetClass(binary=False)
    dataset.load()

    class_dist = dataset.get_class_distribution()
    rare_classes = class_dist[class_dist < args.threshold].index.tolist()
    rare_classes = [c for c in rare_classes if c != "Normal"]

    logger.info(f"Class distribution:\n{class_dist}")
    logger.info(f"Rare classes (< {args.threshold} samples): {rare_classes}")

    if not rare_classes:
        logger.info("No rare classes found. Try lowering --threshold.")
        return

    X_train, X_test, y_train, y_test = dataset.train_test_split(
        test_size=0.3, random_state=42,
    )

    # Evaluate models
    models = {
        "TabICL": TabICLModel(),
        "TabPFN": TabPFNModel(),
        "RandomForest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\n--- {name} ---")
        try:
            pipeline = EvaluationPipeline(model)
            full_metrics = pipeline.evaluate(
                X_train, y_train, X_test, y_test,
            )

            rare_metrics = compute_rare_class_metrics(
                y_test.values, model.predict(X_test.values),
                threshold=args.threshold,
            )

            results[name] = {
                "full": full_metrics,
                "rare": rare_metrics,
            }

            logger.info(f"Full F1-macro: {full_metrics['f1_macro']:.4f}")
            logger.info(f"Rare F1-macro: {rare_metrics.get('rare_f1_macro', 'N/A')}")

            # Plot per-class F1
            try:
                fig = plot_per_class_f1(
                    full_metrics["per_class"],
                    title=f"{name} — Per-Class F1 ({args.dataset.upper()})",
                    output_path=os.path.join(args.output_dir,
                                             f"rare_{args.dataset}_{name}_f1.png"),
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"{name} failed: {e}")

    # Summary table
    import pandas as pd
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Full_F1_macro": f"{res['full']['f1_macro']:.4f}",
            "Rare_F1_macro": f"{res['rare'].get('rare_f1_macro', 0):.4f}",
            "Rare_DR": f"{res['rare'].get('rare_detection_rate', 0):.4f}",
            "Full_Accuracy": f"{res['full']['accuracy']:.4f}",
        })
    summary = pd.DataFrame(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir,
                                f"rare_attack_{args.dataset}.csv"), index=False)
    logger.info(f"\n{summary.to_string(index=False)}")
    logger.info("Rare attack evaluation complete.")


if __name__ == "__main__":
    main()
