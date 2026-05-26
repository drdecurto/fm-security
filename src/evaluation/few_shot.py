"""
Few-shot evaluation for OT/ICS intrusion detection.

Evaluates model performance across different training set sizes (k-shot),
which is the key experiment for demonstrating foundation model advantages
in data-scarce industrial environments.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.preprocessing import subsample_per_class
from src.evaluation.metrics import compute_metrics, aggregate_runs

logger = logging.getLogger(__name__)


class FewShotEvaluator:
    """
    Evaluates models at different k-shot regimes.

    For each value of k, subsamples k instances per class from the training set,
    fits the model, and evaluates on the full test set. Repeats n_runs times
    for statistical robustness.

    Args:
        model_factory: Callable that returns a fresh model instance.
        k_values: List of k values to evaluate.
        n_runs: Number of repetitions per k.
        normal_label: Label for normal class.
    """

    def __init__(
        self,
        model_factory,
        k_values: List[int] = None,
        n_runs: int = 10,
        normal_label: str = "Normal",
    ):
        self.model_factory = model_factory
        self.k_values = k_values or [5, 10, 25, 50, 100, 500]
        self.n_runs = n_runs
        self.normal_label = normal_label

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """
        Run few-shot evaluation across all k values.

        Args:
            X_train: Full training features (subsampled per k).
            y_train: Full training labels.
            X_test: Test features (fixed across all k).
            y_test: Test labels.

        Returns:
            Dictionary with results per k value.
        """
        results = {}
        model_name = self.model_factory().name

        logger.info(f"\n{'='*60}")
        logger.info(f"Few-Shot Evaluation: {model_name}")
        logger.info(f"k values: {self.k_values}, n_runs: {self.n_runs}")
        logger.info(f"Train pool: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"{'='*60}")

        for k in self.k_values:
            logger.info(f"\n--- k = {k} ---")
            k_results = []

            for run in range(self.n_runs):
                # Subsample training set
                X_sub, y_sub = subsample_per_class(
                    X_train, y_train, k=k, random_state=42 + run,
                )

                actual_k = y_sub.value_counts().to_dict()
                if run == 0:
                    logger.info(f"Actual samples per class: {actual_k}")

                # Fit and predict
                model = self.model_factory()
                try:
                    model.fit(X_sub.values, y_sub.values)
                    y_pred = model.predict(X_test.values)

                    metrics = compute_metrics(
                        y_test.values, y_pred,
                        normal_label=self.normal_label,
                    )
                    metrics["k"] = k
                    metrics["run"] = run
                    metrics["actual_train_size"] = len(y_sub)
                    k_results.append(metrics)

                except Exception as e:
                    logger.warning(f"Run {run+1} failed for k={k}: {e}")
                    continue

            if k_results:
                aggregated = aggregate_runs(k_results)
                aggregated["k"] = k
                results[k] = aggregated

                logger.info(f"k={k}: Accuracy={aggregated.get('accuracy_mean',0):.4f} "
                             f"± {aggregated.get('accuracy_std',0):.4f}, "
                             f"F1={aggregated.get('f1_macro_mean',0):.4f} "
                             f"± {aggregated.get('f1_macro_std',0):.4f}")

        return {
            "model_name": model_name,
            "k_values": self.k_values,
            "n_runs": self.n_runs,
            "results_per_k": results,
        }

    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert few-shot results to a summary DataFrame."""
        rows = []
        for k, res in results["results_per_k"].items():
            rows.append({
                "Model": results["model_name"],
                "k": k,
                "Accuracy_mean": res.get("accuracy_mean", np.nan),
                "Accuracy_std": res.get("accuracy_std", np.nan),
                "F1_macro_mean": res.get("f1_macro_mean", np.nan),
                "F1_macro_std": res.get("f1_macro_std", np.nan),
                "F1_weighted_mean": res.get("f1_weighted_mean", np.nan),
                "F1_weighted_std": res.get("f1_weighted_std", np.nan),
                "FAR_mean": res.get("false_alarm_rate_mean", np.nan),
                "FAR_std": res.get("false_alarm_rate_std", np.nan),
                "DR_mean": res.get("detection_rate_mean", np.nan),
                "DR_std": res.get("detection_rate_std", np.nan),
                "MCC_mean": res.get("mcc_mean", np.nan),
                "MCC_std": res.get("mcc_std", np.nan),
            })
        return pd.DataFrame(rows)


class FewShotComparison:
    """
    Compare multiple models across k-shot regimes.

    Usage:
        comparison = FewShotComparison(model_factories, k_values=[5,10,50,100])
        results_df = comparison.run(X_train, y_train, X_test, y_test)
    """

    def __init__(
        self,
        model_factories: Dict[str, callable],
        k_values: List[int] = None,
        n_runs: int = 10,
        output_dir: str = "results",
    ):
        self.model_factories = model_factories
        self.k_values = k_values or [5, 10, 25, 50, 100, 500]
        self.n_runs = n_runs
        self.output_dir = output_dir

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str = "dataset",
    ) -> pd.DataFrame:
        """Run few-shot comparison for all models."""
        all_dfs = []

        for name, factory in self.model_factories.items():
            evaluator = FewShotEvaluator(
                model_factory=factory,
                k_values=self.k_values,
                n_runs=self.n_runs,
            )
            results = evaluator.evaluate(X_train, y_train, X_test, y_test)
            df = evaluator.to_dataframe(results)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)

        # Save
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir,
                                f"few_shot_{dataset_name}.csv")
        combined.to_csv(csv_path, index=False)
        logger.info(f"Few-shot results saved to {csv_path}")

        return combined
