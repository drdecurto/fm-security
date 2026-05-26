"""
Evaluation pipeline for running models on OT/ICS datasets.

Handles the full evaluate loop: fit, predict, compute metrics,
and optionally repeat across multiple runs for statistical robustness.
"""

import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models._base import BaseModel
from src.evaluation.metrics import compute_metrics, aggregate_runs

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Pipeline for evaluating a single model on a dataset split.

    Usage:
        pipeline = EvaluationPipeline(model)
        results = pipeline.evaluate(X_train, y_train, X_test, y_test)
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        normal_label: str = "Normal",
    ) -> Dict:
        """
        Run single evaluation: fit + predict + metrics.

        Returns:
            Dictionary with all metrics plus timing information.
        """
        logger.info(f"=== Evaluating {self.model.name} ===")
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Fit
        t0 = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Predict
        t0 = time.time()
        y_pred = self.model.predict(X_test)
        predict_time = time.time() - t0

        # Probabilities (if available)
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_test)
            except Exception:
                pass

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba=y_proba,
                                  normal_label=normal_label)

        metrics["model_name"] = self.model.name
        metrics["fit_time_seconds"] = fit_time
        metrics["predict_time_seconds"] = predict_time
        metrics["n_train"] = len(y_train)
        metrics["n_test"] = len(y_test)

        logger.info(f"[{self.model.name}] Accuracy: {metrics['accuracy']:.4f}, "
                     f"F1-macro: {metrics['f1_macro']:.4f}, "
                     f"FAR: {metrics['false_alarm_rate']:.4f}, "
                     f"DR: {metrics['detection_rate']:.4f}")
        logger.info(f"[{self.model.name}] Fit: {fit_time:.2f}s, "
                     f"Predict: {predict_time:.2f}s")

        return metrics


class MultiRunPipeline:
    """
    Run evaluation across multiple random splits for statistical robustness.

    Usage:
        pipeline = MultiRunPipeline(dataset, model, n_runs=5)
        results = pipeline.run()
    """

    def __init__(
        self,
        dataset,
        model_factory,
        n_runs: int = 5,
        test_size: float = 0.3,
        normal_label: str = "Normal",
    ):
        """
        Args:
            dataset: Loaded BaseDataset instance.
            model_factory: Callable that returns a fresh model instance.
            n_runs: Number of evaluation runs.
            test_size: Test split fraction.
            normal_label: Label for normal class.
        """
        self.dataset = dataset
        self.model_factory = model_factory
        self.n_runs = n_runs
        self.test_size = test_size
        self.normal_label = normal_label

    def run(self) -> Dict:
        """
        Execute multiple evaluation runs.

        Returns:
            Aggregated metrics (mean ± std) and individual run results.
        """
        all_results = []

        for i in range(self.n_runs):
            logger.info(f"\n--- Run {i+1}/{self.n_runs} ---")

            X_train, X_test, y_train, y_test = self.dataset.train_test_split(
                test_size=self.test_size,
                random_state=42 + i,
            )

            model = self.model_factory()
            pipeline = EvaluationPipeline(model)
            result = pipeline.evaluate(
                X_train, y_train, X_test, y_test,
                normal_label=self.normal_label,
            )
            all_results.append(result)

        aggregated = aggregate_runs(all_results)
        aggregated["individual_runs"] = all_results
        aggregated["model_name"] = all_results[0]["model_name"]
        aggregated["dataset_name"] = self.dataset.name

        logger.info(f"\n=== {aggregated['model_name']} on {self.dataset.name} ===")
        logger.info(f"Accuracy: {aggregated.get('accuracy_mean', 0):.4f} "
                     f"± {aggregated.get('accuracy_std', 0):.4f}")
        logger.info(f"F1-macro: {aggregated.get('f1_macro_mean', 0):.4f} "
                     f"± {aggregated.get('f1_macro_std', 0):.4f}")

        return aggregated


class ModelComparisonPipeline:
    """
    Compare multiple models on the same dataset.

    Usage:
        pipeline = ModelComparisonPipeline(dataset, model_factories)
        results = pipeline.run()
    """

    def __init__(
        self,
        dataset,
        model_factories: Dict[str, callable],
        n_runs: int = 5,
        test_size: float = 0.3,
        output_dir: str = "results",
    ):
        self.dataset = dataset
        self.model_factories = model_factories
        self.n_runs = n_runs
        self.test_size = test_size
        self.output_dir = output_dir

    def run(self) -> pd.DataFrame:
        """
        Run all models and return comparison DataFrame.
        """
        all_results = []

        for model_name, factory in self.model_factories.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*60}")

            try:
                multi_pipeline = MultiRunPipeline(
                    self.dataset, factory,
                    n_runs=self.n_runs,
                    test_size=self.test_size,
                )
                result = multi_pipeline.run()
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                all_results.append({"model_name": model_name, "error": str(e)})

        # Build comparison table
        rows = []
        for r in all_results:
            if "error" in r:
                rows.append({"Model": r["model_name"], "Error": r["error"]})
            else:
                rows.append({
                    "Model": r["model_name"],
                    "Accuracy": f"{r.get('accuracy_mean',0):.4f} ± {r.get('accuracy_std',0):.4f}",
                    "F1-macro": f"{r.get('f1_macro_mean',0):.4f} ± {r.get('f1_macro_std',0):.4f}",
                    "F1-weighted": f"{r.get('f1_weighted_mean',0):.4f} ± {r.get('f1_weighted_std',0):.4f}",
                    "FAR": f"{r.get('false_alarm_rate_mean',0):.4f} ± {r.get('false_alarm_rate_std',0):.4f}",
                    "DR": f"{r.get('detection_rate_mean',0):.4f} ± {r.get('detection_rate_std',0):.4f}",
                    "MCC": f"{r.get('mcc_mean',0):.4f} ± {r.get('mcc_std',0):.4f}",
                    "Fit(s)": f"{r.get('fit_time_seconds_mean',0):.2f}",
                    "Pred(s)": f"{r.get('predict_time_seconds_mean',0):.2f}",
                })

        comparison_df = pd.DataFrame(rows)

        # Save results
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir,
                                f"comparison_{self.dataset.name}.csv")
        comparison_df.to_csv(csv_path, index=False)

        json_path = os.path.join(self.output_dir,
                                 f"comparison_{self.dataset.name}_full.json")
        # Serialize (skip non-serializable items)
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if k == "individual_runs":
                    continue
                if isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    sr[k] = float(v)
                else:
                    try:
                        json.dumps(v)
                        sr[k] = v
                    except (TypeError, ValueError):
                        sr[k] = str(v)
            serializable.append(sr)

        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"\nResults saved to {csv_path}")
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df
