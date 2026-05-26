from src.evaluation.metrics import compute_metrics, compute_rare_class_metrics, aggregate_runs
from src.evaluation.pipeline import EvaluationPipeline, MultiRunPipeline, ModelComparisonPipeline
from src.evaluation.few_shot import FewShotEvaluator, FewShotComparison
from src.evaluation.visualization import (
    plot_model_comparison, plot_few_shot_curves,
    plot_confusion_matrix, plot_class_distribution,
    plot_per_class_f1,
)

__all__ = [
    "compute_metrics", "compute_rare_class_metrics", "aggregate_runs",
    "EvaluationPipeline", "MultiRunPipeline", "ModelComparisonPipeline",
    "FewShotEvaluator", "FewShotComparison",
    "plot_model_comparison", "plot_few_shot_curves",
    "plot_confusion_matrix", "plot_class_distribution",
    "plot_per_class_f1",
]
