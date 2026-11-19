"""
Multi-class Confusion Matrix and metrics summary
mirroring com.trafficsign.evaluation.ConfusionMatrix.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class PerClassMetric:
    class_id: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricsSummary:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_samples: int
    per_class_metrics: List[PerClassMetric]


class ConfusionMatrix:
    """
    Confusion Matrix implementation for multi-class classification evaluation.
    Computes Overall Accuracy, Per-Class Precision, Recall, and Macro-Averaged F1 score.
    """

    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError(f"Number of classes must be positive: {num_classes}")
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_samples = 0

    def record(self, true_class: int, predicted_class: int) -> None:
        self._check_class_bounds(true_class)
        self._check_class_bounds(predicted_class)
        self.matrix[true_class, predicted_class] += 1
        self.total_samples += 1

    def record_batch(self, true_classes: Sequence[int], predicted_classes: Sequence[int]) -> None:
        if true_classes is None or predicted_classes is None:
            raise ValueError("Arrays must not be null")
        if len(true_classes) != len(predicted_classes):
            raise ValueError(f"Batch length mismatch: {len(true_classes)} vs {len(predicted_classes)}")
        for y_t, y_p in zip(true_classes, predicted_classes):
            self.record(y_t, y_p)

    def compute_summary(self) -> MetricsSummary:
        if self.total_samples == 0:
            return MetricsSummary(0.0, 0.0, 0.0, 0.0, 0, [])

        correct = int(np.trace(self.matrix))
        accuracy = float(correct / self.total_samples)

        per_class: List[PerClassMetric] = []
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1 = 0.0

        for c in range(self.num_classes):
            tp = int(self.matrix[c, c])
            support = int(np.sum(self.matrix[c, :]))
            pred_count = int(np.sum(self.matrix[:, c]))

            precision = (tp / pred_count) if pred_count > 0 else 0.0
            recall = (tp / support) if support > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1

            per_class.append(PerClassMetric(
                class_id=c,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support
            ))

        macro_precision = float(sum_precision / self.num_classes)
        macro_recall = float(sum_recall / self.num_classes)
        macro_f1 = float(sum_f1 / self.num_classes)

        return MetricsSummary(
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            total_samples=self.total_samples,
            per_class_metrics=per_class
        )

    def _check_class_bounds(self, c: int) -> None:
        if c < 0 or c >= self.num_classes:
            raise IndexError(f"Class index {c} out of bounds for numClasses={self.num_classes}")

    def get_matrix(self) -> np.ndarray:
        return self.matrix.copy()
