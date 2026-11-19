"""
Majority Class Baseline evaluation
mirroring com.trafficsign.evaluation.MajorityClassBaseline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    majority_class_id: int
    training_frequency: int
    test_accuracy: float
    test_macro_f1: float
    total_test_samples: int


class MajorityClassBaseline:
    """
    Computes zero-intelligence baseline metrics by always predicting the most frequent training class.
    """

    @staticmethod
    def evaluate(train_labels: Sequence[int], test_labels: Sequence[int], num_classes: int) -> BaselineResult:
        if train_labels is None or test_labels is None:
            raise ValueError("Labels arrays must not be null")
        if len(train_labels) == 0 or len(test_labels) == 0:
            raise ValueError("Labels arrays must not be empty")

        counts = np.bincount(train_labels, minlength=num_classes)
        majority_class = int(np.argmax(counts))
        freq = int(counts[majority_class])

        test_arr = np.array(test_labels, dtype=np.int64)
        correct = int(np.sum(test_arr == majority_class))
        accuracy = float(correct / len(test_arr))

        # Precision for majority class = correct / len(test_arr)
        # Recall for majority class = 1.0
        # F1 for majority class = 2 * P * R / (P + R)
        p = accuracy
        r = 1.0
        f1_maj = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        macro_f1 = float(f1_maj / num_classes)

        return BaselineResult(
            majority_class_id=majority_class,
            training_frequency=freq,
            test_accuracy=accuracy,
            test_macro_f1=macro_f1,
            total_test_samples=len(test_arr)
        )
