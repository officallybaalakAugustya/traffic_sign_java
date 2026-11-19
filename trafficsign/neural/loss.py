"""
Categorical Cross-Entropy Loss for multi-class classification
mirroring com.trafficsign.neural.loss.CategoricalCrossEntropyLoss.
"""

from __future__ import annotations
import math
from typing import Sequence, Union
import numpy as np
from trafficsign.math.matrix import Matrix


class CategoricalCrossEntropyLoss:
    """
    Categorical Cross-Entropy Loss for multi-class classification:
    L = - (1 / N) * sum_i log(p_{i, y_i} + epsilon)

    When paired with Softmax output layer, the analytical derivative w.r.t logits is:
    dL / dz_{i, c} = (1 / N) * (p_{i, c} - y_{i, c})
    """

    EPSILON = 1e-15

    def compute_loss(self, probabilities: Union[np.ndarray, Sequence[Sequence[float]]],
                     true_labels: Sequence[int]) -> float:
        if probabilities is None or true_labels is None:
            raise ValueError("probabilities and true_labels must not be null")
        probs = np.array(probabilities, dtype=np.float64)
        labels = np.array(true_labels, dtype=np.int64)

        n = probs.shape[0]
        if n == 0:
            raise ValueError("Cannot compute loss on empty batch")
        if n != len(labels):
            raise ValueError(f"Batch size mismatch: {n} vs {len(labels)}")

        num_classes = probs.shape[1]
        total_loss = 0.0

        for i in range(n):
            y = int(labels[i])
            if y < 0 or y >= num_classes:
                raise ValueError(f"Label {y} out of bounds for numClasses={num_classes}")
            p = max(self.EPSILON, min(1.0 - self.EPSILON, probs[i, y]))
            total_loss -= math.log(p)

        return total_loss / n

    def compute_output_gradients(self, probabilities: Union[np.ndarray, Sequence[Sequence[float]]],
                                 true_labels: Sequence[int]) -> Matrix:
        if probabilities is None or true_labels is None:
            raise ValueError("probabilities and true_labels must not be null")
        probs = np.array(probabilities, dtype=np.float64)
        labels = np.array(true_labels, dtype=np.int64)

        n, num_classes = probs.shape
        grad_data = np.zeros((n, num_classes), dtype=np.float64)

        for i in range(n):
            target = int(labels[i])
            for c in range(num_classes):
                target_one_hot = 1.0 if c == target else 0.0
                grad_data[i, c] = (probs[i, c] - target_one_hot) / n

        return Matrix.from_array(grad_data)
