"""
Artificial neuron implementations mirroring com.trafficsign.neural.neuron.
Includes McCullochPittsNeuron and SigmoidNeuron.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Union
import numpy as np


class McCullochPittsNeuron:
    """
    McCulloch-Pitts (1943) artificial neuron model.
    Computes a weighted sum over binary inputs and evaluates a hard step threshold.
    """

    def __init__(self, weights: Sequence[float], threshold: float):
        if weights is None or len(weights) == 0:
            raise ValueError("Neuron must have at least one input weight")
        self._weights = np.array(weights, dtype=np.float64)
        self._threshold = float(threshold)

    @classmethod
    def and_gate(cls) -> McCullochPittsNeuron:
        return cls([1.0, 1.0], 2.0)

    @classmethod
    def or_gate(cls) -> McCullochPittsNeuron:
        return cls([1.0, 1.0], 1.0)

    @classmethod
    def not_gate(cls) -> McCullochPittsNeuron:
        return cls([-1.0], 0.0)

    def activate(self, inputs: Sequence[int]) -> int:
        if inputs is None:
            raise ValueError("inputs must not be null")
        if len(inputs) != len(self._weights):
            raise ValueError(
                f"Input dimension {len(inputs)} does not match weight dimension {len(self._weights)}"
            )

        total = 0.0
        for i, val in enumerate(inputs):
            if val not in (0, 1):
                raise ValueError(
                    f"McCulloch-Pitts neuron accepts strictly binary inputs {{0, 1}}. Received: {val}"
                )
            total += val * self._weights[i]

        return 1 if total >= self._threshold else 0

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def threshold(self) -> float:
        return self._threshold


class SigmoidNeuron:
    """
    Single-unit artificial neuron with smooth, differentiable sigmoid activation.
    Implements numerically stable logistic evaluation to prevent floating-point overflow.
    """

    def __init__(self, weights: Sequence[float], bias: float):
        if weights is None or len(weights) == 0:
            raise ValueError("Neuron must have at least one weight")
        self._weights = np.array(weights, dtype=np.float64)
        self._bias = float(bias)

    @staticmethod
    def sigmoid(z: float) -> float:
        if math.isnan(z):
            raise ValueError("Input to sigmoid is NaN")
        if z < -45.0:
            return 0.0
        if z > 45.0:
            return 1.0
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1.0 + exp_z)

    @staticmethod
    def sigmoid_derivative(z: float) -> float:
        s = SigmoidNeuron.sigmoid(z)
        return s * (1.0 - s)

    def compute_affine(self, inputs: Sequence[float]) -> float:
        if inputs is None:
            raise ValueError("inputs must not be null")
        if len(inputs) != len(self._weights):
            raise ValueError(
                f"Input dimension {len(inputs)} does not match neuron weight dimension {len(self._weights)}"
            )
        arr = np.array(inputs, dtype=np.float64)
        return float(self._bias + np.dot(self._weights, arr))

    def activate(self, inputs: Sequence[float]) -> float:
        z = self.compute_affine(inputs)
        return self.sigmoid(z)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def bias(self) -> float:
        return self._bias
