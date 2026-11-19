"""
Neural activation functions mirroring com.trafficsign.neural.activation.
Includes ActivationFunction, ReLUActivation, SigmoidActivation, and Softmax.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import List, Sequence, Union
import numpy as np
from trafficsign.neural.neuron import SigmoidNeuron


class ActivationFunction(ABC):
    """
    Abstract interface for activation functions.
    """

    @abstractmethod
    def activate(self, z: float) -> float:
        pass

    @abstractmethod
    def derivative(self, z: float, a: float) -> float:
        pass


class ReLUActivation(ActivationFunction):
    """
    Rectified Linear Unit: f(z) = max(0, z)
    """

    def activate(self, z: float) -> float:
        return max(0.0, z)

    def derivative(self, z: float, a: float) -> float:
        return 1.0 if z > 0.0 else 0.0


class SigmoidActivation(ActivationFunction):
    """
    Sigmoid activation: f(z) = 1 / (1 + exp(-z))
    """

    def activate(self, z: float) -> float:
        return SigmoidNeuron.sigmoid(z)

    def derivative(self, z: float, a: float) -> float:
        return a * (1.0 - a)


class Softmax:
    """
    Numerically stable Softmax activation for categorical probability distributions:
    P(y = c) = exp(z_c - max(z)) / sum_j exp(z_j - max(z))
    """

    @staticmethod
    def activate(logits: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        if logits is None or len(logits) == 0:
            raise ValueError("Cannot compute softmax on empty array")
        arr = np.array(logits, dtype=np.float64)
        if np.isnan(arr).any():
            raise ValueError("Logit is NaN")

        max_val = np.max(arr)
        exp_arr = np.exp(arr - max_val)
        sum_exp = np.sum(exp_arr)
        return exp_arr / sum_exp

    @staticmethod
    def activate_batch(batch_logits: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        if batch_logits is None:
            raise ValueError("batchLogits must not be null")
        arr = np.array(batch_logits, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("batchLogits must be 2D")
        max_vals = np.max(arr, axis=1, keepdims=True)
        exp_arr = np.exp(arr - max_vals)
        sum_exp = np.sum(exp_arr, axis=1, keepdims=True)
        return exp_arr / sum_exp
