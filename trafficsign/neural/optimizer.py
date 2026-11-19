"""
Neural network optimizers mirroring com.trafficsign.neural.optimizer.
Includes Optimizer (ABC), StochasticGradientDescent, MomentumOptimizer,
NesterovOptimizer, and BatchGradientDescent.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from trafficsign.math.matrix import Matrix


class Optimizer(ABC):
    """
    Common optimizer interface for neural network parameter updates.
    """

    @abstractmethod
    def initialize(self, num_layers: int, weight_shapes: Sequence[Sequence[int]], bias_lengths: Sequence[int]) -> None:
        pass

    @abstractmethod
    def update(self, layer_idx: int, weights: Matrix, biases: np.ndarray,
               grad_weights: Matrix, grad_biases: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_learning_rate(self) -> float:
        pass


class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent (SGD) with optional learning rate decay schedule:
    eta_t = eta_0 / (1 + decay_rate * epoch)
    """

    def __init__(self, learning_rate: float, decay_rate: float = 0.0):
        if learning_rate <= 0.0 or math.isnan(learning_rate):
            raise ValueError(f"Learning rate must be strictly positive: {learning_rate}")
        if decay_rate < 0.0 or math.isnan(decay_rate):
            raise ValueError(f"Decay rate must be non-negative: {decay_rate}")
        self.initial_learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.current_learning_rate = float(learning_rate)

    def step_epoch(self, epoch: int) -> None:
        self.current_learning_rate = self.initial_learning_rate / (1.0 + self.decay_rate * epoch)

    def initialize(self, num_layers: int, weight_shapes: Sequence[Sequence[int]], bias_lengths: Sequence[int]) -> None:
        self.current_learning_rate = self.initial_learning_rate

    def update(self, layer_idx: int, weights: Matrix, biases: np.ndarray,
               grad_weights: Matrix, grad_biases: np.ndarray) -> None:
        if weights is None or biases is None or grad_weights is None or grad_biases is None:
            raise ValueError("Parameters and gradients must not be null")

        rows = weights.rows()
        cols = weights.cols()

        for r in range(rows):
            for c in range(cols):
                w = weights.get(r, c) - self.current_learning_rate * grad_weights.get(r, c)
                weights.set(r, c, w)

        for b in range(len(biases)):
            biases[b] -= self.current_learning_rate * grad_biases[b]

    def get_learning_rate(self) -> float:
        return self.current_learning_rate


# Alias SGD
SGD = StochasticGradientDescent


class MomentumOptimizer(Optimizer):
    """
    Polyak Momentum Optimizer:
    v_W = gamma * v_W + eta * grad_W
    W   = W - v_W
    v_b = gamma * v_b + eta * grad_b
    b   = b - v_b
    """

    def __init__(self, learning_rate: float, momentum_factor: float):
        if learning_rate <= 0.0 or math.isnan(learning_rate):
            raise ValueError(f"Learning rate must be strictly positive: {learning_rate}")
        if momentum_factor < 0.0 or momentum_factor >= 1.0 or math.isnan(momentum_factor):
            raise ValueError(f"Momentum factor must be in range [0.0, 1.0). Received: {momentum_factor}")
        self.learning_rate = float(learning_rate)
        self.momentum_factor = float(momentum_factor)
        self.velocity_weights: Optional[List[Matrix]] = None
        self.velocity_biases: Optional[List[np.ndarray]] = None

    def initialize(self, num_layers: int, weight_shapes: Sequence[Sequence[int]], bias_lengths: Sequence[int]) -> None:
        self.velocity_weights = [Matrix(weight_shapes[l][0], weight_shapes[l][1]) for l in range(num_layers)]
        self.velocity_biases = [np.zeros(bias_lengths[l], dtype=np.float64) for l in range(num_layers)]

    def update(self, layer_idx: int, weights: Matrix, biases: np.ndarray,
               grad_weights: Matrix, grad_biases: np.ndarray) -> None:
        if weights is None or biases is None or grad_weights is None or grad_biases is None:
            raise ValueError("Parameters and gradients must not be null")
        if self.velocity_weights is None or layer_idx >= len(self.velocity_weights):
            raise RuntimeError(f"MomentumOptimizer not initialized for layer {layer_idx}")

        v_w = self.velocity_weights[layer_idx]
        v_b = self.velocity_biases[layer_idx]
        rows = weights.rows()
        cols = weights.cols()

        for r in range(rows):
            for c in range(cols):
                new_vw = self.momentum_factor * v_w.get(r, c) + self.learning_rate * grad_weights.get(r, c)
                v_w.set(r, c, new_vw)
                weights.set(r, c, weights.get(r, c) - new_vw)

        for b in range(len(biases)):
            new_vb = self.momentum_factor * v_b[b] + self.learning_rate * grad_biases[b]
            v_b[b] = new_vb
            biases[b] -= new_vb

    def get_learning_rate(self) -> float:
        return self.learning_rate


class NesterovOptimizer(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) Optimizer:
    v_new = gamma * v_prev + eta * grad
    W     = W - (gamma * v_new + eta * grad)
    v     = v_new
    """

    def __init__(self, learning_rate: float, momentum_factor: float):
        if learning_rate <= 0.0 or math.isnan(learning_rate):
            raise ValueError(f"Learning rate must be strictly positive: {learning_rate}")
        if momentum_factor < 0.0 or momentum_factor >= 1.0 or math.isnan(momentum_factor):
            raise ValueError(f"Momentum factor must be in range [0.0, 1.0). Received: {momentum_factor}")
        self.learning_rate = float(learning_rate)
        self.momentum_factor = float(momentum_factor)
        self.velocity_weights: Optional[List[Matrix]] = None
        self.velocity_biases: Optional[List[np.ndarray]] = None

    def initialize(self, num_layers: int, weight_shapes: Sequence[Sequence[int]], bias_lengths: Sequence[int]) -> None:
        self.velocity_weights = [Matrix(weight_shapes[l][0], weight_shapes[l][1]) for l in range(num_layers)]
        self.velocity_biases = [np.zeros(bias_lengths[l], dtype=np.float64) for l in range(num_layers)]

    def update(self, layer_idx: int, weights: Matrix, biases: np.ndarray,
               grad_weights: Matrix, grad_biases: np.ndarray) -> None:
        if weights is None or biases is None or grad_weights is None or grad_biases is None:
            raise ValueError("Parameters and gradients must not be null")
        if self.velocity_weights is None or layer_idx >= len(self.velocity_weights):
            raise RuntimeError(f"NesterovOptimizer not initialized for layer {layer_idx}")

        v_w = self.velocity_weights[layer_idx]
        v_b = self.velocity_biases[layer_idx]
        rows = weights.rows()
        cols = weights.cols()

        for r in range(rows):
            for c in range(cols):
                g = grad_weights.get(r, c)
                v_prev = v_w.get(r, c)
                v_new = self.momentum_factor * v_prev + self.learning_rate * g
                v_w.set(r, c, v_new)
                nesterov_step = self.momentum_factor * v_new + self.learning_rate * g
                weights.set(r, c, weights.get(r, c) - nesterov_step)

        for b in range(len(biases)):
            g_b = grad_biases[b]
            v_prev_b = v_b[b]
            v_new_b = self.momentum_factor * v_prev_b + self.learning_rate * g_b
            v_b[b] = v_new_b
            nesterov_step_b = self.momentum_factor * v_new_b + self.learning_rate * g_b
            biases[b] -= nesterov_step_b

    def get_learning_rate(self) -> float:
        return self.learning_rate


class BatchGradientDescent(Optimizer):
    """
    Standard Batch Gradient Descent optimizer:
    W = W - eta * grad_W
    b = b - eta * grad_b
    """

    def __init__(self, learning_rate: float):
        if learning_rate <= 0.0 or math.isnan(learning_rate) or math.isinf(learning_rate):
            raise ValueError(f"Learning rate must be positive and finite. Received: {learning_rate}")
        self.learning_rate = float(learning_rate)

    def initialize(self, num_layers: int, weight_shapes: Sequence[Sequence[int]], bias_lengths: Sequence[int]) -> None:
        pass

    def update(self, layer_idx: int, weights: Matrix, biases: np.ndarray,
               grad_weights: Matrix, grad_biases: np.ndarray) -> None:
        if weights is None or biases is None or grad_weights is None or grad_biases is None:
            raise ValueError("Parameters and gradients must not be null")

        rows = weights.rows()
        cols = weights.cols()

        for r in range(rows):
            for c in range(cols):
                w = weights.get(r, c) - self.learning_rate * grad_weights.get(r, c)
                weights.set(r, c, w)

        for b in range(len(biases)):
            biases[b] -= self.learning_rate * grad_biases[b]

    def get_learning_rate(self) -> float:
        return self.learning_rate
