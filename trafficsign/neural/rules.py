"""
Classical neural learning rules mirroring com.trafficsign.neural.rules.
Includes HebbianLearning, ErrorCorrectionLearning, CompetitiveLearning, and HopfieldEnergyModel.
"""

from __future__ import annotations
import math
import random
from typing import List, Sequence, Union
import numpy as np


class HebbianLearning:
    """
    Hebbian Learning demonstration (Donald Hebb, 1949).
    Implements standard Hebbian correlation learning and Oja's normalized variant (1982):
    Delta w = eta * y * (x - y * w)
    """

    def __init__(self, input_dimension: int, learning_rate: float, use_oja_rule: bool):
        if input_dimension <= 0:
            raise ValueError(f"Input dimension must be positive: {input_dimension}")
        if learning_rate <= 0.0:
            raise ValueError(f"Learning rate must be positive: {learning_rate}")
        self.dimension = input_dimension
        self.learning_rate = float(learning_rate)
        self.use_oja_rule = use_oja_rule
        self._weights = np.full(input_dimension, 0.1, dtype=np.float64)

    def activate(self, x: Sequence[float]) -> float:
        if x is None:
            raise ValueError("input must not be null")
        arr = np.array(x, dtype=np.float64)
        if arr.size != self.dimension:
            raise ValueError(f"Dimension mismatch: {arr.size} vs {self.dimension}")
        return float(np.dot(self._weights, arr))

    def train_sample(self, x: Sequence[float]) -> None:
        arr = np.array(x, dtype=np.float64)
        y = self.activate(arr)

        if self.use_oja_rule:
            # Oja's rule: delta_w = eta * y * (x - y * w)
            self._weights += self.learning_rate * y * (arr - y * self._weights)
        else:
            # Standard Hebbian: delta_w = eta * y * x
            self._weights += self.learning_rate * y * arr

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()


class ErrorCorrectionLearning:
    """
    Error-Correction Learning (Perceptron Learning Rule / Widrow-Hoff Delta Rule).
    Updates weights in the direction of the prediction error:
    error = target - predicted
    Delta w = eta * error * x
    Delta b = eta * error
    """

    def __init__(self, input_dimension: int, learning_rate: float):
        if input_dimension <= 0:
            raise ValueError(f"Dimension must be positive: {input_dimension}")
        if learning_rate <= 0.0:
            raise ValueError(f"Learning rate must be positive: {learning_rate}")
        self.dimension = input_dimension
        self.learning_rate = float(learning_rate)
        self._weights = np.zeros(input_dimension, dtype=np.float64)
        self._bias = 0.0

    def predict(self, x: Sequence[float]) -> int:
        if x is None:
            raise ValueError("input must not be null")
        arr = np.array(x, dtype=np.float64)
        if arr.size != self.dimension:
            raise ValueError(f"Input dimension mismatch: {arr.size} vs {self.dimension}")
        total = self._bias + np.dot(self._weights, arr)
        return 1 if total >= 0.0 else 0

    def train_sample(self, x: Sequence[float], target: int) -> float:
        if target not in (0, 1):
            raise ValueError(f"Target must be binary {{0, 1}}. Received: {target}")
        arr = np.array(x, dtype=np.float64)
        pred = self.predict(arr)
        error = float(target - pred)

        if error != 0.0:
            self._weights += self.learning_rate * error * arr
            self._bias += self.learning_rate * error

        return abs(error)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def get_bias(self) -> float:
        return self._bias


class CompetitiveLearning:
    """
    Competitive Learning (Winner-Take-All / Vector Quantization).
    Winner k* = argmin_k ||x - w_k||
    Update: w_{k*} = w_{k*} + eta * (x - w_{k*})
    """

    def __init__(self, num_prototypes: int, dimension: int, learning_rate: float, seed: int = 42):
        if num_prototypes <= 0 or dimension <= 0:
            raise ValueError("Dimensions must be positive")
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive")
        self.num_prototypes = num_prototypes
        self.dimension = dimension
        self.learning_rate = float(learning_rate)

        rng = np.random.default_rng(seed)
        self._prototypes = rng.uniform(0.0, 1.0, (num_prototypes, dimension)).astype(np.float64)

    def find_winner(self, x: Sequence[float]) -> int:
        if x is None:
            raise ValueError("input must not be null")
        arr = np.array(x, dtype=np.float64)
        if arr.size != self.dimension:
            raise ValueError(f"Input dimension mismatch: {arr.size} vs {self.dimension}")

        diffs = self._prototypes - arr
        dists_sq = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(dists_sq))

    def train_sample(self, x: Sequence[float]) -> int:
        arr = np.array(x, dtype=np.float64)
        winner = self.find_winner(arr)
        self._prototypes[winner] += self.learning_rate * (arr - self._prototypes[winner])
        return winner

    def get_prototypes(self) -> np.ndarray:
        return self._prototypes.copy()


class HopfieldEnergyModel:
    """
    Hopfield Associative Memory and Energy-Based Model (John Hopfield, 1982).
    Stores binary/bipolar patterns in a recurrent symmetric weight matrix and
    minimizes Lyapunov energy:
    E(s) = - 0.5 * sum_{i != j} W_{ij} * s_i * s_j
    """

    def __init__(self, num_units: int):
        if num_units <= 0:
            raise ValueError("Number of units must be positive")
        self.num_units = num_units
        self._weights = np.zeros((num_units, num_units), dtype=np.float64)

    def store_pattern(self, pattern: Sequence[int]) -> None:
        if pattern is None:
            raise ValueError("pattern must not be null")
        arr = np.array(pattern, dtype=np.float64)
        if arr.size != self.num_units:
            raise ValueError(f"Pattern length {arr.size} does not match network size {self.num_units}")

        for val in arr:
            if val not in (-1.0, 1.0):
                raise ValueError(f"Hopfield patterns must be bipolar {{-1, +1}}. Received: {val}")

        scale = 1.0 / self.num_units
        outer = np.outer(arr, arr) * scale
        np.fill_diagonal(outer, 0.0)
        self._weights += outer

    def compute_energy(self, state: Sequence[int]) -> float:
        if state is None:
            raise ValueError("state must not be null")
        arr = np.array(state, dtype=np.float64)
        energy = -0.5 * float(arr @ self._weights @ arr)
        return energy

    def recall(self, noisy_input: Sequence[int], max_iterations: int = 100) -> np.ndarray:
        if noisy_input is None:
            raise ValueError("noisyInput must not be null")
        state = np.array(noisy_input, dtype=np.float64).copy()

        for _ in range(max_iterations):
            changed = False
            for i in range(self.num_units):
                h = float(np.dot(self._weights[i], state))
                new_val = 1.0 if h >= 0.0 else -1.0
                if state[i] != new_val:
                    state[i] = new_val
                    changed = True
            if not changed:
                break

        return state.astype(np.int64)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()
