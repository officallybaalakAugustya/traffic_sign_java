"""
Balanced and smoothed class rebalancing samplers
mirroring com.trafficsign.data.sampling.BalancedSampler.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Sequence
import numpy as np


class BalancedSampler:
    """
    Balanced sampling utilities for resolving heavy long-tailed class imbalance.
    """

    @staticmethod
    def generate_balanced_indices(labels: Sequence[int], num_classes: int, seed: int = 42) -> np.ndarray:
        if labels is None or len(labels) == 0:
            return np.array([], dtype=np.int64)

        class_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            class_to_indices.setdefault(y, []).append(i)

        rng = random.Random(seed)
        total_target_samples = len(labels)
        active_classes = len(class_to_indices)
        samples_per_class = max(1, total_target_samples // active_classes)

        balanced_list: List[int] = []

        for idx_list in class_to_indices.values():
            n = len(idx_list)
            for _ in range(samples_per_class):
                balanced_list.append(idx_list[rng.randint(0, n - 1)])

        class_keys = list(class_to_indices.keys())
        while len(balanced_list) < total_target_samples:
            rand_cls = class_keys[rng.randint(0, active_classes - 1)]
            idx_list = class_to_indices[rand_cls]
            balanced_list.append(idx_list[rng.randint(0, len(idx_list) - 1)])

        rng.shuffle(balanced_list)
        return np.array(balanced_list, dtype=np.int64)

    @staticmethod
    def generate_smoothed_indices(
        labels: Sequence[int],
        num_classes: int,
        power: float = 0.5,
        seed: int = 42
    ) -> np.ndarray:
        if labels is None or len(labels) == 0:
            return np.array([], dtype=np.int64)

        class_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            class_to_indices.setdefault(y, []).append(i)

        rng = random.Random(seed)
        total_target_samples = len(labels)

        # Compute smoothed target counts: N_c^power
        weights = {c: len(idxs) ** power for c, idxs in class_to_indices.items()}
        sum_weights = sum(weights.values())

        balanced_list: List[int] = []
        for c, idxs in class_to_indices.items():
            n_target = max(1, int(round((weights[c] / sum_weights) * total_target_samples)))
            n = len(idxs)
            for _ in range(n_target):
                balanced_list.append(idxs[rng.randint(0, n - 1)])

        class_keys = list(class_to_indices.keys())
        while len(balanced_list) < total_target_samples:
            rand_cls = class_keys[rng.randint(0, len(class_keys) - 1)]
            idxs = class_to_indices[rand_cls]
            balanced_list.append(idxs[rng.randint(0, len(idxs) - 1)])

        rng.shuffle(balanced_list)
        return np.array(balanced_list[:total_target_samples], dtype=np.int64)
