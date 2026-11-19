import numpy as np
from trafficsign.data.sampling import BalancedSampler


def test_balanced_sampler():
    # Heavily imbalanced labels: 90 of class 0, 10 of class 1
    labels = [0] * 90 + [1] * 10
    balanced_indices = BalancedSampler.generate_balanced_indices(labels, num_classes=2, seed=42)

    assert len(balanced_indices) == 100
    sampled_labels = [labels[i] for i in balanced_indices]
    counts = np.bincount(sampled_labels, minlength=2)

    # Balanced counts should be roughly 50/50
    assert abs(counts[0] - counts[1]) <= 2
