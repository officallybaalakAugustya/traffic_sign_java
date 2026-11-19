import numpy as np
from trafficsign.neural.loss import CategoricalCrossEntropyLoss


def test_categorical_cross_entropy():
    loss_fn = CategoricalCrossEntropyLoss()

    # Perfect prediction
    probs = [[0.0, 1.0, 0.0]]
    labels = [1]
    loss = loss_fn.compute_loss(probs, labels)
    assert loss < 1e-10

    # Gradients w.r.t logits: (p - y) / N
    grads = loss_fn.compute_output_gradients(probs, labels)
    assert grads.rows() == 1
    assert grads.cols() == 3
    assert np.isclose(grads.get(0, 1), 0.0)  # 1.0 - 1.0 = 0
