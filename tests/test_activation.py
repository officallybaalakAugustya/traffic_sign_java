import pytest
import numpy as np
from trafficsign.neural.activation import ReLUActivation, SigmoidActivation, Softmax


def test_relu_activation():
    relu = ReLUActivation()
    assert relu.activate(5.0) == 5.0
    assert relu.activate(-3.0) == 0.0
    assert relu.derivative(2.0, 2.0) == 1.0
    assert relu.derivative(-2.0, 0.0) == 0.0


def test_sigmoid_activation():
    sig = SigmoidActivation()
    assert sig.activate(0.0) == 0.5
    assert sig.derivative(0.0, 0.5) == 0.25


def test_softmax_numerical_stability():
    # Large logits that would overflow standard exp(1000)
    logits = [1000.0, 1001.0, 1000.0]
    probs = Softmax.activate(logits)

    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[1] > probs[0]
    assert np.isclose(probs[0], probs[2])

    batch_logits = [
        [1.0, 2.0, 3.0],
        [10.0, 10.0, 10.0]
    ]
    batch_probs = Softmax.activate_batch(batch_logits)
    assert batch_probs.shape == (2, 3)
    assert np.allclose(np.sum(batch_probs, axis=1), [1.0, 1.0])
    assert np.allclose(batch_probs[1], [1.0 / 3, 1.0 / 3, 1.0 / 3])
