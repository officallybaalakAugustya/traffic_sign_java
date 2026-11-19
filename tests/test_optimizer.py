import numpy as np
from trafficsign.math.matrix import Matrix
from trafficsign.neural.optimizer import StochasticGradientDescent, MomentumOptimizer, NesterovOptimizer, BatchGradientDescent


def test_sgd_and_momentum_updates():
    w = Matrix.from_array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.1, 0.2])
    gw = Matrix.from_array([[0.5, 0.5], [0.5, 0.5]])
    gb = np.array([0.1, 0.1])

    # SGD
    sgd = StochasticGradientDescent(learning_rate=0.1)
    sgd.initialize(1, [[2, 2]], [2])
    sgd.update(0, w, b, gw, gb)
    assert np.isclose(w.get(0, 0), 0.95)  # 1.0 - 0.1 * 0.5
    assert np.isclose(b[0], 0.09)         # 0.1 - 0.1 * 0.1

    # Momentum
    w2 = Matrix.from_array([[1.0, 2.0], [3.0, 4.0]])
    b2 = np.array([0.1, 0.2])
    mom = MomentumOptimizer(learning_rate=0.1, momentum_factor=0.9)
    mom.initialize(1, [[2, 2]], [2])
    mom.update(0, w2, b2, gw, gb)
    assert np.isclose(w2.get(0, 0), 0.95)

    # Second step with accumulated momentum
    mom.update(0, w2, b2, gw, gb)
    # v = 0.9 * 0.05 + 0.1 * 0.5 = 0.045 + 0.05 = 0.095
    # w = 0.95 - 0.095 = 0.855
    assert np.isclose(w2.get(0, 0), 0.855)
