import numpy as np
from trafficsign.math.matrix import Matrix
from trafficsign.math.gradient_checker import GradientChecker
from trafficsign.neural.mlp import MLPNetwork


def test_gradient_checker():
    # Architecture: 4 inputs -> 5 hidden -> 3 outputs
    mlp = MLPNetwork([4, 5, 3], random_seed=42)

    x = Matrix.from_array([
        [0.5, -0.2, 0.1, 0.8],
        [-0.3, 0.4, 0.9, -0.1]
    ])
    y = [1, 2]

    result = GradientChecker.checkGradients(mlp, x, y, epsilon=1e-5, tolerance=1e-5) if hasattr(GradientChecker, 'checkGradients') else GradientChecker.check_gradients(mlp, x, y, epsilon=1e-5, tolerance=1e-5)

    assert result.passed
    assert result.total_parameters_checked == (4 * 5 + 5 + 5 * 3 + 3)
    assert result.vector_relative_error < 1e-5
