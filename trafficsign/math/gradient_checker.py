"""
Numerical finite-difference gradient checking engine
mirroring com.trafficsign.math.GradientChecker.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence
import numpy as np

from trafficsign.math.matrix import Matrix

if TYPE_CHECKING:
    from trafficsign.neural.mlp import MLPNetwork


@dataclass(frozen=True)
class GradientCheckResult:
    max_relative_error: float
    vector_relative_error: float
    total_parameters_checked: int
    passed: bool


class GradientChecker:
    """
    Numerical finite-difference gradient checking engine.
    Computes:
    grad_num(theta) = (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)
    and compares against analytical gradients:
    error = ||grad_anal - grad_num|| / (||grad_anal|| + ||grad_num||)
    """

    DEFAULT_EPSILON = 1e-5

    @staticmethod
    def check_gradients(
        mlp: MLPNetwork,
        x: Matrix,
        y: Sequence[int],
        epsilon: float = DEFAULT_EPSILON,
        tolerance: float = 1e-5
    ) -> GradientCheckResult:
        if mlp is None or x is None or y is None:
            raise ValueError("Arguments must not be null")

        # 1. Analytical backward pass
        fwd = mlp.forward(x)
        anal_grads = mlp.backward(fwd, y)

        max_rel_error = 0.0
        sum_sq_diff = 0.0
        sum_sq_anal = 0.0
        sum_sq_num = 0.0
        param_count = 0

        num_layers = mlp.get_num_layers()

        # 2. Numerical check for weights
        for l in range(num_layers):
            w = mlp.get_weights(l)
            grad_w = anal_grads.grad_weights[l]

            for r in range(w.rows()):
                for c in range(w.cols()):
                    orig = w.get(r, c)
                    g_anal = grad_w.get(r, c)

                    # theta + epsilon
                    mlp.set_weight(l, r, c, orig + epsilon)
                    loss_plus = mlp.evaluate_loss(x, y)

                    # theta - epsilon
                    mlp.set_weight(l, r, c, orig - epsilon)
                    loss_minus = mlp.evaluate_loss(x, y)

                    # restore original
                    mlp.set_weight(l, r, c, orig)

                    g_num = (loss_plus - loss_minus) / (2.0 * epsilon)

                    diff = g_anal - g_num
                    rel_err = abs(diff) / (max(abs(g_anal), abs(g_num)) + 1e-12)
                    if rel_err > max_rel_error:
                        max_rel_error = rel_err

                    sum_sq_diff += diff * diff
                    sum_sq_anal += g_anal * g_anal
                    sum_sq_num += g_num * g_num
                    param_count += 1

            # 3. Numerical check for biases
            b = mlp.get_biases(l)
            grad_b = anal_grads.grad_biases[l]

            for i in range(len(b)):
                orig = b[i]
                g_anal = grad_b[i]

                mlp.set_bias(l, i, orig + epsilon)
                loss_plus = mlp.evaluate_loss(x, y)

                mlp.set_bias(l, i, orig - epsilon)
                loss_minus = mlp.evaluate_loss(x, y)

                mlp.set_bias(l, i, orig)

                g_num = (loss_plus - loss_minus) / (2.0 * epsilon)

                diff = g_anal - g_num
                rel_err = abs(diff) / (max(abs(g_anal), abs(g_num)) + 1e-12)
                if rel_err > max_rel_error:
                    max_rel_error = rel_err

                sum_sq_diff += diff * diff
                sum_sq_anal += g_anal * g_anal
                sum_sq_num += g_num * g_num
                param_count += 1

        norm_diff = math.sqrt(sum_sq_diff)
        norm_anal = math.sqrt(sum_sq_anal)
        norm_num = math.sqrt(sum_sq_num)
        vector_rel_error = norm_diff / (norm_anal + norm_num + 1e-12)
        passed = vector_rel_error < tolerance

        return GradientCheckResult(
            max_relative_error=max_rel_error,
            vector_relative_error=vector_rel_error,
            total_parameters_checked=param_count,
            passed=passed
        )
