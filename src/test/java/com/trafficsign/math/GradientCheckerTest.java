package com.trafficsign.math;

import com.trafficsign.neural.activation.ActivationFunction;
import com.trafficsign.neural.activation.SigmoidActivation;
import com.trafficsign.neural.mlp.MLPNetwork;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class GradientCheckerTest {

    @Test
    @DisplayName("Analytical backpropagation gradients match finite-difference approximations (relative error < 1e-6)")
    void shouldPassFiniteDifferenceGradientCheck() {
        // Small 2-layer network with continuous smooth sigmoid activation for exact gradient checking
        int[] layers = {3, 4, 3};
        ActivationFunction[] acts = {new SigmoidActivation()};
        MLPNetwork mlp = new MLPNetwork(layers, acts, 12345L);

        // Deterministic synthetic batch of 4 samples
        Matrix X = new Matrix(new double[][]{
                {0.1, 0.7, -0.3},
                {-0.5, 0.2, 0.8},
                {0.9, -0.1, 0.4},
                {-0.2, -0.8, 0.1}
        });
        int[] y = new int[]{0, 2, 1, 0};

        double epsilon = 1e-5;
        double tolerance = 1e-5;

        GradientChecker.GradientCheckResult result = GradientChecker.checkGradients(mlp, X, y, epsilon, tolerance);

        assertThat(result.passed())
                .as("Numerical gradient check must pass within tolerance " + tolerance)
                .isTrue();

        assertThat(result.vectorRelativeError())
                .as("Vector Euclidean relative error must be strictly less than 1e-6")
                .isLessThan(1e-6);

        assertThat(result.totalParametersChecked())
                .as("All weight and bias parameters across all layers must be verified")
                .isEqualTo((3 * 4 + 4) + (4 * 3 + 3)); // 16 + 15 = 31 parameters
    }
}
