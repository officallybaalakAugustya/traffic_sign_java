package com.trafficsign.neural.optimizer;

import com.trafficsign.math.Matrix;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class OptimizerTest {

    @Test
    @DisplayName("BatchGradientDescent updates weights and biases along negative gradient")
    void shouldUpdateWithBGD() {
        BatchGradientDescent bgd = new BatchGradientDescent(0.1);
        Matrix w = new Matrix(new double[][]{{1.0, 2.0}});
        double[] b = new double[]{0.5};

        Matrix gradW = new Matrix(new double[][]{{0.2, -0.4}});
        double[] gradB = new double[]{0.1};

        bgd.update(0, w, b, gradW, gradB);

        // w = w - 0.1 * gradW -> [1.0 - 0.02, 2.0 - (-0.04)] = [0.98, 2.04]
        assertThat(w.get(0, 0)).isCloseTo(0.98, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(w.get(0, 1)).isCloseTo(2.04, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(b[0]).isCloseTo(0.49, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    @DisplayName("MomentumOptimizer maintains and updates velocity correctly across steps")
    void shouldAccumulateVelocityInMomentum() {
        // lr = 0.1, gamma = 0.9
        MomentumOptimizer momentum = new MomentumOptimizer(0.1, 0.9);
        momentum.initialize(1, new int[][]{{1, 1}}, new int[]{1});

        Matrix w = new Matrix(new double[][]{{5.0}});
        double[] b = new double[]{0.0};

        Matrix gradW = new Matrix(new double[][]{{2.0}});
        double[] gradB = new double[]{0.0};

        // Step 1: v_0 = 0.9 * 0 + 0.1 * 2 = 0.2. W = 5.0 - 0.2 = 4.8
        momentum.update(0, w, b, gradW, gradB);
        assertThat(w.get(0, 0)).isCloseTo(4.8, org.assertj.core.data.Offset.offset(1e-6));

        // Step 2: v_1 = 0.9 * 0.2 + 0.1 * 2 = 0.18 + 0.2 = 0.38. W = 4.8 - 0.38 = 4.42
        momentum.update(0, w, b, gradW, gradB);
        assertThat(w.get(0, 0)).isCloseTo(4.42, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    @DisplayName("NesterovOptimizer applies look-ahead step")
    void shouldApplyNesterovLookAhead() {
        NesterovOptimizer nag = new NesterovOptimizer(0.1, 0.9);
        nag.initialize(1, new int[][]{{1, 1}}, new int[]{1});

        Matrix w = new Matrix(new double[][]{{5.0}});
        double[] b = new double[]{0.0};

        Matrix gradW = new Matrix(new double[][]{{2.0}});
        double[] gradB = new double[]{0.0};

        // Step 1: v_new = 0.9 * 0 + 0.1 * 2 = 0.2.
        // Step size = 0.9 * 0.2 + 0.1 * 2 = 0.18 + 0.20 = 0.38
        // W = 5.0 - 0.38 = 4.62
        nag.update(0, w, b, gradW, gradB);
        assertThat(w.get(0, 0)).isCloseTo(4.62, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    @DisplayName("Optimizer instances maintain independent states and do not leak between models")
    void shouldMaintainIndependentState() {
        MomentumOptimizer opt1 = new MomentumOptimizer(0.1, 0.9);
        opt1.initialize(1, new int[][]{{1, 1}}, new int[]{1});

        MomentumOptimizer opt2 = new MomentumOptimizer(0.1, 0.9);
        opt2.initialize(1, new int[][]{{1, 1}}, new int[]{1});

        Matrix w1 = new Matrix(new double[][]{{1.0}});
        double[] b1 = new double[]{0.0};
        opt1.update(0, w1, b1, new Matrix(new double[][]{{10.0}}), new double[]{0.0});

        assertThat(opt1.getVelocityWeights(0).get(0, 0)).isGreaterThan(0.0);
        assertThat(opt2.getVelocityWeights(0).get(0, 0)).isEqualTo(0.0);
    }

    @Test
    @DisplayName("Optimizer validates hyperparameter boundaries")
    void shouldValidateHyperparameters() {
        assertThatThrownBy(() -> new BatchGradientDescent(-0.1))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new MomentumOptimizer(0.1, 1.2)) // gamma >= 1.0
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new NesterovOptimizer(0.0, 0.9)) // lr <= 0
                .isInstanceOf(IllegalArgumentException.class);
    }
}
