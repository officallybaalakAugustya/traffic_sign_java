package com.trafficsign.neural.mlp;

import com.trafficsign.math.Matrix;
import com.trafficsign.neural.activation.ActivationFunction;
import com.trafficsign.neural.activation.ReLUActivation;
import com.trafficsign.neural.activation.SigmoidActivation;
import com.trafficsign.neural.optimizer.BatchGradientDescent;
import com.trafficsign.neural.optimizer.MomentumOptimizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MLPNetworkTest {

    @Test
    @DisplayName("Should initialize MLP with configurable layers and deterministic random seed")
    void shouldInitializeDeterministically() {
        int[] layers = {4, 6, 3};
        MLPNetwork net1 = new MLPNetwork(layers, 42L);
        MLPNetwork net2 = new MLPNetwork(layers, 42L);

        // Verify identical weights with same seed
        assertThat(net1.getWeights(0).toArray()).isDeepEqualTo(net2.getWeights(0).toArray());
        assertThat(net1.getWeights(1).toArray()).isDeepEqualTo(net2.getWeights(1).toArray());

        // Verify different seed produces different weights
        MLPNetwork net3 = new MLPNetwork(layers, 999L);
        assertThat(net1.getWeights(0).get(0, 0)).isNotEqualTo(net3.getWeights(0).get(0, 0));
    }

    @Test
    @DisplayName("Forward pass output probabilities must sum to 1.0 for every sample")
    void shouldProduceValidProbabilities() {
        MLPNetwork net = new MLPNetwork(new int[]{3, 5, 4}, 42L);
        Matrix input = new Matrix(new double[][]{
                {0.5, -0.2, 0.8},
                {-1.0, 0.4, 0.0}
        });

        MLPNetwork.ForwardPassResult fwd = net.forward(input);
        double[][] probs = fwd.outputProbabilities();

        assertThat(probs.length).isEqualTo(2);
        assertThat(probs[0].length).isEqualTo(4);

        for (double[] p : probs) {
            double sum = 0.0;
            for (double prob : p) {
                assertThat(prob).isBetween(0.0, 1.0);
                sum += prob;
            }
            assertThat(sum).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-6));
        }
    }

    @Test
    @DisplayName("MLP training loss decreases significantly on a deterministic learnable dataset")
    void shouldDecreaseLossOnLearnableProblem() {
        // Linearly/non-linearly separable 3-class toy dataset in 2D
        Matrix X = new Matrix(new double[][]{
                {0.1, 0.2}, {0.2, 0.1}, {0.15, 0.25},  // Class 0
                {0.8, 0.9}, {0.9, 0.8}, {0.85, 0.95},  // Class 1
                {0.1, 0.9}, {0.2, 0.8}, {0.15, 0.85}   // Class 2
        });
        int[] y = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2};

        // Small 2-hidden-layer network: 2 -> 8 -> 8 -> 3
        int[] layers = {2, 8, 8, 3};
        ActivationFunction[] acts = {new ReLUActivation(), new ReLUActivation()};
        MLPNetwork net = new MLPNetwork(layers, acts, 42L);

        BatchGradientDescent optimizer = new BatchGradientDescent(0.2);

        double initialLoss = net.evaluateLoss(X, y);

        // Train for 150 epochs
        double finalLoss = initialLoss;
        for (int epoch = 0; epoch < 150; epoch++) {
            finalLoss = net.trainBatch(X, y, optimizer);
        }

        // Loss should decrease by at least 60%
        assertThat(finalLoss).isLessThan(initialLoss * 0.4);

        // Accuracy on training set should reach 100%
        int[] preds = net.predict(X);
        assertThat(preds).containsExactly(y);
    }

    @Test
    @DisplayName("Should throw on input dimension mismatch")
    void shouldThrowOnInputMismatch() {
        MLPNetwork net = new MLPNetwork(new int[]{4, 3, 2}, 42L);
        Matrix invalidInput = new Matrix(2, 5); // Expected cols=4, got 5

        assertThatThrownBy(() -> net.forward(invalidInput))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("does not match network input dimension");
    }
}
