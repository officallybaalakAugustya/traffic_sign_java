package com.trafficsign.neural.optimizer;

import com.trafficsign.math.Matrix;
import java.util.Objects;

/**
 * Standard Batch Gradient Descent optimizer:
 * W = W - eta * grad_W
 * b = b - eta * grad_b
 */
public class BatchGradientDescent implements Optimizer {

    private final double learningRate;

    public BatchGradientDescent(double learningRate) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate) || Double.isInfinite(learningRate)) {
            throw new IllegalArgumentException("Learning rate must be positive and finite. Received: " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public void initialize(int numLayers, int[][] weightShapes, int[] biasLengths) {
        // Stateless optimizer
    }

    @Override
    public void update(int layerIdx, Matrix weights, double[] biases, Matrix gradWeights, double[] gradBiases) {
        Objects.requireNonNull(weights, "weights must not be null");
        Objects.requireNonNull(biases, "biases must not be null");
        Objects.requireNonNull(gradWeights, "gradWeights must not be null");
        Objects.requireNonNull(gradBiases, "gradBiases must not be null");

        int rows = weights.rows();
        int cols = weights.cols();

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double updatedW = weights.get(r, c) - learningRate * gradWeights.get(r, c);
                weights.set(r, c, updatedW);
            }
        }

        for (int b = 0; b < biases.length; b++) {
            biases[b] -= learningRate * gradBiases[b];
        }
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }
}
