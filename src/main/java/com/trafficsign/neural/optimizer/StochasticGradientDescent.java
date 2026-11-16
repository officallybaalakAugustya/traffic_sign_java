package com.trafficsign.neural.optimizer;

import com.trafficsign.math.Matrix;
import java.util.Objects;

/**
 * Stochastic Gradient Descent (SGD) with optional learning rate decay schedule:
 * eta_t = eta_0 / (1 + decayRate * epoch)
 */
public class StochasticGradientDescent implements Optimizer {

    private final double initialLearningRate;
    private final double decayRate;
    private double currentLearningRate;

    public StochasticGradientDescent(double learningRate) {
        this(learningRate, 0.0);
    }

    public StochasticGradientDescent(double learningRate, double decayRate) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate)) {
            throw new IllegalArgumentException("Learning rate must be strictly positive: " + learningRate);
        }
        if (decayRate < 0.0 || Double.isNaN(decayRate)) {
            throw new IllegalArgumentException("Decay rate must be non-negative: " + decayRate);
        }
        this.initialLearningRate = learningRate;
        this.decayRate = decayRate;
        this.currentLearningRate = learningRate;
    }

    public void stepEpoch(int epoch) {
        this.currentLearningRate = initialLearningRate / (1.0 + decayRate * epoch);
    }

    @Override
    public void initialize(int numLayers, int[][] weightShapes, int[] biasLengths) {
        this.currentLearningRate = initialLearningRate;
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
                double updatedW = weights.get(r, c) - currentLearningRate * gradWeights.get(r, c);
                weights.set(r, c, updatedW);
            }
        }

        for (int b = 0; b < biases.length; b++) {
            biases[b] -= currentLearningRate * gradBiases[b];
        }
    }

    @Override
    public double getLearningRate() {
        return currentLearningRate;
    }
}
