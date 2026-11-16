package com.trafficsign.neural.rules;

import java.util.Objects;

/**
 * Hebbian Learning demonstration (Donald Hebb, 1949).
 * Implements standard Hebbian correlation learning ("neurons that fire together wire together")
 * and Oja's normalized variant (1982) preventing weight divergence:
 * Delta w = eta * y * (x - y * w)
 */
public class HebbianLearning {

    private final double[] weights;
    private final double learningRate;
    private final boolean useOjaRule;

    public HebbianLearning(int inputDimension, double learningRate, boolean useOjaRule) {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive: " + inputDimension);
        }
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        }
        this.weights = new double[inputDimension];
        for (int i = 0; i < inputDimension; i++) {
            this.weights[i] = 0.1;
        }
        this.learningRate = learningRate;
        this.useOjaRule = useOjaRule;
    }

    public double activate(double[] input) {
        Objects.requireNonNull(input, "input must not be null");
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Dimension mismatch: " + input.length + " vs " + weights.length);
        }
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * input[i];
        }
        return sum;
    }

    public void trainSample(double[] input) {
        double y = activate(input);

        for (int i = 0; i < weights.length; i++) {
            if (useOjaRule) {
                // Oja's rule: delta_w = eta * y * (x - y * w)
                weights[i] += learningRate * y * (input[i] - y * weights[i]);
            } else {
                // Standard Hebbian: delta_w = eta * y * x
                weights[i] += learningRate * y * input[i];
            }
        }
    }

    public double[] getWeights() {
        return weights.clone();
    }
}
