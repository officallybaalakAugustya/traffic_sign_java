package com.trafficsign.neural.rules;

import java.util.Objects;

/**
 * Error-Correction Learning (Perceptron Learning Rule / Widrow-Hoff Delta Rule).
 * Updates weights in the direction of the prediction error:
 * error = target - predicted
 * Delta w = eta * error * x
 * Delta b = eta * error
 */
public class ErrorCorrectionLearning {

    private final double[] weights;
    private double bias;
    private final double learningRate;

    public ErrorCorrectionLearning(int inputDimension, double learningRate) {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + inputDimension);
        }
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        }
        this.weights = new double[inputDimension];
        this.bias = 0.0;
        this.learningRate = learningRate;
    }

    public int predict(double[] x) {
        Objects.requireNonNull(x, "input must not be null");
        if (x.length != weights.length) {
            throw new IllegalArgumentException("Input dimension mismatch: " + x.length + " vs " + weights.length);
        }

        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * x[i];
        }

        return sum >= 0.0 ? 1 : 0;
    }

    public double trainSample(double[] x, int target) {
        if (target != 0 && target != 1) {
            throw new IllegalArgumentException("Target must be binary {0, 1}. Received: " + target);
        }

        int pred = predict(x);
        double error = target - pred;

        if (error != 0.0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] += learningRate * error * x[i];
            }
            bias += learningRate * error;
        }

        return Math.abs(error);
    }

    public double[] getWeights() {
        return weights.clone();
    }

    public double getBias() {
        return bias;
    }
}
