package com.trafficsign.neural.neuron;

import java.util.Objects;

/**
 * Single-unit artificial neuron with smooth, differentiable sigmoid activation.
 * Implements numerically stable logistic evaluation to prevent floating-point overflow.
 */
public class SigmoidNeuron {

    private final double[] weights;
    private final double bias;

    public SigmoidNeuron(double[] weights, double bias) {
        Objects.requireNonNull(weights, "weights must not be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("Neuron must have at least one weight");
        }
        this.weights = weights.clone();
        this.bias = bias;
    }

    /**
     * Numerically stable sigmoid function:
     * sigma(z) = 1 / (1 + exp(-z)) for z >= 0
     * sigma(z) = exp(z) / (1 + exp(z)) for z < 0
     */
    public static double sigmoid(double z) {
        if (Double.isNaN(z)) {
            throw new IllegalArgumentException("Input to sigmoid is NaN");
        }
        if (z < -45.0) {
            return 0.0;
        }
        if (z > 45.0) {
            return 1.0;
        }
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        } else {
            double expZ = Math.exp(z);
            return expZ / (1.0 + expZ);
        }
    }

    /**
     * Derivative of sigmoid activation: d/dz sigma(z) = sigma(z) * (1 - sigma(z))
     */
    public static double sigmoidDerivative(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }

    public double computeAffine(double[] inputs) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input dimension " + inputs.length
                    + " does not match neuron weight dimension " + weights.length);
        }
        double z = bias;
        for (int i = 0; i < inputs.length; i++) {
            z += inputs[i] * weights[i];
        }
        return z;
    }

    public double activate(double[] inputs) {
        double z = computeAffine(inputs);
        return sigmoid(z);
    }

    public double[] getWeights() {
        return weights.clone();
    }

    public double getBias() {
        return bias;
    }
}
