package com.trafficsign.neural.neuron;

import java.util.Objects;

/**
 * McCulloch-Pitts (1943) artificial neuron model.
 * Computes a weighted sum over binary inputs and evaluates a hard step threshold.
 */
public class McCullochPittsNeuron {

    private final double[] weights;
    private final double threshold;

    public McCullochPittsNeuron(double[] weights, double threshold) {
        Objects.requireNonNull(weights, "weights must not be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("Neuron must have at least one input weight");
        }
        this.weights = weights.clone();
        this.threshold = threshold;
    }

    public static McCullochPittsNeuron andGate() {
        return new McCullochPittsNeuron(new double[]{1.0, 1.0}, 2.0);
    }

    public static McCullochPittsNeuron orGate() {
        return new McCullochPittsNeuron(new double[]{1.0, 1.0}, 1.0);
    }

    public static McCullochPittsNeuron notGate() {
        return new McCullochPittsNeuron(new double[]{-1.0}, 0.0);
    }

    public int activate(int[] inputs) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input dimension " + inputs.length
                    + " does not match weight dimension " + weights.length);
        }

        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] != 0 && inputs[i] != 1) {
                throw new IllegalArgumentException("McCulloch-Pitts neuron accepts strictly binary inputs {0, 1}. Received: " + inputs[i]);
            }
            sum += inputs[i] * weights[i];
        }

        return sum >= threshold ? 1 : 0;
    }

    public double[] getWeights() {
        return weights.clone();
    }

    public double getThreshold() {
        return threshold;
    }
}
