package com.trafficsign.neural.rules;

import java.util.Objects;
import java.util.Random;

/**
 * Competitive Learning (Winner-Take-All / Vector Quantization).
 * An unsupervised neural rule where units compete to represent input patterns:
 * Winner k* = argmin_k ||x - w_k||
 * Update: w_{k*} = w_{k*} + eta * (x - w_{k*})
 */
public class CompetitiveLearning {

    private final int numPrototypes;
    private final int dimension;
    private final double[][] prototypes;
    private final double learningRate;

    public CompetitiveLearning(int numPrototypes, int dimension, double learningRate, long seed) {
        if (numPrototypes <= 0 || dimension <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.numPrototypes = numPrototypes;
        this.dimension = dimension;
        this.learningRate = learningRate;
        this.prototypes = new double[numPrototypes][dimension];

        Random rng = new Random(seed);
        for (int k = 0; k < numPrototypes; k++) {
            for (int d = 0; d < dimension; d++) {
                prototypes[k][d] = rng.nextDouble();
            }
        }
    }

    public int findWinner(double[] x) {
        Objects.requireNonNull(x, "input must not be null");
        if (x.length != dimension) {
            throw new IllegalArgumentException("Input dimension mismatch: " + x.length + " vs " + dimension);
        }

        int winner = 0;
        double minDistanceSq = Double.MAX_VALUE;

        for (int k = 0; k < numPrototypes; k++) {
            double distSq = 0.0;
            for (int d = 0; d < dimension; d++) {
                double diff = x[d] - prototypes[k][d];
                distSq += diff * diff;
            }
            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                winner = k;
            }
        }

        return winner;
    }

    public int trainSample(double[] x) {
        int winner = findWinner(x);

        // Update only the winning prototype towards x
        for (int d = 0; d < dimension; d++) {
            prototypes[winner][d] += learningRate * (x[d] - prototypes[winner][d]);
        }

        return winner;
    }

    public double[][] getPrototypes() {
        double[][] copy = new double[numPrototypes][dimension];
        for (int k = 0; k < numPrototypes; k++) {
            System.arraycopy(prototypes[k], 0, copy[k], 0, dimension);
        }
        return copy;
    }
}
