package com.trafficsign.neural.activation;

import java.util.Objects;

/**
 * Numerically stable Softmax activation for categorical probability distributions:
 * P(y = c) = exp(z_c - max(z)) / sum_j exp(z_j - max(z))
 */
public final class Softmax {

    private Softmax() {
    }

    public static double[] activate(double[] logits) {
        Objects.requireNonNull(logits, "logits must not be null");
        if (logits.length == 0) {
            throw new IllegalArgumentException("Cannot compute softmax on empty array");
        }

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double z : logits) {
            if (Double.isNaN(z)) {
                throw new IllegalArgumentException("Logit is NaN");
            }
            if (z > maxLogit) {
                maxLogit = z;
            }
        }

        double[] exp = new double[logits.length];
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - maxLogit);
            sumExp += exp[i];
        }

        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exp[i] / sumExp;
        }

        return probabilities;
    }

    public static double[][] activateBatch(double[][] batchLogits) {
        Objects.requireNonNull(batchLogits, "batchLogits must not be null");
        double[][] batchProbs = new double[batchLogits.length][];
        for (int i = 0; i < batchLogits.length; i++) {
            batchProbs[i] = activate(batchLogits[i]);
        }
        return batchProbs;
    }
}
