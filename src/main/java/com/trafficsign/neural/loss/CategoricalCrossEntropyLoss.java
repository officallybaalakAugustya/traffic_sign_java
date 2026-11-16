package com.trafficsign.neural.loss;

import com.trafficsign.math.Matrix;
import java.util.Objects;

/**
 * Categorical Cross-Entropy Loss for multi-class classification:
 * L = - (1 / N) * sum_i log(p_{i, y_i} + epsilon)
 *
 * When paired with Softmax output layer, the analytical derivative w.r.t logits is:
 * dL / dz_{i, c} = (1 / N) * (p_{i, c} - y_{i, c})
 */
public class CategoricalCrossEntropyLoss {

    public static final double EPSILON = 1e-15;

    public double computeLoss(double[][] probabilities, int[] trueLabels) {
        Objects.requireNonNull(probabilities, "probabilities must not be null");
        Objects.requireNonNull(trueLabels, "trueLabels must not be null");
        int n = probabilities.length;
        if (n == 0) {
            throw new IllegalArgumentException("Cannot compute loss on empty batch");
        }
        if (n != trueLabels.length) {
            throw new IllegalArgumentException("Batch size mismatch: " + n + " vs " + trueLabels.length);
        }

        double totalLoss = 0.0;
        int numClasses = probabilities[0].length;

        for (int i = 0; i < n; i++) {
            int y = trueLabels[i];
            if (y < 0 || y >= numClasses) {
                throw new IllegalArgumentException("Label " + y + " out of bounds for numClasses=" + numClasses);
            }
            double prob = Math.max(EPSILON, Math.min(1.0 - EPSILON, probabilities[i][y]));
            totalLoss -= Math.log(prob);
        }

        return totalLoss / n;
    }

    public Matrix computeOutputGradients(double[][] probabilities, int[] trueLabels) {
        Objects.requireNonNull(probabilities, "probabilities must not be null");
        Objects.requireNonNull(trueLabels, "trueLabels must not be null");
        int n = probabilities.length;
        int numClasses = probabilities[0].length;

        double[][] gradData = new double[n][numClasses];
        for (int i = 0; i < n; i++) {
            int target = trueLabels[i];
            for (int c = 0; c < numClasses; c++) {
                double targetOneHot = (c == target) ? 1.0 : 0.0;
                gradData[i][c] = (probabilities[i][c] - targetOneHot) / n;
            }
        }

        return new Matrix(gradData);
    }
}
