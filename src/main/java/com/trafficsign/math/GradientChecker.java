package com.trafficsign.math;

import com.trafficsign.neural.mlp.MLPNetwork;

import java.util.Objects;

/**
 * Numerical finite-difference gradient checking engine.
 * Computes:
 * grad_num(theta) = (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)
 * and compares against analytical gradients using relative error:
 * error = ||grad_anal - grad_num|| / (||grad_anal|| + ||grad_num||)
 */
public final class GradientChecker {

    public static final double DEFAULT_EPSILON = 1e-5;

    public record GradientCheckResult(
            double maxRelativeError,
            double vectorRelativeError,
            int totalParametersChecked,
            boolean passed
    ) {
    }

    private GradientChecker() {
    }

    public static GradientCheckResult checkGradients(MLPNetwork mlp, Matrix X, int[] y, double epsilon, double tolerance) {
        Objects.requireNonNull(mlp, "mlp must not be null");
        Objects.requireNonNull(X, "X must not be null");
        Objects.requireNonNull(y, "y must not be null");

        // 1. Analytical backward pass
        MLPNetwork.ForwardPassResult fwd = mlp.forward(X);
        MLPNetwork.Gradients analGrads = mlp.backward(fwd, y);

        double maxRelError = 0.0;
        double sumSqDiff = 0.0;
        double sumSqAnal = 0.0;
        double sumSqNum = 0.0;
        int paramCount = 0;

        int numLayers = mlp.getNumLayers();

        // 2. Numerical check for weights
        for (int l = 0; l < numLayers; l++) {
            Matrix w = mlp.getWeights(l);
            Matrix gradW = analGrads.gradWeights()[l];

            for (int r = 0; r < w.rows(); r++) {
                for (int c = 0; c < w.cols(); c++) {
                    double orig = w.get(r, c);
                    double gAnal = gradW.get(r, c);

                    // theta + epsilon
                    mlp.setWeight(l, r, c, orig + epsilon);
                    double lossPlus = mlp.evaluateLoss(X, y);

                    // theta - epsilon
                    mlp.setWeight(l, r, c, orig - epsilon);
                    double lossMinus = mlp.evaluateLoss(X, y);

                    // restore original
                    mlp.setWeight(l, r, c, orig);

                    double gNum = (lossPlus - lossMinus) / (2.0 * epsilon);

                    double diff = gAnal - gNum;
                    double relErr = Math.abs(diff) / (Math.max(Math.abs(gAnal), Math.abs(gNum)) + 1e-12);
                    if (relErr > maxRelError) {
                        maxRelError = relErr;
                    }

                    sumSqDiff += diff * diff;
                    sumSqAnal += gAnal * gAnal;
                    sumSqNum += gNum * gNum;
                    paramCount++;
                }
            }

            // 3. Numerical check for biases
            double[] b = mlp.getBiases(l);
            double[] gradB = analGrads.gradBiases()[l];

            for (int i = 0; i < b.length; i++) {
                double orig = b[i];
                double gAnal = gradB[i];

                mlp.setBias(l, i, orig + epsilon);
                double lossPlus = mlp.evaluateLoss(X, y);

                mlp.setBias(l, i, orig - epsilon);
                double lossMinus = mlp.evaluateLoss(X, y);

                mlp.setBias(l, i, orig);

                double gNum = (lossPlus - lossMinus) / (2.0 * epsilon);

                double diff = gAnal - gNum;
                double relErr = Math.abs(diff) / (Math.max(Math.abs(gAnal), Math.abs(gNum)) + 1e-12);
                if (relErr > maxRelError) {
                    maxRelError = relErr;
                }

                sumSqDiff += diff * diff;
                sumSqAnal += gAnal * gAnal;
                sumSqNum += gNum * gNum;
                paramCount++;
            }
        }

        double normDiff = Math.sqrt(sumSqDiff);
        double normSum = Math.sqrt(sumSqAnal) + Math.sqrt(sumSqNum);
        double vectorRelError = normSum > 0 ? (normDiff / normSum) : 0.0;

        boolean passed = vectorRelError <= tolerance;
        return new GradientCheckResult(maxRelError, vectorRelError, paramCount, passed);
    }
}
