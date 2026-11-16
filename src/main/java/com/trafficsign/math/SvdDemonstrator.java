package com.trafficsign.math;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.util.Objects;

/**
 * Singular Value Decomposition (SVD) engine and low-rank approximation demonstrator.
 * Uses Apache Commons Math3 for numerically robust SVD decomposition:
 * A = U * S * V^T
 */
public final class SvdDemonstrator {

    public record SvdResult(
            Matrix u,
            double[] singularValues,
            Matrix v,
            Matrix reconstructedRankK,
            int rankK,
            double frobeniusError,
            double energyPreservedRatio
    ) {
    }

    private SvdDemonstrator() {
    }

    public static SvdResult decomposeAndReconstruct(Matrix input, int rankK) {
        Objects.requireNonNull(input, "Input matrix must not be null");
        int m = input.rows();
        int n = input.cols();
        int maxRank = Math.min(m, n);

        if (rankK <= 0 || rankK > maxRank) {
            throw new IllegalArgumentException("Rank k (" + rankK + ") must be in range [1, " + maxRank + "]");
        }

        // Convert to Commons Math RealMatrix
        RealMatrix realMatrix = new Array2DRowRealMatrix(input.toArray());
        SingularValueDecomposition svd = new SingularValueDecomposition(realMatrix);

        double[] singularValues = svd.getSingularValues();
        RealMatrix u = svd.getU();
        RealMatrix v = svd.getV();

        // Compute rank-k approximation: A_k = U_k * S_k * V_k^T
        RealMatrix approx = new Array2DRowRealMatrix(m, n);
        for (int k = 0; k < rankK; k++) {
            double sigma = singularValues[k];
            for (int r = 0; r < m; r++) {
                double uVal = u.getEntry(r, k);
                for (int c = 0; c < n; c++) {
                    double vVal = v.getEntry(c, k);
                    approx.addToEntry(r, c, sigma * uVal * vVal);
                }
            }
        }

        // Compute Frobenius error ||A - A_k||_F
        double errorSquared = 0.0;
        double totalEnergy = 0.0;
        double approxEnergy = 0.0;

        for (int k = 0; k < singularValues.length; k++) {
            totalEnergy += singularValues[k] * singularValues[k];
            if (k < rankK) {
                approxEnergy += singularValues[k] * singularValues[k];
            }
        }

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                double diff = input.get(r, c) - approx.getEntry(r, c);
                errorSquared += diff * diff;
            }
        }

        double frobeniusError = Math.sqrt(errorSquared);
        double energyRatio = totalEnergy > 0 ? (approxEnergy / totalEnergy) : 1.0;

        return new SvdResult(
                new Matrix(u.getData()),
                singularValues.clone(),
                new Matrix(v.getData()),
                new Matrix(approx.getData()),
                rankK,
                frobeniusError,
                energyRatio
        );
    }
}
