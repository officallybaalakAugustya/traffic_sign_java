package com.trafficsign.neural.optimizer;

import com.trafficsign.math.Matrix;
import java.util.Objects;

/**
 * Nesterov Accelerated Gradient (NAG) Optimizer:
 * Computes look-ahead momentum adjustments to reduce overshoot and oscillation:
 *
 * v_new = gamma * v_prev + eta * grad
 * W     = W - (gamma * v_new + eta * grad)
 * v     = v_new
 */
public class NesterovOptimizer implements Optimizer {

    private final double learningRate;
    private final double momentumFactor; // gamma

    private Matrix[] velocityWeights;
    private double[][] velocityBiases;

    public NesterovOptimizer(double learningRate, double momentumFactor) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate)) {
            throw new IllegalArgumentException("Learning rate must be strictly positive: " + learningRate);
        }
        if (momentumFactor < 0.0 || momentumFactor >= 1.0 || Double.isNaN(momentumFactor)) {
            throw new IllegalArgumentException("Momentum factor must be in range [0.0, 1.0). Received: " + momentumFactor);
        }
        this.learningRate = learningRate;
        this.momentumFactor = momentumFactor;
    }

    @Override
    public void initialize(int numLayers, int[][] weightShapes, int[] biasLengths) {
        this.velocityWeights = new Matrix[numLayers];
        this.velocityBiases = new double[numLayers][];

        for (int l = 0; l < numLayers; l++) {
            this.velocityWeights[l] = new Matrix(weightShapes[l][0], weightShapes[l][1]);
            this.velocityBiases[l] = new double[biasLengths[l]];
        }
    }

    @Override
    public void update(int layerIdx, Matrix weights, double[] biases, Matrix gradWeights, double[] gradBiases) {
        Objects.requireNonNull(weights, "weights must not be null");
        Objects.requireNonNull(biases, "biases must not be null");
        Objects.requireNonNull(gradWeights, "gradWeights must not be null");
        Objects.requireNonNull(gradBiases, "gradBiases must not be null");

        if (velocityWeights == null || layerIdx >= velocityWeights.length) {
            throw new IllegalStateException("NesterovOptimizer not initialized for layer " + layerIdx);
        }

        Matrix vW = velocityWeights[layerIdx];
        double[] vb = velocityBiases[layerIdx];

        int rows = weights.rows();
        int cols = weights.cols();

        // Nesterov update for weights
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double g = gradWeights.get(r, c);
                double vPrev = vW.get(r, c);
                double vNew = momentumFactor * vPrev + learningRate * g;
                vW.set(r, c, vNew);

                double nesterovStep = momentumFactor * vNew + learningRate * g;
                weights.set(r, c, weights.get(r, c) - nesterovStep);
            }
        }

        // Nesterov update for biases
        for (int b = 0; b < biases.length; b++) {
            double g = gradBiases[b];
            double vPrev = vb[b];
            double vNew = momentumFactor * vPrev + learningRate * g;
            vb[b] = vNew;

            double nesterovStep = momentumFactor * vNew + learningRate * g;
            biases[b] -= nesterovStep;
        }
    }

    public Matrix getVelocityWeights(int layerIdx) {
        return velocityWeights[layerIdx];
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    public double getMomentumFactor() {
        return momentumFactor;
    }
}
