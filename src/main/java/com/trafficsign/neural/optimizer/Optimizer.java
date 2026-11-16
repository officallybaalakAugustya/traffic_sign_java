package com.trafficsign.neural.optimizer;

import com.trafficsign.math.Matrix;

/**
 * Common optimizer interface for neural network parameter updates.
 */
public interface Optimizer {

    void initialize(int numLayers, int[][] weightShapes, int[] biasLengths);

    void update(int layerIdx, Matrix weights, double[] biases, Matrix gradWeights, double[] gradBiases);

    double getLearningRate();
}
