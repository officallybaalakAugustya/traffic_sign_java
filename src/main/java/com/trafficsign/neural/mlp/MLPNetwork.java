package com.trafficsign.neural.mlp;

import com.trafficsign.math.Matrix;
import com.trafficsign.neural.activation.ActivationFunction;
import com.trafficsign.neural.activation.ReLUActivation;
import com.trafficsign.neural.activation.Softmax;
import com.trafficsign.neural.loss.CategoricalCrossEntropyLoss;
import com.trafficsign.neural.optimizer.Optimizer;

import java.util.Objects;
import java.util.Random;

/**
 * Configurable Feed-Forward Multi-Layer Perceptron (MLP) implemented in pure Java.
 * Features Xavier/Glorot parameter initialization, forward propagation,
 * analytical backpropagation, and multi-class Softmax cross-entropy loss.
 */
public class MLPNetwork {

    private final int[] layerSizes;
    private final int numLayers; // number of weight matrices = layerSizes.length - 1
    private final ActivationFunction[] activations;
    private final Matrix[] weights;
    private final double[][] biases;
    private final CategoricalCrossEntropyLoss lossFunction;

    public record Gradients(Matrix[] gradWeights, double[][] gradBiases) {
    }

    public record ForwardPassResult(Matrix[] zValues, Matrix[] activations, double[][] outputProbabilities) {
    }

    public MLPNetwork(int[] layerSizes, long randomSeed) {
        this(layerSizes, createDefaultActivations(layerSizes.length - 1), randomSeed);
    }

    public MLPNetwork(int[] layerSizes, ActivationFunction[] activations, long randomSeed) {
        Objects.requireNonNull(layerSizes, "layerSizes must not be null");
        Objects.requireNonNull(activations, "activations must not be null");
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("MLP must have at least an input and output layer (length >= 2)");
        }
        for (int size : layerSizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("Layer size must be positive: " + size);
            }
        }
        if (activations.length != layerSizes.length - 2) {
            throw new IllegalArgumentException("Expected " + (layerSizes.length - 2)
                    + " hidden activation functions, but received " + activations.length);
        }

        this.layerSizes = layerSizes.clone();
        this.numLayers = layerSizes.length - 1;
        this.activations = activations.clone();
        this.weights = new Matrix[numLayers];
        this.biases = new double[numLayers][];
        this.lossFunction = new CategoricalCrossEntropyLoss();

        initializeParameters(randomSeed);
    }

    private static ActivationFunction[] createDefaultActivations(int numWeightLayers) {
        int hiddenLayers = numWeightLayers - 1;
        ActivationFunction[] funcs = new ActivationFunction[hiddenLayers];
        for (int i = 0; i < hiddenLayers; i++) {
            funcs[i] = new ReLUActivation();
        }
        return funcs;
    }

    /**
     * Xavier / Glorot Uniform initialization:
     * W ~ U[-limit, +limit] where limit = sqrt(6 / (fan_in + fan_out))
     */
    private void initializeParameters(long seed) {
        Random rng = new Random(seed);
        for (int l = 0; l < numLayers; l++) {
            int fanIn = layerSizes[l];
            int fanOut = layerSizes[l + 1];
            double limit = Math.sqrt(6.0 / (fanIn + fanOut));

            this.weights[l] = Matrix.random(fanIn, fanOut, rng, -limit, limit);
            this.biases[l] = new double[fanOut]; // Zero initialization for biases
        }
    }

    public ForwardPassResult forward(Matrix inputX) {
        Objects.requireNonNull(inputX, "inputX must not be null");
        if (inputX.cols() != layerSizes[0]) {
            throw new IllegalArgumentException("Input features " + inputX.cols()
                    + " does not match network input dimension " + layerSizes[0]);
        }

        int batchSize = inputX.rows();
        Matrix[] zValues = new Matrix[numLayers];
        Matrix[] actValues = new Matrix[numLayers + 1];
        actValues[0] = inputX;

        for (int l = 0; l < numLayers; l++) {
            Matrix prevA = actValues[l];
            Matrix w = weights[l];
            double[] b = biases[l];

            // Linear transformation Z = A * W + b
            Matrix z = prevA.multiply(w);
            for (int r = 0; r < batchSize; r++) {
                for (int c = 0; c < b.length; c++) {
                    z.set(r, c, z.get(r, c) + b[c]);
                }
            }
            zValues[l] = z;

            // Activation
            if (l < numLayers - 1) {
                // Hidden layer activation
                ActivationFunction act = activations[l];
                Matrix a = z.apply(act::activate);
                actValues[l + 1] = a;
            } else {
                // Output layer: Softmax
                double[][] zArr = z.toArray();
                double[][] probs = Softmax.activateBatch(zArr);
                actValues[l + 1] = new Matrix(probs);
            }
        }

        double[][] outputProbs = actValues[numLayers].toArray();
        return new ForwardPassResult(zValues, actValues, outputProbs);
    }

    public Gradients backward(ForwardPassResult fwd, int[] trueLabels) {
        Objects.requireNonNull(fwd, "ForwardPassResult must not be null");
        Objects.requireNonNull(trueLabels, "trueLabels must not be null");

        Matrix[] actValues = fwd.activations();
        Matrix[] zValues = fwd.zValues();
        double[][] outputProbs = fwd.outputProbabilities();

        int batchSize = actValues[0].rows();
        Matrix[] gradWeights = new Matrix[numLayers];
        double[][] gradBiases = new double[numLayers][];

        // 1. Output error delta^[L] = (1 / N) * (probs - target)
        Matrix delta = lossFunction.computeOutputGradients(outputProbs, trueLabels);

        // 2. Backpropagate layer by layer
        for (int l = numLayers - 1; l >= 0; l--) {
            Matrix prevA = actValues[l];

            // grad_W^[l] = (A^[l-1])^T * delta^[l]
            gradWeights[l] = prevA.transpose().multiply(delta);

            // grad_b^[l] = sum across batch of delta^[l]
            gradBiases[l] = delta.sumCols();

            if (l > 0) {
                // Propagate delta to previous layer: delta^[l-1] = (delta^[l] * (W^[l])^T) . sigma'(Z^[l-1])
                Matrix wT = weights[l].transpose();
                Matrix deltaPrev = delta.multiply(wT);

                ActivationFunction act = activations[l - 1];
                Matrix zPrev = zValues[l - 1];
                Matrix actDeriv = new Matrix(batchSize, layerSizes[l]);

                for (int r = 0; r < batchSize; r++) {
                    for (int c = 0; c < layerSizes[l]; c++) {
                        double zVal = zPrev.get(r, c);
                        double aVal = actValues[l].get(r, c);
                        actDeriv.set(r, c, act.derivative(zVal, aVal));
                    }
                }

                delta = deltaPrev.elementMultiply(actDeriv);
            }
        }

        return new Gradients(gradWeights, gradBiases);
    }

    public double trainBatch(Matrix batchX, int[] batchY, Optimizer optimizer) {
        ForwardPassResult fwd = forward(batchX);
        double loss = lossFunction.computeLoss(fwd.outputProbabilities(), batchY);
        Gradients grads = backward(fwd, batchY);

        for (int l = 0; l < numLayers; l++) {
            optimizer.update(l, weights[l], biases[l], grads.gradWeights()[l], grads.gradBiases()[l]);
        }

        return loss;
    }

    public double evaluateLoss(Matrix X, int[] y) {
        ForwardPassResult fwd = forward(X);
        return lossFunction.computeLoss(fwd.outputProbabilities(), y);
    }

    public int[] predict(Matrix X) {
        ForwardPassResult fwd = forward(X);
        double[][] probs = fwd.outputProbabilities();
        int[] predictions = new int[probs.length];

        for (int i = 0; i < probs.length; i++) {
            int maxIdx = 0;
            double maxP = probs[i][0];
            for (int c = 1; c < probs[i].length; c++) {
                if (probs[i][c] > maxP) {
                    maxP = probs[i][c];
                    maxIdx = c;
                }
            }
            predictions[i] = maxIdx;
        }

        return predictions;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int[] getLayerSizes() {
        return layerSizes.clone();
    }

    public Matrix getWeights(int layer) {
        return weights[layer];
    }

    public double[] getBiases(int layer) {
        return biases[layer].clone();
    }

    public void setWeight(int layer, int r, int c, double val) {
        weights[layer].set(r, c, val);
    }

    public void setBias(int layer, int idx, double val) {
        biases[layer][idx] = val;
    }

    public void setWeights(int layer, Matrix w) {
        this.weights[layer] = Objects.requireNonNull(w, "weights matrix must not be null");
    }

    public void setBiases(int layer, double[] b) {
        this.biases[layer] = Objects.requireNonNull(b, "biases array must not be null").clone();
    }
}

