package com.trafficsign.neural.rules;

import java.util.Objects;

/**
 * Hopfield Associative Memory and Energy-Based Model (John Hopfield, 1982).
 * Stores binary/bipolar patterns in a recurrent symmetric weight matrix and
 * minimizes Lyapunov energy:
 * E(s) = - 0.5 * sum_{i != j} W_{ij} * s_i * s_j
 */
public class HopfieldEnergyModel {

    private final int numUnits;
    private final double[][] weights;

    public HopfieldEnergyModel(int numUnits) {
        if (numUnits <= 0) {
            throw new IllegalArgumentException("Number of units must be positive");
        }
        this.numUnits = numUnits;
        this.weights = new double[numUnits][numUnits];
    }

    public void storePattern(int[] pattern) {
        Objects.requireNonNull(pattern, "pattern must not be null");
        if (pattern.length != numUnits) {
            throw new IllegalArgumentException("Pattern length " + pattern.length + " does not match network size " + numUnits);
        }

        for (int val : pattern) {
            if (val != -1 && val != 1) {
                throw new IllegalArgumentException("Hopfield patterns must be bipolar {-1, +1}. Received: " + val);
            }
        }

        // Hebbian outer product storage: W_ij += (1/N) * s_i * s_j (zero diagonal)
        double scale = 1.0 / numUnits;
        for (int i = 0; i < numUnits; i++) {
            for (int j = 0; j < numUnits; j++) {
                if (i != j) {
                    weights[i][j] += scale * pattern[i] * pattern[j];
                }
            }
        }
    }

    public double computeEnergy(int[] state) {
        Objects.requireNonNull(state, "state must not be null");
        double energy = 0.0;
        for (int i = 0; i < numUnits; i++) {
            for (int j = 0; j < numUnits; j++) {
                energy -= 0.5 * weights[i][j] * state[i] * state[j];
            }
        }
        return energy;
    }

    public int[] recall(int[] noisyInput, int maxIterations) {
        Objects.requireNonNull(noisyInput, "noisyInput must not be null");
        int[] state = noisyInput.clone();

        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            for (int i = 0; i < numUnits; i++) {
                double activation = 0.0;
                for (int j = 0; j < numUnits; j++) {
                    activation += weights[i][j] * state[j];
                }
                int newState = activation >= 0.0 ? 1 : -1;
                if (newState != state[i]) {
                    state[i] = newState;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }

        return state;
    }

    public double[][] getWeights() {
        double[][] copy = new double[numUnits][numUnits];
        for (int i = 0; i < numUnits; i++) {
            System.arraycopy(weights[i], 0, copy[i], 0, numUnits);
        }
        return copy;
    }
}
