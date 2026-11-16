package com.trafficsign.neural.rules;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LearningRulesTest {

    @Test
    @DisplayName("Hebbian learning with Oja's rule prevents unbounded weight explosion")
    void shouldVerifyOjaHebbianBoundedWeights() {
        HebbianLearning hebb = new HebbianLearning(2, 0.05, true);
        double[] input = {1.0, 1.0};

        // Train for 100 iterations
        for (int i = 0; i < 100; i++) {
            hebb.trainSample(input);
        }

        double[] w = hebb.getWeights();
        // Under Oja's rule, ||w||_2 converges to 1.0
        double norm = Math.sqrt(w[0] * w[0] + w[1] * w[1]);
        assertThat(norm).isCloseTo(1.0, org.assertj.core.data.Offset.offset(0.05));
    }

    @Test
    @DisplayName("Competitive learning moves winning prototype towards presented input clusters")
    void shouldMoveWinnerTowardsCluster() {
        // 2 prototypes in 2D space
        CompetitiveLearning competitive = new CompetitiveLearning(2, 2, 0.1, 42L);

        // Repeatedly present samples around cluster (0.9, 0.9)
        double[] sampleClusterA = {0.9, 0.9};
        for (int i = 0; i < 50; i++) {
            competitive.trainSample(sampleClusterA);
        }

        int winner = competitive.findWinner(sampleClusterA);
        double[][] prototypes = competitive.getPrototypes();

        // Winning prototype should have moved close to (0.9, 0.9)
        assertThat(prototypes[winner][0]).isGreaterThan(0.7);
        assertThat(prototypes[winner][1]).isGreaterThan(0.7);
    }

    @Test
    @DisplayName("Error-correction learning converges to zero error on linearly separable data")
    void shouldConvergeErrorCorrection() {
        // Learn logical OR gate: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
        ErrorCorrectionLearning perceptron = new ErrorCorrectionLearning(2, 0.1);
        double[][] X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        int[] y = {0, 1, 1, 1};

        // Train for 20 epochs
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int i = 0; i < X.length; i++) {
                perceptron.trainSample(X[i], y[i]);
            }
        }

        for (int i = 0; i < X.length; i++) {
            assertThat(perceptron.predict(X[i])).isEqualTo(y[i]);
        }
    }

    @Test
    @DisplayName("Hopfield network retrieves stored bipolar pattern from a corrupted probe and minimizes Lyapunov energy")
    void shouldRetrievePatternAndMinimizeEnergy() {
        int n = 4;
        HopfieldEnergyModel hopfield = new HopfieldEnergyModel(n);

        // Store pattern: [+1, -1, +1, -1]
        int[] pattern = {1, -1, 1, -1};
        hopfield.storePattern(pattern);

        double storedEnergy = hopfield.computeEnergy(pattern);

        // Corrupt pattern by flipping 1 bit: [+1, +1, +1, -1]
        int[] corrupted = {1, 1, 1, -1};
        double corruptedEnergy = hopfield.computeEnergy(corrupted);

        // Corrupted state has higher energy than stable state
        assertThat(corruptedEnergy).isGreaterThan(storedEnergy);

        // Recall pattern
        int[] recalled = hopfield.recall(corrupted, 10);
        assertThat(recalled).containsExactly(pattern);

        double recalledEnergy = hopfield.computeEnergy(recalled);
        assertThat(recalledEnergy).isEqualTo(storedEnergy);
    }
}
