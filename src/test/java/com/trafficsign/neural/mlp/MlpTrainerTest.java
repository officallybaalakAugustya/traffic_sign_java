package com.trafficsign.neural.mlp;

import com.trafficsign.evaluation.ConfusionMatrix;
import com.trafficsign.math.Matrix;
import com.trafficsign.neural.optimizer.MomentumOptimizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class MlpTrainerTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("MlpTrainer trains on synthetic data and successfully saves and reloads checkpoints")
    void testTrainAndCheckpointReload() throws IOException {
        int numSamples = 20;
        int numFeatures = 32 * 32 * 3;
        int numClasses = 85;

        double[][] trainXData = new double[numSamples][numFeatures];
        int[] trainY = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            trainXData[i][0] = i * 0.05;
            trainY[i] = i % numClasses;
        }

        Matrix trainX = new Matrix(trainXData);
        Matrix valX = trainX;
        int[] valY = trainY;
        Matrix testX = trainX;
        int[] testY = trainY;

        MlpTrainer.LoadedDataset dataset = new MlpTrainer.LoadedDataset(trainX, trainY, valX, valY, testX, testY);

        MLPNetwork net = new MLPNetwork(new int[]{numFeatures, 32, numClasses}, 42L);
        MlpTrainer trainer = new MlpTrainer();
        MomentumOptimizer opt = new MomentumOptimizer(0.01, 0.9);

        MlpTrainer.TrainingRunResult result = trainer.train(
                net, dataset, 2, 8, opt, tempDir, 42L
        );

        assertThat(result.epochHistory()).hasSize(2);
        assertThat(result.checkpointPath()).exists();

        MLPNetwork reloaded = trainer.loadCheckpoint(result.checkpointPath());
        assertThat(reloaded.getLayerSizes()).containsExactly(numFeatures, 32, numClasses);

        ConfusionMatrix cmOrig = trainer.evaluate(net, valX, valY, numClasses);
        ConfusionMatrix cmReloaded = trainer.evaluate(reloaded, valX, valY, numClasses);

        assertThat(cmOrig.computeSummary().accuracy()).isEqualTo(cmReloaded.computeSummary().accuracy());
    }
}
