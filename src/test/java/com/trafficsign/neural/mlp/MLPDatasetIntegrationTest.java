package com.trafficsign.neural.mlp;

import com.trafficsign.data.ImagePreprocessor;
import com.trafficsign.data.TrafficSignClass;
import com.trafficsign.math.Matrix;
import com.trafficsign.neural.optimizer.BatchGradientDescent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.assertj.core.api.Assertions.assertThat;

class MLPDatasetIntegrationTest {

    @Test
    @DisplayName("Should seamlessly integrate ImagePreprocessor 32x32x3 tensors with MLPNetwork 3072 input")
    void shouldIntegratePreprocessorWithMLP() {
        ImagePreprocessor preprocessor = new ImagePreprocessor(32, 32, false);

        // Create 3 synthetic sample images representing 3 classes (e.g. 0: 20km/h, 1: 30km/h, 14: Stop)
        int[] classIds = new int[]{0, 1, 14};
        int numSamples = classIds.length;
        int inputFeatures = 32 * 32 * 3; // 3072

        double[][] batchX = new double[numSamples][inputFeatures];

        for (int i = 0; i < numSamples; i++) {
            BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
            // Unique color per sample to simulate sign color signatures
            int rgb = (i * 80 << 16) | (i * 40 << 8) | (200 - i * 50);
            for (int y = 0; y < 32; y++) {
                for (int x = 0; x < 32; x++) {
                    img.setRGB(x, y, rgb);
                }
            }

            // Convert to CHW float tensor [3, 32, 32]
            float[][][] tensor = preprocessor.toTensorCHW(img);

            // Flatten [3, 32, 32] into 3072 double vector
            int idx = 0;
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < 32; y++) {
                    for (int x = 0; x < 32; x++) {
                        batchX[i][idx++] = tensor[c][y][x];
                    }
                }
            }
            assertThat(idx).isEqualTo(inputFeatures);
        }

        // Configure MLP with input 3072 and output 43 (TrafficSignClass.totalClasses())
        int[] layerSizes = {3072, 64, TrafficSignClass.totalClasses()};
        MLPNetwork mlp = new MLPNetwork(layerSizes, 42L);

        Matrix inputMatrix = new Matrix(batchX);

        // Forward pass
        MLPNetwork.ForwardPassResult fwd = mlp.forward(inputMatrix);
        double[][] outputProbs = fwd.outputProbabilities();

        assertThat(outputProbs.length).isEqualTo(numSamples);
        assertThat(outputProbs[0].length).isEqualTo(TrafficSignClass.totalClasses());

        // Perform one training step with BatchGradientDescent
        BatchGradientDescent optimizer = new BatchGradientDescent(0.01);
        double initialLoss = mlp.trainBatch(inputMatrix, classIds, optimizer);

        assertThat(initialLoss).isPositive().isNotNaN().isNotInfinite();

        // Predictions map to valid TrafficSignClass entries
        int[] predictions = mlp.predict(inputMatrix);
        for (int pred : predictions) {
            assertThat(TrafficSignClass.fromId(pred)).isPresent();
        }
    }
}
