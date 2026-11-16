package com.trafficsign.neural.djl;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class BalancedDetectionLossTest {

    @Test
    @DisplayName("BalancedDetectionLoss computes finite loss and valid gradients on imbalanced grid")
    void testLossAndGradients() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            DetectionTrainer.BalancedDetectionLoss loss = new DetectionTrainer.BalancedDetectionLoss(1.0f, 5.0f);

            float[] target = new float[1 * 5 * 8 * 8];
            target[0 * 64 + 3 * 8 + 4] = 1.0f;
            target[1 * 64 + 3 * 8 + 4] = 0.5f;
            target[2 * 64 + 3 * 8 + 4] = 0.5f;
            target[3 * 64 + 3 * 8 + 4] = 0.0f;
            target[4 * 64 + 3 * 8 + 4] = 0.0f;

            float[] pred = new float[1 * 5 * 8 * 8];

            NDArray y = manager.create(target, new Shape(1, 5, 8, 8));
            NDArray x = manager.create(pred, new Shape(1, 5, 8, 8));
            x.setRequiresGradient(true);

            NDArray lossVal = loss.evaluate(new NDList(y), new NDList(x));
            float val = lossVal.getFloat();

            assertThat(Float.isFinite(val)).isTrue();
            assertThat(val).isGreaterThan(0.0f);
        }
    }
}
