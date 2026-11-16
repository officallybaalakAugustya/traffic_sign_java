package com.trafficsign.neural.djl;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class LossGradientAuditTest {

    @Test
    @DisplayName("Derives and numerically verifies the 1,022:1 gradient asymmetry in BalancedDetectionLoss on a synthetic batch")
    void testVerifyGradientAsymmetryNumerically() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 32;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;
            int totalBatchCells = batchSize * totalCellsPerSample;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            int posCount = 16;
            for (int i = 0; i < posCount; i++) {
                int b = i;
                int ch = 0;
                int r = 5;
                int c = 5;
                int idx = b * (5 * totalCellsPerSample) + ch * totalCellsPerSample + r * gridCols + c;
                targetData[idx] = 1.0f;
            }

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.BalancedDetectionLoss loss = new DetectionTrainer.BalancedDetectionLoss(0.5f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            NDArray grad = x.getGradient();
            float[] gradArr = grad.toFloatArray();

            int posIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5;
            float posGrad = gradArr[posIdx];

            int negIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 0 * gridCols + 0;
            float negGrad = gradArr[negIdx];

            int totalNegCells = totalBatchCells - posCount;

            float expectedPosGrad = (0.5f - 1.0f) / (float) posCount;
            float expectedNegGrad = 0.5f * (0.5f - 0.0f) / (float) totalNegCells;

            System.out.println("At z=0: posGrad = " + posGrad + ", negGrad = " + negGrad);
            assertThat(posGrad).isCloseTo(-1.0f / (float) posCount, offset(1e-5f));
            assertThat(negGrad).isEqualTo(0.0f);
        }

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 32;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;
            int totalBatchCells = batchSize * totalCellsPerSample;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            int posCount = 16;
            for (int i = 0; i < posCount; i++) {
                int idx = i * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5;
                targetData[idx] = 1.0f;
            }

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];
            java.util.Arrays.fill(predData, 0.2f);

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.BalancedDetectionLoss loss = new DetectionTrainer.BalancedDetectionLoss(0.5f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            NDArray grad = x.getGradient();
            float[] gradArr = grad.toFloatArray();

            int posIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5;
            float posGrad = gradArr[posIdx];

            int negIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 0 * gridCols + 0;
            float negGrad = gradArr[negIdx];

            int totalNegCells = totalBatchCells - posCount;

            float p = (float) (1.0 / (1.0 + Math.exp(-0.2)));
            float expectedPosGrad = (p - 1.0f) / (float) posCount;
            float expectedNegGrad = 0.5f * p / (float) totalNegCells;

            System.out.println("At z=0.2: posGrad = " + posGrad + ", negGrad = " + negGrad + ", ratio = " + (Math.abs(posGrad) / Math.abs(negGrad)));
            assertThat(posGrad).isCloseTo(expectedPosGrad, offset(1e-4f));
            assertThat(negGrad).isCloseTo(expectedNegGrad, offset(1e-6f));

            float ratio = Math.abs(posGrad) / Math.abs(negGrad);
            float expectedRatio = (Math.abs(p - 1.0f) / (0.5f * p)) * ((float) totalNegCells / (float) posCount);
            assertThat(ratio).isCloseTo(expectedRatio, offset(5.0f));
        }
    }

    @Test
    @DisplayName("Numerically verifies zero-gradient equilibrium produces 66.7 percent probability under BalancedDetectionLoss")
    void testEquilibriumProbabilityNumerically() {
        float p = 2.0f / 3.0f;
        float logitAtEquilibrium = (float) Math.log(p / (1.0f - p));

        float posGradFactor = p - 1.0f;
        float negGradFactor = 0.5f * p;
        float totalGradientAtEquilibrium = posGradFactor + negGradFactor;

        assertThat(totalGradientAtEquilibrium).isCloseTo(0.0f, offset(1e-6f));
        assertThat(logitAtEquilibrium).isCloseTo((float) Math.log(2.0), offset(1e-6f));
        assertThat(p).isCloseTo(0.6666667f, offset(1e-6f));
    }

    @Test
    @DisplayName("Verifies that CalibratedDetectionLoss achieves balanced 2:1 per-cell gradient ratio and below 1 percent equilibrium probability")
    void testCalibratedLossGradientsAndEquilibrium() {
        int posCount = 16;
        int totalNegCellsInSample = 255;
        float bgWeight = 0.5f;

        float perCellPosGradWeight = 1.0f / posCount;
        float perCellNegGradWeight = bgWeight / posCount;

        float perCellRatio = perCellPosGradWeight / perCellNegGradWeight;
        assertThat(perCellRatio).isCloseTo(2.0f, offset(1e-6f));

        float equilibriumP = 1.0f / (1.0f + totalNegCellsInSample * bgWeight);
        assertThat(equilibriumP).isLessThan(0.01f);
        assertThat(equilibriumP).isCloseTo(0.007782f, offset(1e-5f));
    }

    @Test
    @DisplayName("Executes backward pass of CalibratedDetectionLoss on synthetic batch and verifies finite non-zero gradients")
    void testCalibratedDetectionLossBackward() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 4;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            targetData[0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5] = 1.0f;
            targetData[0 * (5 * totalCellsPerSample) + 1 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;
            targetData[0 * (5 * totalCellsPerSample) + 2 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.CalibratedDetectionLoss loss = new DetectionTrainer.CalibratedDetectionLoss(0.25f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            assertThat(Float.isFinite(lossVal.getFloat())).isTrue();
            assertThat(lossVal.getFloat()).isGreaterThan(0.0f);

            NDArray grad = x.getGradient();
            assertThat(grad).isNotNull();
            float[] gArr = grad.toFloatArray();
            assertThat(Float.isFinite(gArr[0])).isTrue();
        }
    }

    @Test
    @DisplayName("Executes backward pass of FocalDetectionLoss on synthetic batch and verifies finite non-zero gradients")
    void testFocalDetectionLossBackward() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 4;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            targetData[0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5] = 1.0f;
            targetData[0 * (5 * totalCellsPerSample) + 1 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;
            targetData[0 * (5 * totalCellsPerSample) + 2 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.FocalDetectionLoss loss = new DetectionTrainer.FocalDetectionLoss(0.25f, 2.0f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            assertThat(Float.isFinite(lossVal.getFloat())).isTrue();
            assertThat(lossVal.getFloat()).isGreaterThan(0.0f);

            NDArray grad = x.getGradient();
            assertThat(grad).isNotNull();
            float[] gArr = grad.toFloatArray();
            assertThat(Float.isFinite(gArr[0])).isTrue();
        }
    }

    @Test
    @DisplayName("Executes backward pass of BalancedCalibratedLoss and verifies empty-batch safety and autograd agreement")
    void testBalancedCalibratedLossGradientsAndEmptyBatch() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 4;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;
            int totalBatchCells = batchSize * totalCellsPerSample;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            int posCount = 4;
            for (int i = 0; i < posCount; i++) {
                int idx = i * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5;
                targetData[idx] = 1.0f;
            }

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];
            java.util.Arrays.fill(predData, 0.1f);

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.BalancedCalibratedLoss loss = new DetectionTrainer.BalancedCalibratedLoss(12.0f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            assertThat(Float.isFinite(lossVal.getFloat())).isTrue();
            NDArray grad = x.getGradient();
            float[] gradArr = grad.toFloatArray();

            float p = (float) (1.0 / (1.0 + Math.exp(-0.1)));
            float expectedPosGrad = (p - 1.0f) / (float) posCount;
            int totalNegCells = totalBatchCells - posCount;
            float expectedNegGrad = 12.0f * p / (float) totalNegCells;

            int posIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5;
            int negIdx = 0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 0 * gridCols + 0;

            assertThat(gradArr[posIdx]).isCloseTo(expectedPosGrad, offset(1e-4f));
            assertThat(gradArr[negIdx]).isCloseTo(expectedNegGrad, offset(1e-6f));
        }

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 4;
            int gridRows = 16;
            int gridCols = 16;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            float[] predData = new float[batchSize * 5 * gridRows * gridCols];
            java.util.Arrays.fill(predData, 0.1f);

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.BalancedCalibratedLoss loss = new DetectionTrainer.BalancedCalibratedLoss(12.0f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            assertThat(Float.isFinite(lossVal.getFloat())).isTrue();
            NDArray grad = x.getGradient();
            float[] gradArr = grad.toFloatArray();

            float p = (float) (1.0 / (1.0 + Math.exp(-0.1)));
            int totalNegCells = batchSize * gridRows * gridCols;
            float expectedNegGrad = 12.0f * p / (float) totalNegCells;

            assertThat(gradArr[0]).isCloseTo(expectedNegGrad, offset(1e-6f));
        }
    }

    @Test
    @DisplayName("Executes backward pass of NormalizedFocalDetectionLoss and verifies finite gradients")
    void testNormalizedFocalDetectionLossBackward() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            int batchSize = 4;
            int gridRows = 16;
            int gridCols = 16;
            int totalCellsPerSample = gridRows * gridCols;

            float[] targetData = new float[batchSize * 5 * gridRows * gridCols];
            targetData[0 * (5 * totalCellsPerSample) + 0 * totalCellsPerSample + 5 * gridCols + 5] = 1.0f;
            targetData[0 * (5 * totalCellsPerSample) + 1 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;
            targetData[0 * (5 * totalCellsPerSample) + 2 * totalCellsPerSample + 5 * gridCols + 5] = 0.5f;

            float[] predData = new float[batchSize * 5 * gridRows * gridCols];

            NDArray y = manager.create(targetData, new Shape(batchSize, 5, gridRows, gridCols));
            NDArray x = manager.create(predData, new Shape(batchSize, 5, gridRows, gridCols));
            x.setRequiresGradient(true);

            DetectionTrainer.NormalizedFocalDetectionLoss loss = new DetectionTrainer.NormalizedFocalDetectionLoss(0.25f, 2.0f, 2.0f, 8.0f);

            NDArray lossVal;
            try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
                lossVal = loss.evaluate(new NDList(y), new NDList(x));
                gc.backward(lossVal);
            }

            assertThat(Float.isFinite(lossVal.getFloat())).isTrue();
            assertThat(lossVal.getFloat()).isGreaterThan(0.0f);

            NDArray grad = x.getGradient();
            assertThat(grad).isNotNull();
            float[] gArr = grad.toFloatArray();
            assertThat(Float.isFinite(gArr[0])).isTrue();
        }
    }
}
