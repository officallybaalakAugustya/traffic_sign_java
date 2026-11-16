package com.trafficsign.neural.djl;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.data.detection.DetectionUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class DetectionTrainer {

    public record TrainingMetrics(
            int epoch,
            float trainLoss,
            float valLoss,
            double precision,
            double recall,
            double ap50,
            double f1,
            long epochTimeMs
    ) {
    }

    public record TrainingRunResult(
            List<TrainingMetrics> history,
            float bestValLoss,
            double bestAp50,
            double bestF1,
            int bestEpoch,
            Path checkpointPath,
            long totalTrainingTimeMs
    ) {
    }

    public record EvaluationMetrics(
            double precision,
            double recall,
            double ap50,
            double f1,
            int tp,
            int fp,
            int fn,
            int totalGroundTruth,
            double averageLatencyMs
    ) {
    }

    private final CnnTrainingConfig config;

    public DetectionTrainer(CnnTrainingConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    public TrainingRunResult train(
            Block block,
            float[][] trainImages,
            float[][][][] trainTargets,
            float[][] valImages,
            float[][][][] valTargets,
            List<List<BoundingBox>> valGroundTruthBoxes,
            int epochs,
            Path checkpointDir
    ) throws IOException {
        return train(block, trainImages, trainTargets, valImages, valTargets, valGroundTruthBoxes, epochs, checkpointDir, new DetectionGridLoss(), 0.30f, "detector_best");
    }

    public TrainingRunResult train(
            Block block,
            float[][] trainImages,
            float[][][][] trainTargets,
            float[][] valImages,
            float[][][][] valTargets,
            List<List<BoundingBox>> valGroundTruthBoxes,
            int epochs,
            Path checkpointDir,
            Loss loss,
            float evalConfThreshold,
            String checkpointSubdir
    ) throws IOException {
        Files.createDirectories(checkpointDir);

        Optimizer opt = Optimizer.adam().optLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(loss).optOptimizer(opt);

        List<TrainingMetrics> history = new ArrayList<>();
        float bestValLoss = Float.MAX_VALUE;
        double bestAp50 = -1.0;
        double bestF1 = -1.0;
        int bestEpoch = -1;
        Path bestModelDir = checkpointDir.resolve(checkpointSubdir);
        Files.createDirectories(bestModelDir);

        int gridRows = trainTargets.length > 0 && trainTargets[0].length > 0 ? trainTargets[0][0].length : 8;
        int gridCols = trainTargets.length > 0 && trainTargets[0].length > 0 && trainTargets[0][0].length > 0 ? trainTargets[0][0][0].length : 8;
        int targetElementsPerSample = 5 * gridRows * gridCols;

        long startTotal = System.currentTimeMillis();

        try (Model model = Model.newInstance("sign_detector", config.device())) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(trainConfig)) {
                Shape inputShape = new Shape(config.batchSize(), 3, 128, 128);
                trainer.initialize(inputShape);

                int nTrain = trainImages.length;
                int[] indices = new int[nTrain];
                for (int i = 0; i < nTrain; i++) indices[i] = i;
                Random rng = new Random(config.randomSeed());

                for (int epoch = 1; epoch <= epochs; epoch++) {
                    long epochStart = System.currentTimeMillis();

                    for (int i = nTrain - 1; i > 0; i--) {
                        int j = rng.nextInt(i + 1);
                        int tmp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = tmp;
                    }

                    float epochTrainLossSum = 0f;
                    int trainBatches = 0;

                    for (int start = 0; start < nTrain; start += config.batchSize()) {
                        int end = Math.min(start + config.batchSize(), nTrain);
                        int curBatch = end - start;

                        float[] bPixels = new float[curBatch * 3 * 128 * 128];
                        float[] bTargets = new float[curBatch * targetElementsPerSample];

                        for (int b = 0; b < curBatch; b++) {
                            int idx = indices[start + b];
                            System.arraycopy(trainImages[idx], 0, bPixels, b * 49152, 49152);

                            float[][][] tGrid = trainTargets[idx];
                            int tIdx = b * targetElementsPerSample;
                            for (int ch = 0; ch < 5; ch++) {
                                for (int r = 0; r < gridRows; r++) {
                                    for (int c = 0; c < gridCols; c++) {
                                        bTargets[tIdx++] = tGrid[ch][r][c];
                                    }
                                }
                            }
                        }

                        try (NDManager batchManager = NDManager.newBaseManager(config.device())) {
                            NDArray x = batchManager.create(bPixels, new Shape(curBatch, 3, 128, 128));
                            NDArray y = batchManager.create(bTargets, new Shape(curBatch, 5, gridRows, gridCols));

                            try (GradientCollector gc = trainer.newGradientCollector()) {
                                NDList preds = trainer.forward(new NDList(x));
                                NDArray lossArr = loss.evaluate(new NDList(y), preds);
                                float lVal = lossArr.getFloat();
                                epochTrainLossSum += lVal;
                                trainBatches++;

                                gc.backward(lossArr);
                            }
                            trainer.step();
                        }
                    }

                    float avgTrainLoss = (trainBatches > 0) ? (epochTrainLossSum / trainBatches) : 0f;

                    float valLoss = evaluateLoss(model, valImages, valTargets, loss);
                    EvaluationMetrics valMetrics = evaluateDetections(model, valImages, valGroundTruthBoxes, evalConfThreshold, 0.50);

                    long epochTime = System.currentTimeMillis() - epochStart;

                    TrainingMetrics tm = new TrainingMetrics(
                            epoch,
                            avgTrainLoss,
                            valLoss,
                            valMetrics.precision(),
                            valMetrics.recall(),
                            valMetrics.ap50(),
                            valMetrics.f1(),
                            epochTime
                    );
                    history.add(tm);

                    boolean isBetter = false;
                    if (valMetrics.f1() > bestF1) {
                        isBetter = true;
                    } else if (Math.abs(valMetrics.f1() - bestF1) < 1e-6 && valMetrics.ap50() > bestAp50) {
                        isBetter = true;
                    } else if (Math.abs(valMetrics.f1() - bestF1) < 1e-6 && Math.abs(valMetrics.ap50() - bestAp50) < 1e-6 && valLoss < bestValLoss) {
                        isBetter = true;
                    }

                    if (isBetter) {
                        bestAp50 = valMetrics.ap50();
                        bestF1 = valMetrics.f1();
                        bestValLoss = valLoss;
                        bestEpoch = epoch;

                        model.setProperty("Epoch", String.valueOf(epoch));
                        model.setProperty("ValAp50", String.valueOf(bestAp50));
                        model.setProperty("ValF1", String.valueOf(bestF1));
                        model.setProperty("ValLoss", String.valueOf(valLoss));
                        model.save(bestModelDir, "sign_detector");
                    }
                }
            }
        }

        long totalTime = System.currentTimeMillis() - startTotal;

        return new TrainingRunResult(
                history,
                bestValLoss,
                bestAp50,
                bestF1,
                bestEpoch,
                bestModelDir,
                totalTime
        );
    }

    public float evaluateLoss(Model model, float[][] images, float[][][][] targets, Loss loss) {
        int n = images.length;
        if (n == 0) return 0f;

        int gridRows = targets.length > 0 && targets[0].length > 0 ? targets[0][0].length : 8;
        int gridCols = targets.length > 0 && targets[0].length > 0 && targets[0][0].length > 0 ? targets[0][0][0].length : 8;
        int targetElementsPerSample = 5 * gridRows * gridCols;

        int batchSize = config.batchSize();
        float totalLoss = 0f;
        int batches = 0;

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(start + batchSize, n);
                int curBatch = end - start;

                float[] bPixels = new float[curBatch * 3 * 128 * 128];
                float[] bTargets = new float[curBatch * targetElementsPerSample];

                for (int b = 0; b < curBatch; b++) {
                    System.arraycopy(images[start + b], 0, bPixels, b * 49152, 49152);
                    float[][][] tGrid = targets[start + b];
                    int tIdx = b * targetElementsPerSample;
                    for (int ch = 0; ch < 5; ch++) {
                        for (int r = 0; r < gridRows; r++) {
                            for (int c = 0; c < gridCols; c++) {
                                bTargets[tIdx++] = tGrid[ch][r][c];
                            }
                        }
                    }
                }

                NDArray x = manager.create(bPixels, new Shape(curBatch, 3, 128, 128));
                NDArray y = manager.create(bTargets, new Shape(curBatch, 5, gridRows, gridCols));

                NDList preds = model.getBlock().forward(ps, new NDList(x), false);
                NDArray lArr = loss.evaluate(new NDList(y), preds);
                totalLoss += lArr.getFloat();
                batches++;
            }
        }

        return (batches > 0) ? (totalLoss / batches) : 0f;
    }

    public EvaluationMetrics evaluateDetections(
            Model model,
            float[][] images,
            List<List<BoundingBox>> groundTruthBoxes,
            float confThreshold,
            double iouThreshold
    ) {
        int n = images.length;
        if (n == 0) {
            return new EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        int tp = 0;
        int fp = 0;
        int fn = 0;
        int totalGt = 0;

        long inferTimeSum = 0;

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int i = 0; i < n; i++) {
                List<BoundingBox> gtList = groundTruthBoxes.get(i);
                totalGt += gtList.size();

                NDArray x = manager.create(images[i], new Shape(1, 3, 128, 128));

                long t0 = System.nanoTime();
                NDArray out = model.getBlock().forward(ps, new NDList(x), false).head();
                inferTimeSum += (System.nanoTime() - t0);

                Shape outShape = out.getShape();
                int gridRows = (int) outShape.get(outShape.dimension() - 2);
                int gridCols = (int) outShape.get(outShape.dimension() - 1);

                float[] flatOut = out.toFloatArray();
                float[][][] gridTensor = new float[5][gridRows][gridCols];
                int idx = 0;
                for (int ch = 0; ch < 5; ch++) {
                    for (int r = 0; r < gridRows; r++) {
                        for (int c = 0; c < gridCols; c++) {
                            gridTensor[ch][r][c] = flatOut[idx++];
                        }
                    }
                }

                List<DetectionUtils.DetectionPrediction> rawDets = DetectionUtils.decodeGridPredictions(
                        gridTensor, confThreshold, gridCols, gridRows
                );
                List<DetectionUtils.DetectionPrediction> finalDets = DetectionUtils.nonMaximumSuppression(
                        rawDets, 0.40
                );

                boolean[] gtMatched = new boolean[gtList.size()];

                for (var pred : finalDets) {
                    double bestIou = 0.0;
                    int bestGtIdx = -1;

                    for (int g = 0; g < gtList.size(); g++) {
                        if (gtMatched[g]) continue;
                        double iou = pred.box().computeIoU(gtList.get(g));
                        if (iou > bestIou) {
                            bestIou = iou;
                            bestGtIdx = g;
                        }
                    }

                    if (bestIou >= iouThreshold && bestGtIdx >= 0) {
                        tp++;
                        gtMatched[bestGtIdx] = true;
                    } else {
                        fp++;
                    }
                }

                for (boolean matched : gtMatched) {
                    if (!matched) fn++;
                }
            }
        }

        double precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
        double f1 = (precision + recall > 1e-9) ? (2.0 * precision * recall) / (precision + recall) : 0.0;
        double ap50 = precision * recall;
        double avgLatencyMs = ((double) inferTimeSum / n) / 1_000_000.0;

        return new EvaluationMetrics(precision, recall, ap50, f1, tp, fp, fn, totalGt, avgLatencyMs);
    }

    public static class DetectionGridLoss extends Loss {
        public DetectionGridLoss() {
            super("DetectionGridLoss");
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray confLoss = predConf.sub(targetConf).square();

            NDArray mask = targetConf;

            NDArray predBox = pred.get(":, 1:5, :, :");
            NDArray targetBox = target.get(":, 1:5, :, :");

            NDArray boxLoss = predBox.sub(targetBox).square().mul(mask);

            NDArray totalLoss = confLoss.add(boxLoss.mul(5.0f)).mean();
            return totalLoss;
        }
    }

    public static class BalancedDetectionLoss extends Loss {
        private final float bgWeight;
        private final float boxWeight;

        public BalancedDetectionLoss() {
            this(0.5f, 5.0f);
        }

        public BalancedDetectionLoss(float bgWeight, float boxWeight) {
            super("BalancedDetectionLoss");
            this.bgWeight = bgWeight;
            this.boxWeight = boxWeight;
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray relu = ai.djl.nn.Activation.relu(predConf);
            NDArray softplus = ai.djl.nn.Activation.softPlus(predConf.abs().neg());
            NDArray bce = relu.sub(predConf.mul(targetConf)).add(softplus);

            NDArray posMask = targetConf;
            NDArray negMask = posMask.neg().add(1.0f);

            NDArray posCount = posMask.sum();
            float nPos = posCount.getFloat();
            float normPos = Math.max(1.0f, nPos);

            NDArray negCount = negMask.sum();
            float nNeg = negCount.getFloat();
            float normNeg = Math.max(1.0f, nNeg);

            NDArray posConfLoss = bce.mul(posMask).sum().div(normPos);
            NDArray negConfLoss = bce.mul(negMask).sum().div(normNeg);

            NDArray confLoss = posConfLoss.add(negConfLoss.mul(bgWeight));

            NDArray predXY = ai.djl.nn.Activation.sigmoid(pred.get(":, 1:3, :, :"));
            NDArray targetXY = target.get(":, 1:3, :, :");
            NDArray xyLoss = predXY.sub(targetXY).square().mul(posMask).sum().div(normPos);

            NDArray predWH = pred.get(":, 3:5, :, :");
            NDArray targetWH = target.get(":, 3:5, :, :");
            NDArray whLoss = predWH.sub(targetWH).square().mul(posMask).sum().div(normPos);

            NDArray boxLoss = xyLoss.add(whLoss);

            return confLoss.add(boxLoss.mul(boxWeight));
        }
    }

    public static class CalibratedDetectionLoss extends Loss {
        private final float bgWeight;
        private final float boxWeight;

        public CalibratedDetectionLoss() {
            this(0.25f, 8.0f);
        }

        public CalibratedDetectionLoss(float bgWeight, float boxWeight) {
            super("CalibratedDetectionLoss");
            this.bgWeight = bgWeight;
            this.boxWeight = boxWeight;
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray relu = ai.djl.nn.Activation.relu(predConf);
            NDArray softplus = ai.djl.nn.Activation.softPlus(predConf.abs().neg());
            NDArray bce = relu.sub(predConf.mul(targetConf)).add(softplus);

            NDArray posMask = targetConf;
            NDArray negMask = posMask.neg().add(1.0f);

            NDArray posCount = posMask.sum();
            float nPos = posCount.getFloat();
            float normPos = Math.max(1.0f, nPos);

            NDArray posConfLoss = bce.mul(posMask).sum().div(normPos);
            NDArray negConfLoss = bce.mul(negMask).sum().div(normPos);

            NDArray confLoss = posConfLoss.add(negConfLoss.mul(bgWeight));

            NDArray predXY = ai.djl.nn.Activation.sigmoid(pred.get(":, 1:3, :, :"));
            NDArray targetXY = target.get(":, 1:3, :, :");
            NDArray xyLoss = predXY.sub(targetXY).square().mul(posMask).sum().div(normPos);

            NDArray predWH = pred.get(":, 3:5, :, :");
            NDArray targetWH = target.get(":, 3:5, :, :");
            NDArray whLoss = predWH.sub(targetWH).square().mul(posMask).sum().div(normPos);

            NDArray boxLoss = xyLoss.add(whLoss);

            return confLoss.add(boxLoss.mul(boxWeight));
        }
    }

    public static class FocalDetectionLoss extends Loss {
        private final float alpha;
        private final float gamma;
        private final float boxWeight;

        public FocalDetectionLoss() {
            this(0.25f, 2.0f, 8.0f);
        }

        public FocalDetectionLoss(float alpha, float gamma, float boxWeight) {
            super("FocalDetectionLoss");
            this.alpha = alpha;
            this.gamma = gamma;
            this.boxWeight = boxWeight;
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray p = ai.djl.nn.Activation.sigmoid(predConf);
            NDArray relu = ai.djl.nn.Activation.relu(predConf);
            NDArray softplus = ai.djl.nn.Activation.softPlus(predConf.abs().neg());
            NDArray bce = relu.sub(predConf.mul(targetConf)).add(softplus);

            NDArray posMask = targetConf;
            NDArray negMask = posMask.neg().add(1.0f);

            NDArray posModulator = p.neg().add(1.0f).square();
            NDArray negModulator = p.square();

            NDArray posCount = posMask.sum();
            float nPos = posCount.getFloat();
            float normPos = Math.max(1.0f, nPos);

            NDArray posConfLoss = bce.mul(posModulator).mul(alpha).mul(posMask).sum().div(normPos);
            NDArray negConfLoss = bce.mul(negModulator).mul(1.0f - alpha).mul(negMask).sum().div(normPos);

            NDArray confLoss = posConfLoss.add(negConfLoss);

            NDArray predXY = ai.djl.nn.Activation.sigmoid(pred.get(":, 1:3, :, :"));
            NDArray targetXY = target.get(":, 1:3, :, :");
            NDArray xyLoss = predXY.sub(targetXY).square().mul(posMask).sum().div(normPos);

            NDArray predWH = pred.get(":, 3:5, :, :");
            NDArray targetWH = target.get(":, 3:5, :, :");
            NDArray whLoss = predWH.sub(targetWH).square().mul(posMask).sum().div(normPos);

            NDArray boxLoss = xyLoss.add(whLoss);

            return confLoss.add(boxLoss.mul(boxWeight));
        }
    }

    public static class BalancedCalibratedLoss extends Loss {
        private final float bgWeight;
        private final float boxWeight;

        public BalancedCalibratedLoss() {
            this(12.0f, 8.0f);
        }

        public BalancedCalibratedLoss(float bgWeight, float boxWeight) {
            super("BalancedCalibratedLoss");
            this.bgWeight = bgWeight;
            this.boxWeight = boxWeight;
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray relu = ai.djl.nn.Activation.relu(predConf);
            NDArray softplus = ai.djl.nn.Activation.softPlus(predConf.abs().neg());
            NDArray bce = relu.sub(predConf.mul(targetConf)).add(softplus);

            NDArray posMask = targetConf;
            NDArray negMask = posMask.neg().add(1.0f);

            NDArray posCount = posMask.sum();
            float nPos = posCount.getFloat();
            float normPos = Math.max(1.0f, nPos);

            NDArray negCount = negMask.sum();
            float nNeg = negCount.getFloat();
            float normNeg = Math.max(1.0f, nNeg);

            NDArray posConfLoss = bce.mul(posMask).sum().div(normPos);
            NDArray negConfLoss = bce.mul(negMask).sum().div(normNeg);

            NDArray confLoss = posConfLoss.add(negConfLoss.mul(bgWeight));

            NDArray predXY = ai.djl.nn.Activation.sigmoid(pred.get(":, 1:3, :, :"));
            NDArray targetXY = target.get(":, 1:3, :, :");
            NDArray xyLoss = predXY.sub(targetXY).square().mul(posMask).sum().div(normPos);

            NDArray predWH = pred.get(":, 3:5, :, :");
            NDArray targetWH = target.get(":, 3:5, :, :");
            NDArray whLoss = predWH.sub(targetWH).square().mul(posMask).sum().div(normPos);

            NDArray boxLoss = xyLoss.add(whLoss);

            return confLoss.add(boxLoss.mul(boxWeight));
        }
    }

    public static class NormalizedFocalDetectionLoss extends Loss {
        private final float alpha;
        private final float gamma;
        private final float bgWeight;
        private final float boxWeight;

        public NormalizedFocalDetectionLoss() {
            this(0.25f, 2.0f, 2.0f, 8.0f);
        }

        public NormalizedFocalDetectionLoss(float alpha, float gamma, float bgWeight, float boxWeight) {
            super("NormalizedFocalDetectionLoss");
            this.alpha = alpha;
            this.gamma = gamma;
            this.bgWeight = bgWeight;
            this.boxWeight = boxWeight;
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray target = labels.head();

            NDArray predConf = pred.get(":, 0:1, :, :");
            NDArray targetConf = target.get(":, 0:1, :, :");

            NDArray p = ai.djl.nn.Activation.sigmoid(predConf);
            NDArray relu = ai.djl.nn.Activation.relu(predConf);
            NDArray softplus = ai.djl.nn.Activation.softPlus(predConf.abs().neg());
            NDArray bce = relu.sub(predConf.mul(targetConf)).add(softplus);

            NDArray posMask = targetConf;
            NDArray negMask = posMask.neg().add(1.0f);

            NDArray posModulator = p.neg().add(1.0f).square();
            NDArray negModulator = p.square();

            NDArray posCount = posMask.sum();
            float nPos = posCount.getFloat();
            float normPos = Math.max(1.0f, nPos);

            NDArray negCount = negMask.sum();
            float nNeg = negCount.getFloat();
            float normNeg = Math.max(1.0f, nNeg);

            NDArray posConfLoss = bce.mul(posModulator).mul(alpha).mul(posMask).sum().div(normPos);
            NDArray negConfLoss = bce.mul(negModulator).mul(1.0f - alpha).mul(negMask).sum().div(normNeg);

            NDArray confLoss = posConfLoss.add(negConfLoss.mul(bgWeight));

            NDArray predXY = ai.djl.nn.Activation.sigmoid(pred.get(":, 1:3, :, :"));
            NDArray targetXY = target.get(":, 1:3, :, :");
            NDArray xyLoss = predXY.sub(targetXY).square().mul(posMask).sum().div(normPos);

            NDArray predWH = pred.get(":, 3:5, :, :");
            NDArray targetWH = target.get(":, 3:5, :, :");
            NDArray whLoss = predWH.sub(targetWH).square().mul(posMask).sum().div(normPos);

            NDArray boxLoss = xyLoss.add(whLoss);

            return confLoss.add(boxLoss.mul(boxWeight));
        }
    }
}
