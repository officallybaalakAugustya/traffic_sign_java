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
import com.trafficsign.data.augmentation.ImageAugmentor;
import com.trafficsign.evaluation.ConfusionMatrix;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class BalancedCnnOptimizer {

    public record TrainingMetrics(
            int epoch,
            float trainLoss,
            float valLoss,
            double valAccuracy,
            double valMacroF1,
            double valWeightedF1,
            double valMacroRecall,
            int zeroRecallClasses,
            long epochTimeMs
    ) {
    }

    public record DetailedEvaluation(
            double accuracy,
            double macroF1,
            double weightedF1,
            double macroPrecision,
            double macroRecall,
            int zeroRecallClasses,
            int totalSamples,
            ConfusionMatrix.PerClassMetric[] perClassMetrics,
            ConfusionMatrix confusionMatrix
    ) {
    }

    public record OptimizationRunResult(
            String experimentName,
            List<TrainingMetrics> epochHistory,
            float bestValLoss,
            double bestValMacroF1,
            double bestValAccuracy,
            int bestEpoch,
            Path checkpointPath,
            long totalTrainingTimeMs,
            DetailedEvaluation validationEvaluation,
            DetailedEvaluation testEvaluation
    ) {
    }

    private final CnnTrainingConfig config;

    public BalancedCnnOptimizer(CnnTrainingConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    public OptimizationRunResult trainAndEvaluate(
            String experimentName,
            Block block,
            float[][] trainX,
            int[] trainY,
            float[][] valX,
            int[] valY,
            float[][] testX,
            int[] testY,
            int numClasses,
            int epochs,
            Path checkpointDir,
            float[] classWeights,
            int[] customSampleIndices,
            ImageAugmentor augmentor
    ) throws IOException, ai.djl.MalformedModelException {
        Files.createDirectories(checkpointDir);

        Loss loss;
        if (classWeights != null && classWeights.length == numClasses) {
            loss = new CappedWeightedLoss(classWeights);
        } else {
            loss = Loss.softmaxCrossEntropyLoss();
        }

        Optimizer opt;
        if ("sgd".equalsIgnoreCase(config.optimizerName())) {
            opt = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        } else {
            opt = Optimizer.adam().optLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        }

        DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(loss).optOptimizer(opt);

        List<TrainingMetrics> history = new ArrayList<>();
        float bestValLoss = Float.MAX_VALUE;
        double bestValMacroF1 = -1.0;
        double bestValAcc = 0.0;
        int bestEpoch = -1;
        Path bestModelDir = checkpointDir.resolve(experimentName + "_best");
        Files.createDirectories(bestModelDir);

        long startTotal = System.currentTimeMillis();

        try (Model model = Model.newInstance(experimentName, config.device())) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(trainConfig)) {
                int samplePixels = trainX[0].length;
                int res = (int) Math.round(Math.sqrt(samplePixels / 3));
                Shape inputShape = new Shape(config.batchSize(), 3, res, res);
                trainer.initialize(inputShape);

                int nTrain = (customSampleIndices != null) ? customSampleIndices.length : trainX.length;
                int[] baseIndices = new int[nTrain];
                for (int i = 0; i < nTrain; i++) {
                    baseIndices[i] = (customSampleIndices != null) ? customSampleIndices[i] : i;
                }

                Random rng = new Random(config.randomSeed());

                for (int epoch = 1; epoch <= epochs; epoch++) {
                    long epochStart = System.currentTimeMillis();

                    for (int i = nTrain - 1; i > 0; i--) {
                        int j = rng.nextInt(i + 1);
                        int tmp = baseIndices[i];
                        baseIndices[i] = baseIndices[j];
                        baseIndices[j] = tmp;
                    }

                    float epochTrainLossSum = 0f;
                    int trainBatches = 0;

                    for (int start = 0; start < nTrain; start += config.batchSize()) {
                        int end = Math.min(start + config.batchSize(), nTrain);
                        int curBatch = end - start;

                        float[] bPixels = new float[curBatch * samplePixels];
                        long[] bLabels = new long[curBatch];

                        for (int b = 0; b < curBatch; b++) {
                            int idx = baseIndices[start + b];
                            float[] original = trainX[idx];

                            if (augmentor != null && augmentor.getConfig().enabled()) {
                                float[] aug = augmentPixels(original, augmentor);
                                System.arraycopy(aug, 0, bPixels, b * samplePixels, samplePixels);
                            } else {
                                System.arraycopy(original, 0, bPixels, b * samplePixels, samplePixels);
                            }
                            bLabels[b] = trainY[idx];
                        }

                        try (NDManager batchManager = NDManager.newBaseManager(config.device())) {
                            NDArray x = batchManager.create(bPixels, new Shape(curBatch, 3, res, res));
                            NDArray y = batchManager.create(bLabels, new Shape(curBatch));

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

                    DetailedEvaluation valEval = evaluateDetailed(model, valX, valY, numClasses);
                    float valLoss = evaluateLoss(model, valX, valY, loss);

                    long epochTime = System.currentTimeMillis() - epochStart;

                    TrainingMetrics tm = new TrainingMetrics(
                            epoch,
                            avgTrainLoss,
                            valLoss,
                            valEval.accuracy(),
                            valEval.macroF1(),
                            valEval.weightedF1(),
                            valEval.macroRecall(),
                            valEval.zeroRecallClasses(),
                            epochTime
                    );
                    history.add(tm);

                    if (valEval.macroF1() > bestValMacroF1 || (valEval.macroF1() == bestValMacroF1 && valLoss < bestValLoss)) {
                        bestValMacroF1 = valEval.macroF1();
                        bestValLoss = valLoss;
                        bestValAcc = valEval.accuracy();
                        bestEpoch = epoch;

                        model.setProperty("Epoch", String.valueOf(epoch));
                        model.setProperty("ValMacroF1", String.valueOf(bestValMacroF1));
                        model.setProperty("ValAcc", String.valueOf(bestValAcc));
                        model.save(bestModelDir, experimentName);
                    }
                }
            }
        }

        long totalTime = System.currentTimeMillis() - startTotal;

        DetailedEvaluation finalValEval;
        DetailedEvaluation finalTestEval;
        try (Model evalModel = Model.newInstance(experimentName + "_eval", config.device())) {
            evalModel.setBlock(IndianTrafficSignCnnBlock.build(numClasses));
            evalModel.load(bestModelDir, experimentName);

            finalValEval = evaluateDetailed(evalModel, valX, valY, numClasses);
            finalTestEval = evaluateDetailed(evalModel, testX, testY, numClasses);
        }

        return new OptimizationRunResult(
                experimentName,
                history,
                bestValLoss,
                bestValMacroF1,
                bestValAcc,
                bestEpoch,
                bestModelDir,
                totalTime,
                finalValEval,
                finalTestEval
        );
    }

    public DetailedEvaluation evaluateDetailed(Model model, float[][] datasetX, int[] datasetY, int numClasses) {
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        int n = datasetX.length;
        if (n == 0) {
            return new DetailedEvaluation(0, 0, 0, 0, 0, numClasses, 0, new ConfusionMatrix.PerClassMetric[0], cm);
        }

        int evalBatchSize = config.batchSize();
        int samplePixels = datasetX[0].length;
        int res = (int) Math.round(Math.sqrt(samplePixels / 3));

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int start = 0; start < n; start += evalBatchSize) {
                int end = Math.min(start + evalBatchSize, n);
                int curBatch = end - start;

                float[] bPixels = new float[curBatch * samplePixels];
                for (int b = 0; b < curBatch; b++) {
                    System.arraycopy(datasetX[start + b], 0, bPixels, b * samplePixels, samplePixels);
                }

                NDArray x = manager.create(bPixels, new Shape(curBatch, 3, res, res));
                NDArray logits = model.getBlock().forward(ps, new NDList(x), false).head();
                long[] preds = logits.argMax(1).toLongArray();

                for (int b = 0; b < curBatch; b++) {
                    cm.record(datasetY[start + b], (int) preds[b]);
                }
            }
        }

        ConfusionMatrix.MetricsSummary summary = cm.computeSummary();
        ConfusionMatrix.PerClassMetric[] perClass = summary.perClassMetrics();

        int zeroRecallCount = 0;
        double weightedF1Sum = 0.0;
        for (ConfusionMatrix.PerClassMetric m : perClass) {
            if (m.support() > 0 && m.recall() == 0.0) {
                zeroRecallCount++;
            }
            weightedF1Sum += m.f1() * m.support();
        }
        double weightedF1 = (n > 0) ? (weightedF1Sum / n) : 0.0;

        return new DetailedEvaluation(
                summary.accuracy(),
                summary.macroF1(),
                weightedF1,
                summary.macroPrecision(),
                summary.macroRecall(),
                zeroRecallCount,
                n,
                perClass,
                cm
        );
    }

    public float evaluateLoss(Model model, float[][] datasetX, int[] datasetY, Loss loss) {
        int n = datasetX.length;
        if (n == 0) return 0f;

        int evalBatchSize = config.batchSize();
        int samplePixels = datasetX[0].length;
        int res = (int) Math.round(Math.sqrt(samplePixels / 3));
        float totalLoss = 0f;
        int batches = 0;

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int start = 0; start < n; start += evalBatchSize) {
                int end = Math.min(start + evalBatchSize, n);
                int curBatch = end - start;

                float[] bPixels = new float[curBatch * samplePixels];
                long[] bLabels = new long[curBatch];
                for (int b = 0; b < curBatch; b++) {
                    System.arraycopy(datasetX[start + b], 0, bPixels, b * samplePixels, samplePixels);
                    bLabels[b] = datasetY[start + b];
                }

                NDArray x = manager.create(bPixels, new Shape(curBatch, 3, res, res));
                NDArray y = manager.create(bLabels, new Shape(curBatch));

                NDList preds = model.getBlock().forward(ps, new NDList(x), false);
                NDArray lArr = loss.evaluate(new NDList(y), preds);
                totalLoss += lArr.getFloat();
                batches++;
            }
        }

        return (batches > 0) ? (totalLoss / batches) : 0f;
    }

    private static float[] augmentPixels(float[] chwFlat, ImageAugmentor augmentor) {
        int res = (int) Math.round(Math.sqrt(chwFlat.length / 3));
        int plane = res * res;
        BufferedImage img = new BufferedImage(res, res, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                int r = (int) (Math.max(0f, Math.min(1f, chwFlat[0 * plane + y * res + x])) * 255f);
                int g = (int) (Math.max(0f, Math.min(1f, chwFlat[1 * plane + y * res + x])) * 255f);
                int b = (int) (Math.max(0f, Math.min(1f, chwFlat[2 * plane + y * res + x])) * 255f);
                int rgb = (r << 16) | (g << 8) | b;
                img.setRGB(x, y, rgb);
            }
        }

        BufferedImage aug = augmentor.augment(img);
        float[] outFlat = new float[chwFlat.length];
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                int rgb = aug.getRGB(x, y);
                outFlat[0 * plane + y * res + x] = ((rgb >> 16) & 0xFF) / 255f;
                outFlat[1 * plane + y * res + x] = ((rgb >> 8) & 0xFF) / 255f;
                outFlat[2 * plane + y * res + x] = (rgb & 0xFF) / 255f;
            }
        }
        return outFlat;
    }

    private static class CappedWeightedLoss extends Loss {
        private final float[] weights;

        public CappedWeightedLoss(float[] weights) {
            super("CappedWeightedLoss");
            this.weights = weights.clone();
        }

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray pred = predictions.head();
            NDArray label = labels.head();
            NDManager m = pred.getManager();

            NDArray logSoftmax = pred.logSoftmax(1);
            NDArray oneHot = label.oneHot((int) pred.getShape().get(1));
            NDArray classWeightTensor = m.create(weights);
            NDArray weightedOneHot = oneHot.mul(classWeightTensor);

            NDArray sampleLoss = logSoftmax.mul(weightedOneHot).sum(new int[]{1}).neg();
            return sampleLoss.mean();
        }
    }
}
