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
import com.trafficsign.evaluation.ConfusionMatrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class IndianCnnTrainer {

    public record TrainingMetrics(
            int epoch,
            float trainLoss,
            double trainAccuracy,
            float valLoss,
            double valAccuracy,
            double valMacroF1,
            long epochTimeMs
    ) {
    }

    public record TrainingRunResult(
            List<TrainingMetrics> epochHistory,
            float bestValLoss,
            double bestValAccuracy,
            double bestValMacroF1,
            int bestEpoch,
            Path checkpointPath,
            long totalTrainingTimeMs
    ) {
    }

    public record BatchData(float[] flattenedPixels, long[] labels, int batchSize) {
    }

    private final CnnTrainingConfig config;

    public IndianCnnTrainer(CnnTrainingConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    public TrainingRunResult train(
            Block block,
            float[][] trainX,
            int[] trainY,
            float[][] valX,
            int[] valY,
            int numClasses,
            int epochs,
            Path checkpointDir,
            float[] classWeights
    ) throws IOException {
        Files.createDirectories(checkpointDir);

        Loss loss;
        if (classWeights != null && classWeights.length == numClasses) {
            loss = new WeightedCrossEntropyLoss(classWeights);
        } else {
            loss = Loss.softmaxCrossEntropyLoss();
        }

        Optimizer opt;
        if ("sgd".equalsIgnoreCase(config.optimizerName())) {
            opt = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        } else {
            opt = Optimizer.adam().optLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        }

        DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(loss)
                .optOptimizer(opt);

        List<TrainingMetrics> history = new ArrayList<>();
        float bestValLoss = Float.MAX_VALUE;
        double bestValMacroF1 = -1.0;
        double bestValAcc = 0.0;
        int bestEpoch = -1;
        Path bestModelDir = checkpointDir.resolve("indian_cnn_best");
        Files.createDirectories(bestModelDir);

        long startTotal = System.currentTimeMillis();

        try (Model model = Model.newInstance("indian_cnn", config.device())) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(trainConfig)) {
                Shape inputShape = new Shape(config.batchSize(), 3, 32, 32);
                trainer.initialize(inputShape);

                int nTrain = trainX.length;
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

                        float[] bPixels = new float[curBatch * 3 * 32 * 32];
                        long[] bLabels = new long[curBatch];

                        for (int b = 0; b < curBatch; b++) {
                            int idx = indices[start + b];
                            System.arraycopy(trainX[idx], 0, bPixels, b * 3072, 3072);
                            bLabels[b] = trainY[idx];
                        }

                        try (NDManager batchManager = NDManager.newBaseManager(config.device())) {
                            NDArray x = batchManager.create(bPixels, new Shape(curBatch, 3, 32, 32));
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

                    ConfusionMatrix valCm = evaluate(model, valX, valY, numClasses);
                    ConfusionMatrix.MetricsSummary valSummary = valCm.computeSummary();
                    float valLoss = evaluateLoss(model, valX, valY, loss);

                    long epochTime = System.currentTimeMillis() - epochStart;

                    TrainingMetrics tm = new TrainingMetrics(
                            epoch,
                            avgTrainLoss,
                            0.0,
                            valLoss,
                            valSummary.accuracy(),
                            valSummary.macroF1(),
                            epochTime
                    );
                    history.add(tm);

                    if (valSummary.macroF1() > bestValMacroF1 || (valSummary.macroF1() == bestValMacroF1 && valLoss < bestValLoss)) {
                        bestValMacroF1 = valSummary.macroF1();
                        bestValLoss = valLoss;
                        bestValAcc = valSummary.accuracy();
                        bestEpoch = epoch;

                        model.setProperty("Epoch", String.valueOf(epoch));
                        model.setProperty("ValMacroF1", String.valueOf(bestValMacroF1));
                        model.setProperty("ValAcc", String.valueOf(bestValAcc));
                        model.save(bestModelDir, "indian_cnn");
                    }
                }
            }
        }

        long totalTime = System.currentTimeMillis() - startTotal;

        return new TrainingRunResult(
                history,
                bestValLoss,
                bestValAcc,
                bestValMacroF1,
                bestEpoch,
                bestModelDir,
                totalTime
        );
    }

    public ConfusionMatrix evaluate(Model model, float[][] datasetX, int[] datasetY, int numClasses) {
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        int n = datasetX.length;
        if (n == 0) return cm;

        int evalBatchSize = config.batchSize();

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int start = 0; start < n; start += evalBatchSize) {
                int end = Math.min(start + evalBatchSize, n);
                int curBatch = end - start;

                float[] bPixels = new float[curBatch * 3072];
                for (int b = 0; b < curBatch; b++) {
                    System.arraycopy(datasetX[start + b], 0, bPixels, b * 3072, 3072);
                }

                NDArray x = manager.create(bPixels, new Shape(curBatch, 3, 32, 32));
                NDArray logits = model.getBlock().forward(ps, new NDList(x), false).head();
                long[] preds = logits.argMax(1).toLongArray();

                for (int b = 0; b < curBatch; b++) {
                    cm.record(datasetY[start + b], (int) preds[b]);
                }
            }
        }

        return cm;
    }

    public float evaluateLoss(Model model, float[][] datasetX, int[] datasetY, Loss loss) {
        int n = datasetX.length;
        if (n == 0) return 0f;

        int evalBatchSize = config.batchSize();
        float totalLoss = 0f;
        int batches = 0;

        try (NDManager manager = NDManager.newBaseManager(config.device())) {
            ParameterStore ps = new ParameterStore(manager, false);

            for (int start = 0; start < n; start += evalBatchSize) {
                int end = Math.min(start + evalBatchSize, n);
                int curBatch = end - start;

                float[] bPixels = new float[curBatch * 3072];
                long[] bLabels = new long[curBatch];
                for (int b = 0; b < curBatch; b++) {
                    System.arraycopy(datasetX[start + b], 0, bPixels, b * 3072, 3072);
                    bLabels[b] = datasetY[start + b];
                }

                NDArray x = manager.create(bPixels, new Shape(curBatch, 3, 32, 32));
                NDArray y = manager.create(bLabels, new Shape(curBatch));

                NDList preds = model.getBlock().forward(ps, new NDList(x), false);
                NDArray lArr = loss.evaluate(new NDList(y), preds);
                totalLoss += lArr.getFloat();
                batches++;
            }
        }

        return (batches > 0) ? (totalLoss / batches) : 0f;
    }

    private static class WeightedCrossEntropyLoss extends Loss {
        private final float[] weights;

        public WeightedCrossEntropyLoss(float[] weights) {
            super("WeightedCrossEntropyLoss");
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
