package com.trafficsign.neural.mlp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.data.DuplicateSafeSplitManifest;
import com.trafficsign.data.ImagePreprocessor;
import com.trafficsign.evaluation.ConfusionMatrix;
import com.trafficsign.math.Matrix;
import com.trafficsign.neural.optimizer.MomentumOptimizer;
import com.trafficsign.neural.optimizer.Optimizer;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class MlpTrainer {

    public record TrainingMetrics(
            int epoch,
            double trainLoss,
            double trainAccuracy,
            double valLoss,
            double valAccuracy,
            double valMacroF1,
            long epochTimeMs
    ) {
    }

    public record TrainingRunResult(
            List<TrainingMetrics> epochHistory,
            double bestValLoss,
            double bestValAccuracy,
            double bestValMacroF1,
            int bestEpoch,
            Path checkpointPath,
            long totalTrainingTimeMs
    ) {
    }

    private final ImagePreprocessor preprocessor;
    private final ObjectMapper objectMapper;

    public MlpTrainer() {
        this(32);
    }

    public MlpTrainer(int targetResolution) {
        this.preprocessor = new ImagePreprocessor(targetResolution, targetResolution, false);
        this.objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public record LoadedDataset(
            Matrix trainX,
            int[] trainY,
            Matrix valX,
            int[] valY,
            Matrix cleanTestX,
            int[] cleanTestY
    ) {
    }

    public LoadedDataset loadFromManifest(
            Path datasetRoot,
            DuplicateSafeSplitManifest.ManifestSummary manifest
    ) throws IOException {
        List<double[]> trainVectors = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();
        List<double[]> valVectors = new ArrayList<>();
        List<Integer> valLabels = new ArrayList<>();
        List<double[]> testVectors = new ArrayList<>();
        List<Integer> testLabels = new ArrayList<>();

        for (DuplicateSafeSplitManifest.PartitionEntry entry : manifest.entries()) {
            if (entry.partition() == DuplicateSafeSplitManifest.PartitionCategory.EXCLUDED_TRAIN_OVERLAP) {
                continue;
            }
            Path imgPath = datasetRoot.resolve(entry.relativePath());
            if (!Files.exists(imgPath)) {
                continue;
            }

            BufferedImage img = preprocessor.loadImage(imgPath);
            float[][][] chw = preprocessor.toTensorCHW(img);
            double[] flat = flatten(chw);

            switch (entry.partition()) {
                case TRAIN -> {
                    trainVectors.add(flat);
                    trainLabels.add(entry.classId());
                }
                case VALIDATION -> {
                    valVectors.add(flat);
                    valLabels.add(entry.classId());
                }
                case CLEAN_TEST -> {
                    testVectors.add(flat);
                    testLabels.add(entry.classId());
                }
                default -> {}
            }
        }

        Matrix trX = toMatrix(trainVectors);
        int[] trY = toIntArray(trainLabels);
        Matrix vX = toMatrix(valVectors);
        int[] vY = toIntArray(valLabels);
        Matrix tX = toMatrix(testVectors);
        int[] tY = toIntArray(testLabels);

        return new LoadedDataset(trX, trY, vX, vY, tX, tY);
    }

    public TrainingRunResult train(
            MLPNetwork network,
            LoadedDataset dataset,
            int epochs,
            int batchSize,
            Optimizer optimizer,
            Path checkpointDir,
            long randomSeed
    ) throws IOException {
        Files.createDirectories(checkpointDir);
        Random rng = new Random(randomSeed);

        int nTrain = dataset.trainX().rows();
        int[] indices = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            indices[i] = i;
        }

        int[][] weightShapes = new int[network.getNumLayers()][2];
        int[] biasLengths = new int[network.getNumLayers()];
        int[] layerSizes = network.getLayerSizes();
        for (int l = 0; l < network.getNumLayers(); l++) {
            weightShapes[l][0] = layerSizes[l];
            weightShapes[l][1] = layerSizes[l + 1];
            biasLengths[l] = layerSizes[l + 1];
        }
        optimizer.initialize(network.getNumLayers(), weightShapes, biasLengths);

        List<TrainingMetrics> history = new ArrayList<>();
        double bestValMacroF1 = -1.0;
        double bestValLoss = Double.MAX_VALUE;
        double bestValAcc = 0.0;
        int bestEpoch = -1;
        Path bestCheckpointFile = checkpointDir.resolve("mlp_best.json");

        long startTotal = System.currentTimeMillis();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long epochStart = System.currentTimeMillis();

            for (int i = nTrain - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }

            double epochLossSum = 0.0;
            int batches = 0;

            for (int start = 0; start < nTrain; start += batchSize) {
                int end = Math.min(start + batchSize, nTrain);
                int curBatchSize = end - start;

                double[][] bx = new double[curBatchSize][dataset.trainX().cols()];
                int[] by = new int[curBatchSize];

                for (int b = 0; b < curBatchSize; b++) {
                    int idx = indices[start + b];
                    for (int c = 0; c < dataset.trainX().cols(); c++) {
                        bx[b][c] = dataset.trainX().get(idx, c);
                    }
                    by[b] = dataset.trainY()[idx];
                }

                Matrix batchMatrix = new Matrix(bx);
                double bLoss = network.trainBatch(batchMatrix, by, optimizer);
                epochLossSum += bLoss;
                batches++;
            }

            double avgTrainLoss = (batches > 0) ? (epochLossSum / batches) : 0.0;

            ConfusionMatrix trainCm = evaluate(network, dataset.trainX(), dataset.trainY(), 85);
            ConfusionMatrix.MetricsSummary trainSummary = trainCm.computeSummary();

            ConfusionMatrix valCm = evaluate(network, dataset.valX(), dataset.valY(), 85);
            ConfusionMatrix.MetricsSummary valSummary = valCm.computeSummary();
            double valLoss = network.evaluateLoss(dataset.valX(), dataset.valY());

            long epochTime = System.currentTimeMillis() - epochStart;

            TrainingMetrics tm = new TrainingMetrics(
                    epoch,
                    avgTrainLoss,
                    trainSummary.accuracy(),
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
                saveCheckpoint(network, epoch, valLoss, valSummary.accuracy(), valSummary.macroF1(), bestCheckpointFile);
            }
        }

        long totalTime = System.currentTimeMillis() - startTotal;

        return new TrainingRunResult(
                history,
                bestValLoss,
                bestValAcc,
                bestValMacroF1,
                bestEpoch,
                bestCheckpointFile,
                totalTime
        );
    }

    public ConfusionMatrix evaluate(MLPNetwork network, Matrix X, int[] y, int numClasses) {
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        if (X.rows() == 0) {
            return cm;
        }

        int[] preds = network.predict(X);
        cm.recordBatch(y, preds);
        return cm;
    }

    public void saveCheckpoint(
            MLPNetwork network,
            int epoch,
            double valLoss,
            double valAcc,
            double valMacroF1,
            Path targetFile
    ) throws IOException {
        List<double[][]> weightsList = new ArrayList<>();
        List<double[]> biasesList = new ArrayList<>();

        for (int l = 0; l < network.getNumLayers(); l++) {
            weightsList.add(network.getWeights(l).toArray());
            biasesList.add(network.getBiases(l));
        }

        MlpCheckpoint ckpt = new MlpCheckpoint(
                network.getLayerSizes(),
                network.getNumLayers(),
                weightsList,
                biasesList,
                epoch,
                valLoss,
                valAcc,
                valMacroF1
        );

        objectMapper.writeValue(targetFile.toFile(), ckpt);
    }

    public MLPNetwork loadCheckpoint(Path checkpointFile) throws IOException {
        MlpCheckpoint ckpt = objectMapper.readValue(checkpointFile.toFile(), MlpCheckpoint.class);
        MLPNetwork net = new MLPNetwork(ckpt.architecture(), 42L);

        for (int l = 0; l < ckpt.numLayers(); l++) {
            net.setWeights(l, new Matrix(ckpt.weights().get(l)));
            net.setBiases(l, ckpt.biases().get(l));
        }

        return net;
    }

    private static double[] flatten(float[][][] chw) {
        int channels = chw.length;
        int h = chw[0].length;
        int w = chw[0][0].length;
        double[] flat = new double[channels * h * w];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    flat[idx++] = chw[c][y][x];
                }
            }
        }
        return flat;
    }

    private static Matrix toMatrix(List<double[]> vectors) {
        if (vectors.isEmpty()) {
            return new Matrix(new double[0][0]);
        }
        double[][] data = new double[vectors.size()][];
        for (int i = 0; i < vectors.size(); i++) {
            data[i] = vectors.get(i);
        }
        return new Matrix(data);
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
