package com.trafficsign.evaluation;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pure Java Confusion Matrix implementation for multi-class classification evaluation.
 * Computes Overall Accuracy, Per-Class Precision, Per-Class Recall, and Macro-Averaged F1 score.
 */
public class ConfusionMatrix {

    private final int numClasses;
    private final int[][] matrix;
    private int totalSamples;

    public record PerClassMetric(int classId, double precision, double recall, double f1, int support) {
    }

    public record MetricsSummary(
            double accuracy,
            double macroPrecision,
            double macroRecall,
            double macroF1,
            int totalSamples,
            PerClassMetric[] perClassMetrics
    ) {
    }

    public ConfusionMatrix(int numClasses) {
        if (numClasses <= 0) {
            throw new IllegalArgumentException("Number of classes must be positive: " + numClasses);
        }
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.totalSamples = 0;
    }

    public synchronized void record(int trueClass, int predictedClass) {
        checkClassBounds(trueClass);
        checkClassBounds(predictedClass);
        matrix[trueClass][predictedClass]++;
        totalSamples++;
    }

    public synchronized void recordBatch(int[] trueClasses, int[] predictedClasses) {
        Objects.requireNonNull(trueClasses, "trueClasses must not be null");
        Objects.requireNonNull(predictedClasses, "predictedClasses must not be null");
        if (trueClasses.length != predictedClasses.length) {
            throw new IllegalArgumentException("Batch length mismatch: " + trueClasses.length + " vs " + predictedClasses.length);
        }
        for (int i = 0; i < trueClasses.length; i++) {
            record(trueClasses[i], predictedClasses[i]);
        }
    }

    public MetricsSummary computeSummary() {
        if (totalSamples == 0) {
            return new MetricsSummary(0.0, 0.0, 0.0, 0.0, 0, new PerClassMetric[0]);
        }

        int correct = 0;
        PerClassMetric[] perClass = new PerClassMetric[numClasses];
        double sumPrecision = 0.0;
        double sumRecall = 0.0;
        double sumF1 = 0.0;

        for (int c = 0; c < numClasses; c++) {
            int tp = matrix[c][c];
            correct += tp;

            // Row sum = actual class instances (support = TP + FN)
            int actualInstances = 0;
            for (int j = 0; j < numClasses; j++) {
                actualInstances += matrix[c][j];
            }

            // Col sum = predicted instances for class c (TP + FP)
            int predictedInstances = 0;
            for (int i = 0; i < numClasses; i++) {
                predictedInstances += matrix[i][c];
            }

            double precision = (predictedInstances > 0) ? ((double) tp / predictedInstances) : 0.0;
            double recall = (actualInstances > 0) ? ((double) tp / actualInstances) : 0.0;
            double f1 = (precision + recall > 0.0) ? (2.0 * precision * recall / (precision + recall)) : 0.0;

            perClass[c] = new PerClassMetric(c, precision, recall, f1, actualInstances);

            sumPrecision += precision;
            sumRecall += recall;
            sumF1 += f1;
        }

        double accuracy = (double) correct / totalSamples;
        double macroPrecision = sumPrecision / numClasses;
        double macroRecall = sumRecall / numClasses;
        double macroF1 = sumF1 / numClasses;

        return new MetricsSummary(accuracy, macroPrecision, macroRecall, macroF1, totalSamples, perClass);
    }

    public int getCount(int trueClass, int predictedClass) {
        checkClassBounds(trueClass);
        checkClassBounds(predictedClass);
        return matrix[trueClass][predictedClass];
    }

    public int[][] getRawMatrix() {
        int[][] copy = new int[numClasses][numClasses];
        for (int i = 0; i < numClasses; i++) {
            System.arraycopy(matrix[i], 0, copy[i], 0, numClasses);
        }
        return copy;
    }

    public int getTotalSamples() {
        return totalSamples;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public double getAccuracy() {
        return computeSummary().accuracy();
    }

    public double getPrecision(int classId) {
        checkClassBounds(classId);
        MetricsSummary s = computeSummary();
        return (classId < s.perClassMetrics().length) ? s.perClassMetrics()[classId].precision() : 0.0;
    }

    public double getRecall(int classId) {
        checkClassBounds(classId);
        MetricsSummary s = computeSummary();
        return (classId < s.perClassMetrics().length) ? s.perClassMetrics()[classId].recall() : 0.0;
    }

    public double getF1(int classId) {
        checkClassBounds(classId);
        MetricsSummary s = computeSummary();
        return (classId < s.perClassMetrics().length) ? s.perClassMetrics()[classId].f1() : 0.0;
    }

    public double getMacroPrecision() {
        return computeSummary().macroPrecision();
    }

    public double getMacroRecall() {
        return computeSummary().macroRecall();
    }

    public double getMacroF1() {
        return computeSummary().macroF1();
    }

    public String formatSummary() {
        MetricsSummary s = computeSummary();
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Evaluation Metrics (Total Samples: %d):\n", s.totalSamples()));
        sb.append(String.format("  Overall Accuracy: %.4f (%.2f%%)\n", s.accuracy(), s.accuracy() * 100.0));
        sb.append(String.format("  Macro Precision : %.4f\n", s.macroPrecision()));
        sb.append(String.format("  Macro Recall    : %.4f\n", s.macroRecall()));
        sb.append(String.format("  Macro F1        : %.4f\n", s.macroF1()));
        return sb.toString();
    }

    private void checkClassBounds(int classId) {
        if (classId < 0 || classId >= numClasses) {
            throw new IllegalArgumentException("Class ID " + classId + " out of valid range [0, " + (numClasses - 1) + "]");
        }
    }
}
