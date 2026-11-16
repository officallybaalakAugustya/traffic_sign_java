package com.trafficsign.evaluation;

import java.util.Arrays;
import java.util.Objects;

public class MajorityClassBaseline {

    public record BaselineResult(
            int majorityClassId,
            int trainMajorityCount,
            int totalTrainSamples,
            double accuracy,
            double macroPrecision,
            double macroRecall,
            double macroF1,
            int totalEvaluatedSamples
    ) {
    }

    public static int findMajorityClass(int[] trainLabels) {
        Objects.requireNonNull(trainLabels, "trainLabels must not be null");
        if (trainLabels.length == 0) {
            throw new IllegalArgumentException("Cannot determine majority class of empty labels");
        }

        int maxLabel = 0;
        for (int l : trainLabels) {
            if (l > maxLabel) maxLabel = l;
        }

        int[] counts = new int[maxLabel + 1];
        for (int l : trainLabels) {
            counts[l]++;
        }

        int majorityClass = 0;
        int maxCount = counts[0];
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                majorityClass = i;
            }
        }

        return majorityClass;
    }

    public static BaselineResult evaluate(int[] trainLabels, int[] evalLabels, int numClasses) {
        Objects.requireNonNull(trainLabels, "trainLabels must not be null");
        Objects.requireNonNull(evalLabels, "evalLabels must not be null");

        int majorityClass = findMajorityClass(trainLabels);
        int trainMajorityCount = 0;
        for (int l : trainLabels) {
            if (l == majorityClass) trainMajorityCount++;
        }

        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        for (int trueLabel : evalLabels) {
            cm.record(trueLabel, majorityClass);
        }

        ConfusionMatrix.MetricsSummary summary = cm.computeSummary();

        return new BaselineResult(
                majorityClass,
                trainMajorityCount,
                trainLabels.length,
                summary.accuracy(),
                summary.macroPrecision(),
                summary.macroRecall(),
                summary.macroF1(),
                evalLabels.length
        );
    }
}
