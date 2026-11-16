package com.trafficsign.data;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public record ClassificationValidationResult(
        Path datasetRoot,
        int totalTrainImages,
        int totalTestImages,
        int trainClassesCount,
        int testClassesCount,
        List<String> classesOnlyInTrain,
        List<String> classesOnlyInTest,
        List<String> emptyClasses,
        Map<String, Integer> trainClassCounts,
        Map<String, Integer> testClassCounts,
        int trainMinSupport,
        int trainMaxSupport,
        double trainImbalanceRatio,
        int testMinSupport,
        int testMaxSupport,
        double testImbalanceRatio,
        List<Path> unexpectedExtensionFiles,
        List<Path> unreadableFiles,
        List<DatasetIntegrityChecker.DuplicateGroup> trainDuplicates,
        List<DatasetIntegrityChecker.DuplicateGroup> testDuplicates,
        int trainValSplitTrainCount,
        int trainValSplitValCount,
        long trainValSplitSeed,
        int crossSplitLeakageCount,
        List<String> recordedAnomalies
) {
    public ClassificationValidationResult {
        Objects.requireNonNull(datasetRoot, "datasetRoot must not be null");
        classesOnlyInTrain = classesOnlyInTrain == null ? List.of() : Collections.unmodifiableList(classesOnlyInTrain);
        classesOnlyInTest = classesOnlyInTest == null ? List.of() : Collections.unmodifiableList(classesOnlyInTest);
        emptyClasses = emptyClasses == null ? List.of() : Collections.unmodifiableList(emptyClasses);
        trainClassCounts = trainClassCounts == null ? Map.of() : Collections.unmodifiableMap(trainClassCounts);
        testClassCounts = testClassCounts == null ? Map.of() : Collections.unmodifiableMap(testClassCounts);
        unexpectedExtensionFiles = unexpectedExtensionFiles == null ? List.of() : Collections.unmodifiableList(unexpectedExtensionFiles);
        unreadableFiles = unreadableFiles == null ? List.of() : Collections.unmodifiableList(unreadableFiles);
        trainDuplicates = trainDuplicates == null ? List.of() : Collections.unmodifiableList(trainDuplicates);
        testDuplicates = testDuplicates == null ? List.of() : Collections.unmodifiableList(testDuplicates);
        recordedAnomalies = recordedAnomalies == null ? List.of() : Collections.unmodifiableList(recordedAnomalies);
    }

    public boolean isClassConsistent() {
        return classesOnlyInTrain.isEmpty() && classesOnlyInTest.isEmpty() && emptyClasses.isEmpty();
    }

    public boolean hasSevereImbalance(double threshold) {
        return trainImbalanceRatio > threshold || Double.isInfinite(trainImbalanceRatio);
    }
}
