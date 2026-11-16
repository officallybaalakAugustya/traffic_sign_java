package com.trafficsign.data;

import java.util.*;

/**
 * Audit validator ensuring zero data leakage between dataset splits.
 * Specifically checks for track-level overlap, duplicate sample paths,
 * and unbalanced class coverage.
 */
public class LeakageDetector {

    public record LeakageReport(
            boolean isClean,
            List<String> errors,
            List<String> warnings,
            Map<SplitType, Set<String>> trackSets,
            Map<SplitType, Map<Integer, Integer>> classDistribution
    ) {
        public boolean hasErrors() {
            return !errors.isEmpty();
        }

        public boolean hasWarnings() {
            return !warnings.isEmpty();
        }
    }

    public LeakageReport inspect(SplitManifest manifest) {
        Objects.requireNonNull(manifest, "manifest must not be null");
        return inspect(manifest.records());
    }

    public LeakageReport inspect(List<SplitRecord> records) {
        Objects.requireNonNull(records, "records must not be null");

        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        Map<SplitType, Set<String>> trackSets = new EnumMap<>(SplitType.class);
        Map<SplitType, Map<Integer, Integer>> classDistribution = new EnumMap<>(SplitType.class);
        for (SplitType st : SplitType.values()) {
            trackSets.put(st, new HashSet<>());
            classDistribution.put(st, new TreeMap<>());
        }

        Set<String> allSampleIds = new HashSet<>();
        Set<String> allPaths = new HashSet<>();

        for (SplitRecord r : records) {
            // Uniqueness checks
            if (!allSampleIds.add(r.sampleId())) {
                errors.add("Duplicate sampleId found: '" + r.sampleId() + "'");
            }
            if (!allPaths.add(r.relativePath())) {
                errors.add("Duplicate relativePath found: '" + r.relativePath() + "'");
            }

            // Track collection
            trackSets.get(r.splitType()).add(r.trackId());

            // Class distribution collection
            Map<Integer, Integer> dist = classDistribution.get(r.splitType());
            dist.put(r.classId(), dist.getOrDefault(r.classId(), 0) + 1);
        }

        // 1. Check for track overlap between any pairwise splits
        Set<String> trainTracks = trackSets.get(SplitType.TRAIN);
        Set<String> valTracks = trackSets.get(SplitType.VALIDATION);
        Set<String> testTracks = trackSets.get(SplitType.TEST);

        Set<String> trainValOverlap = new HashSet<>(trainTracks);
        trainValOverlap.retainAll(valTracks);
        if (!trainValOverlap.isEmpty()) {
            errors.add("Data Leakage Error: " + trainValOverlap.size() + " track(s) shared between TRAIN and VALIDATION: " + trainValOverlap);
        }

        Set<String> trainTestOverlap = new HashSet<>(trainTracks);
        trainTestOverlap.retainAll(testTracks);
        if (!trainTestOverlap.isEmpty()) {
            errors.add("Data Leakage Error: " + trainTestOverlap.size() + " track(s) shared between TRAIN and TEST: " + trainTestOverlap);
        }

        Set<String> valTestOverlap = new HashSet<>(valTracks);
        valTestOverlap.retainAll(testTracks);
        if (!valTestOverlap.isEmpty()) {
            errors.add("Data Leakage Error: " + valTestOverlap.size() + " track(s) shared between VALIDATION and TEST: " + valTestOverlap);
        }

        // 2. Check for missing classes in training set
        Map<Integer, Integer> trainClasses = classDistribution.get(SplitType.TRAIN);
        for (int c = 0; c < TrafficSignClass.totalClasses(); c++) {
            if (!trainClasses.containsKey(c) || trainClasses.get(c) == 0) {
                warnings.add("Class " + c + " (" + TrafficSignClass.requireById(c).getLabelName() + ") has no training samples in this split.");
            }
        }

        boolean clean = errors.isEmpty();
        return new LeakageReport(clean, Collections.unmodifiableList(errors), Collections.unmodifiableList(warnings), trackSets, classDistribution);
    }
}
