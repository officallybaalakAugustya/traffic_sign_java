package com.trafficsign.data;

import java.util.*;

/**
 * Universal grouped and stratified dataset partitioner.
 * Guarantees that all samples sharing a group identifier (e.g., video track,
 * capture session, geographic cluster, or route ID) reside exclusively in one split,
 * preventing sequence and background leakage.
 */
public class GroupedDatasetSplitter {

    private final long randomSeed;
    private final double trainRatio;
    private final double valRatio;
    private final double testRatio;

    public record PartitionResult(
            List<ImageSample> trainSamples,
            List<ImageSample> valSamples,
            List<ImageSample> testSamples
    ) {
        public int totalSamples() {
            return trainSamples.size() + valSamples.size() + testSamples.size();
        }
    }

    public GroupedDatasetSplitter(long randomSeed, double trainRatio, double valRatio, double testRatio) {
        if (trainRatio <= 0 || valRatio <= 0 || testRatio <= 0) {
            throw new IllegalArgumentException("Split ratios must be strictly positive");
        }
        double sum = trainRatio + valRatio + testRatio;
        if (Math.abs(sum - 1.0) > 1e-4) {
            throw new IllegalArgumentException("Split ratios must sum to 1.0. Current sum: " + sum);
        }

        this.randomSeed = randomSeed;
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.testRatio = testRatio;
    }

    public static GroupedDatasetSplitter defaultStandard() {
        return new GroupedDatasetSplitter(42L, 0.70, 0.15, 0.15);
    }

    public PartitionResult partition(List<ImageSample> samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("Cannot partition an empty sample list");
        }

        // 1. Group samples by group/track ID
        Map<String, List<ImageSample>> groupToSamples = new LinkedHashMap<>();
        for (ImageSample s : samples) {
            groupToSamples.computeIfAbsent(s.trackId(), k -> new ArrayList<>()).add(s);
        }

        // 2. Map class -> group IDs (stratification key)
        Map<Integer, List<String>> classToGroups = new TreeMap<>();
        for (var entry : groupToSamples.entrySet()) {
            String groupId = entry.getKey();
            int classId = entry.getValue().getFirst().classId();
            classToGroups.computeIfAbsent(classId, k -> new ArrayList<>()).add(groupId);
        }

        // 3. Shuffle groups deterministically per class
        Random rng = new Random(randomSeed);
        Map<String, SplitType> groupToSplit = new HashMap<>();

        for (var entry : classToGroups.entrySet()) {
            List<String> groups = new ArrayList<>(entry.getValue());
            Collections.shuffle(groups, rng);

            int n = groups.size();
            int trainCount = (int) Math.round(n * trainRatio);
            int valCount = (int) Math.round(n * valRatio);

            if (n == 1) {
                // If only 1 group exists for a rare class, assign to train
                trainCount = 1;
                valCount = 0;
            } else if (n == 2) {
                trainCount = 1;
                valCount = 1;
            }

            for (int i = 0; i < n; i++) {
                String gid = groups.get(i);
                if (i < trainCount) {
                    groupToSplit.put(gid, SplitType.TRAIN);
                } else if (i < trainCount + valCount) {
                    groupToSplit.put(gid, SplitType.VALIDATION);
                } else {
                    groupToSplit.put(gid, SplitType.TEST);
                }
            }
        }

        List<ImageSample> train = new ArrayList<>();
        List<ImageSample> val = new ArrayList<>();
        List<ImageSample> test = new ArrayList<>();

        for (ImageSample s : samples) {
            SplitType split = groupToSplit.getOrDefault(s.trackId(), SplitType.TRAIN);
            switch (split) {
                case TRAIN -> train.add(s);
                case VALIDATION -> val.add(s);
                case TEST -> test.add(s);
            }
        }

        return new PartitionResult(
                Collections.unmodifiableList(train),
                Collections.unmodifiableList(val),
                Collections.unmodifiableList(test)
        );
    }
}
