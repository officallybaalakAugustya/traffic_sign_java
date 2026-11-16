package com.trafficsign.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;

/**
 * Generates reproducible, stratified, track-aware dataset partitions.
 * Strictly assigns all images sharing the same physical track ID to the same split,
 * preventing sequence and background leakage.
 */
public class DatasetSplitter {

    private final long randomSeed;
    private final double trainRatio;
    private final double valRatio;
    private final double testRatio;
    private final ObjectMapper objectMapper;

    public DatasetSplitter(long randomSeed, double trainRatio, double valRatio, double testRatio) {
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
        this.objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public static DatasetSplitter defaultStandard() {
        return new DatasetSplitter(42L, 0.70, 0.15, 0.15);
    }

    public SplitManifest split(String datasetName, List<ImageSample> samples) {
        Objects.requireNonNull(datasetName, "datasetName must not be null");
        Objects.requireNonNull(samples, "samples must not be null");
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("Cannot split an empty sample list");
        }

        // 1. Group samples by trackId
        Map<String, List<ImageSample>> trackToSamples = new LinkedHashMap<>();
        for (ImageSample sample : samples) {
            trackToSamples.computeIfAbsent(sample.trackId(), k -> new ArrayList<>()).add(sample);
        }

        // 2. Determine majority/dominant class for each track (tracks are sign-specific)
        Map<Integer, List<String>> classToTracks = new TreeMap<>();
        for (Map.Entry<String, List<ImageSample>> entry : trackToSamples.entrySet()) {
            String trackId = entry.getKey();
            int classId = entry.getValue().getFirst().classId();
            classToTracks.computeIfAbsent(classId, k -> new ArrayList<>()).add(trackId);
        }

        // 3. Stratified track-level assignment using deterministic pseudo-random shuffling
        Random rng = new Random(randomSeed);
        Map<String, SplitType> trackToSplit = new HashMap<>();

        for (Map.Entry<Integer, List<String>> entry : classToTracks.entrySet()) {
            List<String> tracks = new ArrayList<>(entry.getValue());
            Collections.shuffle(tracks, rng);

            int n = tracks.size();
            int trainCount = (int) Math.round(n * trainRatio);
            int valCount = (int) Math.round(n * valRatio);

            // Handle edge cases for small track counts
            if (n >= 3) {
                if (trainCount == 0) trainCount = 1;
                if (valCount == 0) valCount = 1;
                if (trainCount + valCount >= n) {
                    trainCount = Math.max(1, n - 2);
                    valCount = 1;
                }
            } else if (n == 2) {
                trainCount = 1;
                valCount = 0;
            } else {
                trainCount = 1;
                valCount = 0;
            }

            for (int i = 0; i < n; i++) {
                SplitType assigned;
                if (i < trainCount) {
                    assigned = SplitType.TRAIN;
                } else if (i < trainCount + valCount) {
                    assigned = SplitType.VALIDATION;
                } else {
                    assigned = SplitType.TEST;
                }
                trackToSplit.put(tracks.get(i), assigned);
            }
        }

        // 4. Build SplitRecords for every sample based on its track assignment
        List<SplitRecord> records = new ArrayList<>(samples.size());
        Map<SplitType, Integer> samplesPerSplit = new EnumMap<>(SplitType.class);
        for (SplitType type : SplitType.values()) {
            samplesPerSplit.put(type, 0);
        }

        for (ImageSample sample : samples) {
            SplitType splitType = trackToSplit.getOrDefault(sample.trackId(), SplitType.TRAIN);
            records.add(new SplitRecord(
                    sample.sampleId(),
                    sample.trackId(),
                    sample.classId(),
                    splitType,
                    sample.filePath().toString().replace('\\', '/')
            ));
            samplesPerSplit.put(splitType, samplesPerSplit.get(splitType) + 1);
        }

        return new SplitManifest(
                datasetName,
                Instant.now().toString(),
                randomSeed,
                trainRatio,
                valRatio,
                testRatio,
                samples.size(),
                trackToSamples.size(),
                samplesPerSplit,
                records
        );
    }

    public void saveManifest(SplitManifest manifest, Path destination) throws IOException {
        objectMapper.writeValue(destination.toFile(), manifest);
    }

    public SplitManifest loadManifest(Path source) throws IOException {
        return objectMapper.readValue(source.toFile(), SplitManifest.class);
    }

    public long getRandomSeed() {
        return randomSeed;
    }
}
