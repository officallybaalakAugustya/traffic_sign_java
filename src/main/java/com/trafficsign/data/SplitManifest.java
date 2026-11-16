package com.trafficsign.data;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Serializable manifest detailing the exact partition assignments, random seeds,
 * and class distributions for a dataset version.
 */
public record SplitManifest(
        @JsonProperty("datasetName") String datasetName,
        @JsonProperty("createdAt") String createdAt,
        @JsonProperty("randomSeed") long randomSeed,
        @JsonProperty("trainRatio") double trainRatio,
        @JsonProperty("valRatio") double valRatio,
        @JsonProperty("testRatio") double testRatio,
        @JsonProperty("totalSamples") int totalSamples,
        @JsonProperty("totalTracks") int totalTracks,
        @JsonProperty("samplesPerSplit") Map<SplitType, Integer> samplesPerSplit,
        @JsonProperty("records") List<SplitRecord> records
) {
    @JsonCreator
    public SplitManifest(
            @JsonProperty("datasetName") String datasetName,
            @JsonProperty("createdAt") String createdAt,
            @JsonProperty("randomSeed") long randomSeed,
            @JsonProperty("trainRatio") double trainRatio,
            @JsonProperty("valRatio") double valRatio,
            @JsonProperty("testRatio") double testRatio,
            @JsonProperty("totalSamples") int totalSamples,
            @JsonProperty("totalTracks") int totalTracks,
            @JsonProperty("samplesPerSplit") Map<SplitType, Integer> samplesPerSplit,
            @JsonProperty("records") List<SplitRecord> records
    ) {
        this.datasetName = Objects.requireNonNull(datasetName, "datasetName must not be null");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt must not be null");
        this.randomSeed = randomSeed;
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.testRatio = testRatio;
        this.totalSamples = totalSamples;
        this.totalTracks = totalTracks;
        this.samplesPerSplit = (samplesPerSplit != null) ? Collections.unmodifiableMap(samplesPerSplit) : Map.of();
        this.records = (records != null) ? Collections.unmodifiableList(records) : List.of();
    }
}
