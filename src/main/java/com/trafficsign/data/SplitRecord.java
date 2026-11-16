package com.trafficsign.data;

import java.util.Objects;

/**
 * Immutable record representing the assigned partition for a single sample.
 */
public record SplitRecord(
        String sampleId,
        String trackId,
        int classId,
        SplitType splitType,
        String relativePath
) {
    public SplitRecord {
        Objects.requireNonNull(sampleId, "sampleId must not be null");
        Objects.requireNonNull(trackId, "trackId must not be null");
        Objects.requireNonNull(splitType, "splitType must not be null");
        Objects.requireNonNull(relativePath, "relativePath must not be null");
        if (classId < 0 || classId >= TrafficSignClass.totalClasses()) {
            throw new IllegalArgumentException("classId " + classId + " out of valid range [0, 42]");
        }
    }
}
