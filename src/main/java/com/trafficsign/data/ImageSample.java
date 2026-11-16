package com.trafficsign.data;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable metadata representing a single traffic-sign image sample.
 */
public record ImageSample(
        String sampleId,
        Path filePath,
        int classId,
        String trackId,
        int width,
        int height,
        int roiX1,
        int roiY1,
        int roiX2,
        int roiY2
) {
    public ImageSample {
        Objects.requireNonNull(sampleId, "sampleId must not be null");
        Objects.requireNonNull(filePath, "filePath must not be null");
        Objects.requireNonNull(trackId, "trackId must not be null");
        if (classId < 0) {
            throw new IllegalArgumentException("classId must be non-negative: " + classId);
        }
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException("Dimensions must be non-negative");
        }
    }

    public static ImageSample ofSimple(String sampleId, Path filePath, int classId, String trackId, int width, int height) {
        return new ImageSample(sampleId, filePath, classId, trackId, width, height, 0, 0, width, height);
    }

    public TrafficSignClass getSignClass() {
        return TrafficSignClass.fromId(classId).orElse(null);
    }

    public boolean hasCropRoi() {
        return (roiX2 > roiX1) && (roiY2 > roiY1);
    }

    public int roiWidth() {
        return Math.max(0, roiX2 - roiX1);
    }

    public int roiHeight() {
        return Math.max(0, roiY2 - roiY1);
    }
}
