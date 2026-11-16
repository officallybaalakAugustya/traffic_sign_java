package com.trafficsign.data.detection;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public record TarInspectionSummary(
        int totalArchivesProcessed,
        List<String> archiveNames,
        int totalFrames,
        int totalMatchedPairs,
        int totalJpgOnly,
        int totalJsonOnly,
        int totalPngIgnored,
        int totalObjects,
        int totalTrafficSigns,
        Map<String, Integer> categoryDistribution,
        List<String> annotationErrors,
        List<FrameDetectionRecord> sampleFrames
) {
    public TarInspectionSummary {
        Objects.requireNonNull(archiveNames, "archiveNames must not be null");
        Objects.requireNonNull(categoryDistribution, "categoryDistribution must not be null");
        Objects.requireNonNull(annotationErrors, "annotationErrors must not be null");
        sampleFrames = sampleFrames == null ? List.of() : Collections.unmodifiableList(sampleFrames);
    }
}
