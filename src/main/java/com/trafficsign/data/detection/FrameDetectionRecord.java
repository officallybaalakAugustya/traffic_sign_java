package com.trafficsign.data.detection;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public record FrameDetectionRecord(
        String frameStem,
        String imageName,
        String jsonName,
        int imageWidth,
        int imageHeight,
        List<DetectionAnnotation> annotations,
        boolean hasImage,
        boolean hasJson
) {
    public FrameDetectionRecord {
        Objects.requireNonNull(frameStem, "frameStem must not be null");
        annotations = annotations == null ? List.of() : Collections.unmodifiableList(annotations);
    }

    public List<DetectionAnnotation> getTrafficSignAnnotations() {
        return annotations.stream()
                .filter(DetectionAnnotation::isGenericTrafficSign)
                .toList();
    }

    public int totalObjects() {
        return annotations.size();
    }

    public int trafficSignCount() {
        return (int) annotations.stream()
                .filter(DetectionAnnotation::isGenericTrafficSign)
                .count();
    }
}
