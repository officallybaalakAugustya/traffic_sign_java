package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;

import java.util.Objects;

public record DetectionAnnotation(
        int id,
        String category,
        BoundingBox box,
        Double confidence,
        boolean occluded,
        boolean truncated,
        Long trackId
) {
    public DetectionAnnotation {
        Objects.requireNonNull(category, "category must not be null");
        Objects.requireNonNull(box, "box must not be null");
    }

    public boolean isGenericTrafficSign() {
        return "traffic sign".equalsIgnoreCase(category);
    }
}
