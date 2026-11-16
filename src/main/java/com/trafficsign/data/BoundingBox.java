package com.trafficsign.data;

import java.util.Objects;

/**
 * Immutable geometric representation of a 2D bounding box for sign detection and localization.
 * Supports both normalized coordinates [0.0, 1.0] and absolute pixel coordinates,
 * with Intersection over Union (IoU) calculation and boundary validation.
 */
public record BoundingBox(
        double x1,
        double y1,
        double x2,
        double y2,
        boolean isNormalized,
        int classId,
        String label
) {
    public BoundingBox {
        Objects.requireNonNull(label, "label must not be null");
        if (x1 < 0.0 || y1 < 0.0) {
            throw new IllegalArgumentException("Coordinates must be non-negative: (" + x1 + ", " + y1 + ")");
        }
        if (x2 < x1 || y2 < y1) {
            throw new IllegalArgumentException("Inverted bounding box: (" + x1 + "," + y1 + ") to (" + x2 + "," + y2 + ")");
        }
        if (isNormalized) {
            if (x2 > 1.0001 || y2 > 1.0001) {
                throw new IllegalArgumentException("Normalized coordinates must not exceed 1.0: (" + x2 + ", " + y2 + ")");
            }
        }
    }

    public static BoundingBox ofPixels(double x1, double y1, double x2, double y2, int classId, String label) {
        return new BoundingBox(x1, y1, x2, y2, false, classId, label);
    }

    public static BoundingBox ofNormalized(double x1, double y1, double x2, double y2, int classId, String label) {
        return new BoundingBox(x1, y1, x2, y2, true, classId, label);
    }

    public double width() {
        return x2 - x1;
    }

    public double height() {
        return y2 - y1;
    }

    public double area() {
        return width() * height();
    }

    public BoundingBox toAbsolute(int imgWidth, int imgHeight) {
        if (!isNormalized) {
            return this;
        }
        return new BoundingBox(x1 * imgWidth, y1 * imgHeight, x2 * imgWidth, y2 * imgHeight, false, classId, label);
    }

    public BoundingBox toNormalized(int imgWidth, int imgHeight) {
        if (isNormalized) {
            return this;
        }
        if (imgWidth <= 0 || imgHeight <= 0) {
            throw new IllegalArgumentException("Image dimensions must be positive");
        }
        return new BoundingBox(x1 / imgWidth, y1 / imgHeight, x2 / imgWidth, y2 / imgHeight, true, classId, label);
    }

    public double computeIoU(BoundingBox other) {
        Objects.requireNonNull(other, "other bounding box must not be null");
        if (this.isNormalized != other.isNormalized) {
            throw new IllegalArgumentException("Cannot compute IoU between normalized and unnormalized bounding boxes");
        }

        double interX1 = Math.max(this.x1, other.x1);
        double interY1 = Math.max(this.y1, other.y1);
        double interX2 = Math.min(this.x2, other.x2);
        double interY2 = Math.min(this.y2, other.y2);

        double interWidth = Math.max(0.0, interX2 - interX1);
        double interHeight = Math.max(0.0, interY2 - interY1);
        double intersectionArea = interWidth * interHeight;

        double unionArea = this.area() + other.area() - intersectionArea;
        if (unionArea <= 1e-9) {
            return 0.0;
        }
        return intersectionArea / unionArea;
    }
}
