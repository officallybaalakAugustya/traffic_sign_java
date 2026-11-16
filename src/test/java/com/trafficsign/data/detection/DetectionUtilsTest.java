package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class DetectionUtilsTest {

    @Test
    @DisplayName("DetectionUtils coordinate encoding and inverse decoding is consistent")
    void testEncodeDecodeConsistency() {
        BoundingBox original = BoundingBox.ofNormalized(0.20, 0.30, 0.40, 0.50, 0, "traffic sign");

        int gridCols = 8;
        int gridRows = 8;

        float[] target = DetectionUtils.boxToTarget(original, gridCols, gridRows);
        int row = (int) target[4];
        int col = (int) target[5];
        float cellX = target[0];
        float cellY = target[1];
        float bw = target[2];
        float bh = target[3];

        BoundingBox decoded = DetectionUtils.targetToBox(cellX, cellY, bw, bh, row, col, gridCols, gridRows);

        assertThat(decoded.x1()).isCloseTo(original.x1(), org.assertj.core.data.Offset.offset(1e-4));
        assertThat(decoded.y1()).isCloseTo(original.y1(), org.assertj.core.data.Offset.offset(1e-4));
        assertThat(decoded.x2()).isCloseTo(original.x2(), org.assertj.core.data.Offset.offset(1e-4));
        assertThat(decoded.y2()).isCloseTo(original.y2(), org.assertj.core.data.Offset.offset(1e-4));
        assertThat(decoded.computeIoU(original)).isGreaterThan(0.99);
    }

    @Test
    @DisplayName("NonMaximumSuppression filters duplicate overlapping boxes")
    void testNmsFiltering() {
        BoundingBox b1 = BoundingBox.ofNormalized(0.1, 0.1, 0.3, 0.3, 0, "traffic sign");
        BoundingBox b2 = BoundingBox.ofNormalized(0.11, 0.11, 0.31, 0.31, 0, "traffic sign");
        BoundingBox b3 = BoundingBox.ofNormalized(0.7, 0.7, 0.9, 0.9, 0, "traffic sign");

        DetectionUtils.DetectionPrediction p1 = new DetectionUtils.DetectionPrediction(b1, 0.95f, 1, 1);
        DetectionUtils.DetectionPrediction p2 = new DetectionUtils.DetectionPrediction(b2, 0.85f, 1, 1);
        DetectionUtils.DetectionPrediction p3 = new DetectionUtils.DetectionPrediction(b3, 0.90f, 6, 6);

        List<DetectionUtils.DetectionPrediction> nms = DetectionUtils.nonMaximumSuppression(List.of(p1, p2, p3), 0.5);

        assertThat(nms).hasSize(2);
        assertThat(nms).extracting(DetectionUtils.DetectionPrediction::confidence).containsExactly(0.95f, 0.90f);
    }
}
