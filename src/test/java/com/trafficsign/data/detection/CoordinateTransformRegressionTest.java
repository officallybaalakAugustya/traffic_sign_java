package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class CoordinateTransformRegressionTest {

    @Test
    @DisplayName("Transforms known Indian road scene bounding box to detector coordinates and back within 1e-4 tolerance")
    void testGroundTruthBoxRoundTrip16x16() {
        int nativeWidth = 1920;
        int nativeHeight = 994;
        int gridCols = 16;
        int gridRows = 16;

        BoundingBox nativeBox = BoundingBox.ofPixels(1445.0, 312.0, 1550.0, 420.0, 0, "traffic sign");
        BoundingBox normBox = nativeBox.toNormalized(nativeWidth, nativeHeight);

        float[] target = DetectionUtils.boxToTarget(normBox, gridCols, gridRows);
        float cellX = target[0];
        float cellY = target[1];
        float bw = target[2];
        float bh = target[3];
        int row = (int) target[4];
        int col = (int) target[5];

        BoundingBox recoveredNorm = DetectionUtils.targetToBox(cellX, cellY, bw, bh, row, col, gridCols, gridRows);
        BoundingBox recoveredNative = recoveredNorm.toAbsolute(nativeWidth, nativeHeight);

        assertThat(recoveredNative.x1()).isCloseTo(nativeBox.x1(), offset(0.01));
        assertThat(recoveredNative.y1()).isCloseTo(nativeBox.y1(), offset(0.01));
        assertThat(recoveredNative.x2()).isCloseTo(nativeBox.x2(), offset(0.01));
        assertThat(recoveredNative.y2()).isCloseTo(nativeBox.y2(), offset(0.01));

        double iou = nativeBox.computeIoU(recoveredNative);
        assertThat(iou).isGreaterThan(0.9999);
    }

    @ParameterizedTest
    @CsvSource({
            "8, 8, 100.0, 150.0, 220.0, 270.0",
            "16, 16, 500.0, 200.0, 560.0, 260.0",
            "16, 16, 1850.0, 900.0, 1910.0, 980.0",
            "16, 16, 10.0, 10.0, 45.0, 45.0"
    })
    @DisplayName("Parametric verification of box encoding and decoding across grid resolutions and image regions")
    void testParametricCoordinateRoundTrip(int gridCols, int gridRows, double x1, double y1, double x2, double y2) {
        int imgW = 1920;
        int imgH = 994;

        BoundingBox original = BoundingBox.ofPixels(x1, y1, x2, y2, 0, "traffic sign");
        BoundingBox norm = original.toNormalized(imgW, imgH);

        float[] target = DetectionUtils.boxToTarget(norm, gridCols, gridRows);
        int r = (int) target[4];
        int c = (int) target[5];

        BoundingBox recovered = DetectionUtils.targetToBox(target[0], target[1], target[2], target[3], r, c, gridCols, gridRows)
                .toAbsolute(imgW, imgH);

        assertThat(original.computeIoU(recovered)).isGreaterThan(0.999);
    }

    @Test
    @DisplayName("Calculates exact physical dimensions in 128x128 detector space compared to 48x48 classifier space")
    void testDetectorVsClassifierInputResolutionDimensions() {
        double nativeWidth = 104.851557;
        double nativeHeight = 107.025656;

        double detectorW = nativeWidth * (128.0 / 1920.0);
        double detectorH = nativeHeight * (128.0 / 994.0);

        double classifierW = nativeWidth * (48.0 / 1920.0);
        double classifierH = nativeHeight * (48.0 / 994.0);

        assertThat(detectorW).isCloseTo(6.990, offset(0.01));
        assertThat(detectorH).isCloseTo(13.782, offset(0.01));

        assertThat(classifierW).isCloseTo(2.621, offset(0.01));
        assertThat(classifierH).isCloseTo(5.168, offset(0.01));

        double detectorArea = detectorW * detectorH;
        double cellArea16x16 = 8.0 * 8.0;
        double cellCoveragePercent = (detectorArea / cellArea16x16) * 100.0;

        assertThat(cellCoveragePercent).isGreaterThan(140.0);
    }
}
