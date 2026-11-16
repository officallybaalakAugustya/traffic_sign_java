package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BoundingBoxTest {

    @Test
    @DisplayName("Should correctly calculate dimensions and area of bounding boxes")
    void shouldCalculateDimensionsAndArea() {
        BoundingBox box = BoundingBox.ofPixels(10.0, 20.0, 50.0, 80.0, 5, "SPEED_LIMIT_40");
        assertThat(box.width()).isEqualTo(40.0);
        assertThat(box.height()).isEqualTo(60.0);
        assertThat(box.area()).isEqualTo(2400.0);
        assertThat(!box.isNormalized()).isTrue();
    }

    @Test
    @DisplayName("Should throw IllegalArgumentException on inverted coordinates")
    void shouldRejectInvertedCoordinates() {
        assertThatThrownBy(() -> BoundingBox.ofPixels(50.0, 20.0, 10.0, 80.0, 1, "TEST"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Inverted");

        assertThatThrownBy(() -> BoundingBox.ofPixels(10.0, 80.0, 50.0, 20.0, 1, "TEST"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Inverted");
    }

    @Test
    @DisplayName("Should throw IllegalArgumentException when normalized coordinates exceed 1.0")
    void shouldRejectOutOfBoundsNormalizedCoordinates() {
        assertThatThrownBy(() -> BoundingBox.ofNormalized(0.1, 0.2, 1.2, 0.8, 1, "TEST"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Normalized coordinates must not exceed 1.0");
    }

    @Test
    @DisplayName("Should correctly convert between pixel and normalized coordinates")
    void shouldConvertCoordinates() {
        BoundingBox pixelBox = BoundingBox.ofPixels(100.0, 200.0, 300.0, 600.0, 2, "CATTLE");
        BoundingBox normBox = pixelBox.toNormalized(1000, 1000);

        assertThat(normBox.isNormalized()).isTrue();
        assertThat(normBox.x1()).isEqualTo(0.1);
        assertThat(normBox.y1()).isEqualTo(0.2);
        assertThat(normBox.x2()).isEqualTo(0.3);
        assertThat(normBox.y2()).isEqualTo(0.6);

        BoundingBox reconstructed = normBox.toAbsolute(1000, 1000);
        assertThat(reconstructed.x1()).isEqualTo(100.0);
        assertThat(reconstructed.y1()).isEqualTo(200.0);
        assertThat(reconstructed.x2()).isEqualTo(300.0);
        assertThat(reconstructed.y2()).isEqualTo(600.0);
    }

    @Test
    @DisplayName("Should calculate exact Intersection over Union (IoU) for overlapping and disjoint boxes")
    void shouldComputeIoU() {
        BoundingBox b1 = BoundingBox.ofPixels(0.0, 0.0, 10.0, 10.0, 1, "BOX1");
        BoundingBox b2 = BoundingBox.ofPixels(0.0, 0.0, 10.0, 10.0, 1, "BOX2");

        // Identical boxes -> IoU = 1.0
        assertThat(b1.computeIoU(b2)).isEqualTo(1.0);

        // Disjoint boxes -> IoU = 0.0
        BoundingBox b3 = BoundingBox.ofPixels(20.0, 20.0, 30.0, 30.0, 1, "BOX3");
        assertThat(b1.computeIoU(b3)).isEqualTo(0.0);

        // 50% overlap: b1 is [0,0,10,10] (area 100). b4 is [5,0,15,10] (area 100).
        // Intersection is [5,0,10,10] (area 50). Union is 100 + 100 - 50 = 150.
        // IoU = 50 / 150 = 1/3 ~ 0.333333
        BoundingBox b4 = BoundingBox.ofPixels(5.0, 0.0, 15.0, 10.0, 1, "BOX4");
        assertThat(b1.computeIoU(b4)).isCloseTo(1.0 / 3.0, org.assertj.core.data.Offset.offset(1e-5));
    }
}
