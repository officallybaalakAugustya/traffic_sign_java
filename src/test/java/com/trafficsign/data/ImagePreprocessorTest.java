package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ImagePreprocessorTest {

    @Test
    @DisplayName("Should correctly crop image to Region of Interest (ROI)")
    void shouldCropToRoi() {
        ImagePreprocessor preprocessor = new ImagePreprocessor();
        BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);

        // Crop 20..60 x, 30..80 y
        BufferedImage cropped = preprocessor.cropRoi(img, 20, 30, 60, 80);
        assertThat(cropped.getWidth()).isEqualTo(40);
        assertThat(cropped.getHeight()).isEqualTo(50);
    }

    @Test
    @DisplayName("Should resize image to exact target dimensions via bilinear interpolation")
    void shouldResizeImage() {
        ImagePreprocessor preprocessor = new ImagePreprocessor(32, 32, false);
        BufferedImage source = new BufferedImage(80, 60, BufferedImage.TYPE_INT_RGB);

        BufferedImage resized = preprocessor.resizeBilinear(source, 32, 32);
        assertThat(resized.getWidth()).isEqualTo(32);
        assertThat(resized.getHeight()).isEqualTo(32);
    }

    @Test
    @DisplayName("Should convert image to CHW float tensor bounded in [0.0, 1.0]")
    void shouldConvertToNormalizedCHWTensor() {
        ImagePreprocessor preprocessor = new ImagePreprocessor(32, 32, false);
        BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);

        // Fill with a test color (R=128, G=255, B=0)
        int testRgb = (128 << 16) | (255 << 8) | 0;
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                img.setRGB(x, y, testRgb);
            }
        }

        float[][][] tensor = preprocessor.toTensorCHW(img);

        // Check tensor dimensions [Channels, Height, Width]
        assertThat(tensor.length).isEqualTo(3);
        assertThat(tensor[0].length).isEqualTo(32);
        assertThat(tensor[0][0].length).isEqualTo(32);

        // Check values normalized to [0.0, 1.0]
        assertThat(tensor[0][0][0]).isCloseTo(128.0f / 255.0f, org.assertj.core.data.Offset.offset(1e-4f));
        assertThat(tensor[1][0][0]).isCloseTo(1.0f, org.assertj.core.data.Offset.offset(1e-4f));
        assertThat(tensor[2][0][0]).isCloseTo(0.0f, org.assertj.core.data.Offset.offset(1e-4f));
    }

    @Test
    @DisplayName("Should convert image to HWC float tensor bounded in [0.0, 1.0]")
    void shouldConvertToNormalizedHWCTensor() {
        ImagePreprocessor preprocessor = new ImagePreprocessor(32, 32, false);
        BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, (64 << 16) | (128 << 8) | 192);

        float[][][] tensor = preprocessor.toTensorHWC(img);
        assertThat(tensor.length).isEqualTo(32);
        assertThat(tensor[0].length).isEqualTo(32);
        assertThat(tensor[0][0].length).isEqualTo(3);
        assertThat(tensor[0][0][0]).isCloseTo(64.0f / 255.0f, org.assertj.core.data.Offset.offset(1e-4f));
        assertThat(tensor[0][0][1]).isCloseTo(128.0f / 255.0f, org.assertj.core.data.Offset.offset(1e-4f));
        assertThat(tensor[0][0][2]).isCloseTo(192.0f / 255.0f, org.assertj.core.data.Offset.offset(1e-4f));
    }

    @Test
    @DisplayName("Should validate target dimensions and throw on non-positive values")
    void shouldThrowOnInvalidDimensions() {
        assertThatThrownBy(() -> new ImagePreprocessor(0, 32, false))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new ImagePreprocessor(32, -5, false))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
