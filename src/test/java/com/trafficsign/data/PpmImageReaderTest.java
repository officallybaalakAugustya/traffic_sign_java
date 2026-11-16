package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PpmImageReaderTest {

    @Test
    @DisplayName("Should successfully read a binary P6 PPM image with comments")
    void shouldReadP6BinaryPpm() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        // Header
        baos.write("P6\n# Test synthetic PPM image\n2 2\n255\n".getBytes(StandardCharsets.US_ASCII));
        // 2x2 pixels (RGB)
        // Pixel (0,0): Red (255, 0, 0)
        baos.write(new byte[]{(byte) 255, (byte) 0, (byte) 0});
        // Pixel (1,0): Green (0, 255, 0)
        baos.write(new byte[]{(byte) 0, (byte) 255, (byte) 0});
        // Pixel (0,1): Blue (0, 0, 255)
        baos.write(new byte[]{(byte) 0, (byte) 0, (byte) 255});
        // Pixel (1,1): White (255, 255, 255)
        baos.write(new byte[]{(byte) 255, (byte) 255, (byte) 255});

        BufferedImage img = PpmImageReader.read(baos.toByteArray());

        assertThat(img.getWidth()).isEqualTo(2);
        assertThat(img.getHeight()).isEqualTo(2);
        assertThat(img.getRGB(0, 0) & 0x00FFFFFF).isEqualTo(0x00FF0000); // Red
        assertThat(img.getRGB(1, 0) & 0x00FFFFFF).isEqualTo(0x0000FF00); // Green
        assertThat(img.getRGB(0, 1) & 0x00FFFFFF).isEqualTo(0x000000FF); // Blue
        assertThat(img.getRGB(1, 1) & 0x00FFFFFF).isEqualTo(0x00FFFFFF); // White
    }

    @Test
    @DisplayName("Should successfully read an ASCII P3 PPM image")
    void shouldReadP3AsciiPpm() throws IOException {
        String ppmAscii = """
                P3
                # ASCII 2x1 image
                2 1
                255
                255 0 0   0 255 0
                """;
        BufferedImage img = PpmImageReader.read(ppmAscii.getBytes(StandardCharsets.US_ASCII));

        assertThat(img.getWidth()).isEqualTo(2);
        assertThat(img.getHeight()).isEqualTo(1);
        assertThat(img.getRGB(0, 0) & 0x00FFFFFF).isEqualTo(0x00FF0000); // Red
        assertThat(img.getRGB(1, 0) & 0x00FFFFFF).isEqualTo(0x0000FF00); // Green
    }

    @Test
    @DisplayName("Should throw IOException when encountering unsupported magic format")
    void shouldThrowOnInvalidFormat() {
        byte[] invalidHeader = "P5\n2 2\n255\n".getBytes(StandardCharsets.US_ASCII);
        assertThatThrownBy(() -> PpmImageReader.read(invalidHeader))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("Unsupported PPM format");
    }
}
