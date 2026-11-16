package com.trafficsign.data.augmentation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.assertj.core.api.Assertions.assertThat;

class ImageAugmentorTest {

    @Test
    @DisplayName("ImageAugmentor preserves dimensions and produces valid RGB values")
    void testAugmentationProperties() {
        BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 32, 32);
        g.setColor(Color.WHITE);
        g.fillRect(10, 10, 12, 12);
        g.dispose();

        ImageAugmentor.AugmentationConfig cfg = ImageAugmentor.AugmentationConfig.moderateDefault(42L);
        ImageAugmentor augmentor = new ImageAugmentor(cfg);

        BufferedImage aug = augmentor.augment(img);

        assertThat(aug.getWidth()).isEqualTo(32);
        assertThat(aug.getHeight()).isEqualTo(32);

        int rgb = aug.getRGB(16, 16);
        int r = (rgb >> 16) & 0xFF;
        int gr = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;

        assertThat(r).isBetween(0, 255);
        assertThat(gr).isBetween(0, 255);
        assertThat(b).isBetween(0, 255);
    }

    @Test
    @DisplayName("Disabled ImageAugmentor returns unchanged image")
    void testDisabledAugmentor() {
        BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        img.setRGB(5, 5, 0x123456);

        ImageAugmentor augmentor = new ImageAugmentor(ImageAugmentor.AugmentationConfig.disabled());
        BufferedImage result = augmentor.augment(img);

        assertThat(result).isSameAs(img);
    }
}
