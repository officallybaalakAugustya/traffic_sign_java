package com.trafficsign.data.augmentation;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.util.Objects;
import java.util.Random;

public class ImageAugmentor {

    public record AugmentationConfig(
            boolean enabled,
            double maxRotationDegrees,
            double maxTranslationFraction,
            double minScale,
            double maxScale,
            float brightnessFactorRange,
            float contrastFactorRange,
            long seed
    ) {
        public static AugmentationConfig moderateDefault(long seed) {
            return new AugmentationConfig(
                    true,
                    10.0,
                    0.05,
                    0.90,
                    1.10,
                    0.15f,
                    0.15f,
                    seed
            );
        }

        public static AugmentationConfig photometricOnly(long seed) {
            return new AugmentationConfig(
                    true,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.10f,
                    0.10f,
                    seed
            );
        }

        public static AugmentationConfig disabled() {
            return new AugmentationConfig(
                    false,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0f,
                    0.0f,
                    0L
            );
        }
    }

    private final AugmentationConfig config;
    private final Random rng;

    public ImageAugmentor(AugmentationConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.rng = new Random(config.seed());
    }

    public BufferedImage augment(BufferedImage input) {
        Objects.requireNonNull(input, "input image must not be null");
        if (!config.enabled()) {
            return input;
        }

        int width = input.getWidth();
        int height = input.getHeight();

        BufferedImage transformed = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = transformed.createGraphics();

        try {
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            double angleDeg = (rng.nextDouble() * 2.0 - 1.0) * config.maxRotationDegrees();
            double angleRad = Math.toRadians(angleDeg);

            double maxTx = config.maxTranslationFraction() * width;
            double maxTy = config.maxTranslationFraction() * height;
            double tx = (rng.nextDouble() * 2.0 - 1.0) * maxTx;
            double ty = (rng.nextDouble() * 2.0 - 1.0) * maxTy;

            double scale = config.minScale() + rng.nextDouble() * (config.maxScale() - config.minScale());

            AffineTransform at = new AffineTransform();
            at.translate(width / 2.0 + tx, height / 2.0 + ty);
            at.rotate(angleRad);
            at.scale(scale, scale);
            at.translate(-width / 2.0, -height / 2.0);

            g2d.drawImage(input, at, null);
        } finally {
            g2d.dispose();
        }

        float bJitter = 1.0f + (rng.nextFloat() * 2.0f - 1.0f) * config.brightnessFactorRange();
        float cOffset = (rng.nextFloat() * 2.0f - 1.0f) * (config.contrastFactorRange() * 255.0f);

        RescaleOp rescale = new RescaleOp(bJitter, cOffset, null);
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        rescale.filter(transformed, output);

        return output;
    }

    public AugmentationConfig getConfig() {
        return config;
    }
}
