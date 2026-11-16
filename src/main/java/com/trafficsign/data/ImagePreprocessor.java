package com.trafficsign.data;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Image preprocessing engine for traffic-sign recognition.
 * Handles format decoding (PPM, PNG, JPEG), ROI cropping, bilinear resizing,
 * min-max scaling to [0.0, 1.0], and channel standardization.
 */
public class ImagePreprocessor {

    public static final int DEFAULT_TARGET_WIDTH = 32;
    public static final int DEFAULT_TARGET_HEIGHT = 32;

    // Standard GTSRB dataset channel statistics
    public static final float[] GTSRB_CHANNEL_MEANS = new float[]{0.340f, 0.312f, 0.321f};
    public static final float[] GTSRB_CHANNEL_STDS = new float[]{0.272f, 0.260f, 0.266f};

    private final int targetWidth;
    private final int targetHeight;
    private final boolean standardize;
    private final float[] channelMeans;
    private final float[] channelStds;

    public ImagePreprocessor() {
        this(DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT, false, GTSRB_CHANNEL_MEANS, GTSRB_CHANNEL_STDS);
    }

    public ImagePreprocessor(int targetWidth, int targetHeight, boolean standardize) {
        this(targetWidth, targetHeight, standardize, GTSRB_CHANNEL_MEANS, GTSRB_CHANNEL_STDS);
    }

    public ImagePreprocessor(int targetWidth, int targetHeight, boolean standardize, float[] means, float[] stds) {
        if (targetWidth <= 0 || targetHeight <= 0) {
            throw new IllegalArgumentException("Target dimensions must be strictly positive");
        }
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.standardize = standardize;
        this.channelMeans = Objects.requireNonNull(means, "means must not be null").clone();
        this.channelStds = Objects.requireNonNull(stds, "stds must not be null").clone();
    }

    public BufferedImage loadImage(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        String filename = path.getFileName().toString().toLowerCase();
        if (filename.endsWith(".ppm")) {
            return PpmImageReader.read(path);
        }
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IOException("Unsupported or unreadable image file: " + path);
        }
        return image;
    }

    public BufferedImage cropRoi(BufferedImage source, int roiX1, int roiY1, int roiX2, int roiY2) {
        Objects.requireNonNull(source, "source image must not be null");
        int width = source.getWidth();
        int height = source.getHeight();

        int x1 = Math.max(0, Math.min(roiX1, width - 1));
        int y1 = Math.max(0, Math.min(roiY1, height - 1));
        int x2 = Math.max(x1 + 1, Math.min(roiX2, width));
        int y2 = Math.max(y1 + 1, Math.min(roiY2, height));

        int cropW = x2 - x1;
        int cropH = y2 - y1;

        if (cropW <= 0 || cropH <= 0) {
            return source;
        }

        return source.getSubimage(x1, y1, cropW, cropH);
    }

    public BufferedImage resizeBilinear(BufferedImage source, int width, int height) {
        Objects.requireNonNull(source, "source image must not be null");
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Resize dimensions must be positive");
        }

        if (source.getWidth() == width && source.getHeight() == height) {
            return source;
        }

        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        try {
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.drawImage(source, 0, 0, width, height, null);
        } finally {
            g2d.dispose();
        }
        return resized;
    }

    /**
     * Converts a preprocessed image into a 3D float tensor with shape [Channels, Height, Width] (CHW).
     */
    public float[][][] toTensorCHW(BufferedImage image) {
        BufferedImage resized = resizeBilinear(image, targetWidth, targetHeight);
        float[][][] tensor = new float[3][targetHeight][targetWidth];

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resized.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;

                if (standardize) {
                    r = (r - channelMeans[0]) / channelStds[0];
                    g = (g - channelMeans[1]) / channelStds[1];
                    b = (b - channelMeans[2]) / channelStds[2];
                }

                tensor[0][y][x] = r;
                tensor[1][y][x] = g;
                tensor[2][y][x] = b;
            }
        }
        return tensor;
    }

    /**
     * Converts a preprocessed image into a 3D float tensor with shape [Height, Width, Channels] (HWC).
     */
    public float[][][] toTensorHWC(BufferedImage image) {
        BufferedImage resized = resizeBilinear(image, targetWidth, targetHeight);
        float[][][] tensor = new float[targetHeight][targetWidth][3];

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resized.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;

                if (standardize) {
                    r = (r - channelMeans[0]) / channelStds[0];
                    g = (g - channelMeans[1]) / channelStds[1];
                    b = (b - channelMeans[2]) / channelStds[2];
                }

                tensor[y][x][0] = r;
                tensor[y][x][1] = g;
                tensor[y][x][2] = b;
            }
        }
        return tensor;
    }

    /**
     * Convenience method to process an image from path directly into a CHW tensor.
     */
    public float[][][] process(Path path) throws IOException {
        BufferedImage img = loadImage(path);
        return toTensorCHW(img);
    }

    public int getTargetWidth() {
        return targetWidth;
    }

    public int getTargetHeight() {
        return targetHeight;
    }

    public boolean isStandardize() {
        return standardize;
    }
}
