package com.trafficsign.data;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Pure Java reader for Netpbm PPM (Portable Pixmap) images.
 * Supports binary P6 and ASCII P3 formats commonly used in GTSRB raw datasets.
 */
public final class PpmImageReader {

    private PpmImageReader() {
    }

    public static BufferedImage read(Path path) throws IOException {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            return read(in);
        }
    }

    public static BufferedImage read(byte[] bytes) throws IOException {
        try (InputStream in = new ByteArrayInputStream(bytes)) {
            return read(in);
        }
    }

    public static BufferedImage read(InputStream in) throws IOException {
        BufferedInputStream bis = (in instanceof BufferedInputStream)
                ? (BufferedInputStream) in
                : new BufferedInputStream(in);

        String magic = readToken(bis);
        if (!"P6".equals(magic) && !"P3".equals(magic)) {
            throw new IOException("Unsupported PPM format: '" + magic + "'. Expected 'P6' or 'P3'.");
        }

        int width = Integer.parseInt(readToken(bis));
        int height = Integer.parseInt(readToken(bis));
        int maxVal = Integer.parseInt(readToken(bis));

        if (width <= 0 || height <= 0) {
            throw new IOException("Invalid PPM dimensions: " + width + "x" + height);
        }
        if (maxVal <= 0 || maxVal > 65535) {
            throw new IOException("Invalid PPM max color value: " + maxVal);
        }

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        if ("P6".equals(magic)) {
            readP6Binary(bis, image, width, height, maxVal);
        } else {
            readP3Ascii(bis, image, width, height, maxVal);
        }

        return image;
    }

    private static void readP6Binary(BufferedInputStream in, BufferedImage image, int width, int height, int maxVal)
            throws IOException {
        boolean isTwoByte = maxVal > 255;
        byte[] buffer = new byte[width * 3 * (isTwoByte ? 2 : 1)];
        int[] rgbRow = new int[width];

        for (int y = 0; y < height; y++) {
            int bytesRead = 0;
            while (bytesRead < buffer.length) {
                int read = in.read(buffer, bytesRead, buffer.length - bytesRead);
                if (read == -1) {
                    throw new IOException("Unexpected end of PPM binary data at row " + y);
                }
                bytesRead += read;
            }

            int bufIdx = 0;
            for (int x = 0; x < width; x++) {
                int r;
                int g;
                int b;
                if (!isTwoByte) {
                    r = (buffer[bufIdx++] & 0xFF) * 255 / maxVal;
                    g = (buffer[bufIdx++] & 0xFF) * 255 / maxVal;
                    b = (buffer[bufIdx++] & 0xFF) * 255 / maxVal;
                } else {
                    int rRaw = ((buffer[bufIdx++] & 0xFF) << 8) | (buffer[bufIdx++] & 0xFF);
                    int gRaw = ((buffer[bufIdx++] & 0xFF) << 8) | (buffer[bufIdx++] & 0xFF);
                    int bRaw = ((buffer[bufIdx++] & 0xFF) << 8) | (buffer[bufIdx++] & 0xFF);
                    r = rRaw * 255 / maxVal;
                    g = gRaw * 255 / maxVal;
                    b = bRaw * 255 / maxVal;
                }
                rgbRow[x] = (r << 16) | (g << 8) | b;
            }
            image.setRGB(0, y, width, 1, rgbRow, 0, width);
        }
    }

    private static void readP3Ascii(BufferedInputStream in, BufferedImage image, int width, int height, int maxVal)
            throws IOException {
        int[] rgbRow = new int[width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = Integer.parseInt(readToken(in)) * 255 / maxVal;
                int g = Integer.parseInt(readToken(in)) * 255 / maxVal;
                int b = Integer.parseInt(readToken(in)) * 255 / maxVal;
                rgbRow[x] = (r << 16) | (g << 8) | b;
            }
            image.setRGB(0, y, width, 1, rgbRow, 0, width);
        }
    }

    private static String readToken(BufferedInputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;

        while ((c = in.read()) != -1) {
            if (c == '#') {
                while ((c = in.read()) != -1 && c != '\n' && c != '\r') {
                }
            } else if (!Character.isWhitespace(c)) {
                break;
            }
        }

        if (c == -1) {
            throw new IOException("Unexpected end of stream while reading token");
        }

        sb.append((char) c);
        while ((c = in.read()) != -1) {
            if (Character.isWhitespace(c) || c == '#') {
                if (c == '#') {
                    while ((c = in.read()) != -1 && c != '\n' && c != '\r') {
                    }
                }
                break;
            }
            sb.append((char) c);
        }

        return sb.toString();
    }
}
