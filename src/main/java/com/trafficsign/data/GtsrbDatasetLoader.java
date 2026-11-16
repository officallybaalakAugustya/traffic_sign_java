package com.trafficsign.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Robust loader and validator for the German Traffic Sign Recognition Benchmark (GTSRB).
 * Reads official GTSRB directory structures, parses semicolons-separated CSV annotations,
 * validates bounding box ROI coordinates, and maps files to ImageSample records.
 */
public class GtsrbDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(GtsrbDatasetLoader.class);

    public record DatasetValidationSummary(
            int totalSamples,
            int totalTracks,
            int classCount,
            Map<Integer, Integer> perClassCounts,
            List<String> validationErrors,
            boolean isFullyValid
    ) {
    }

    public static List<ImageSample> loadTrainingDataset(Path gtsrbImagesRoot) throws IOException {
        Objects.requireNonNull(gtsrbImagesRoot, "gtsrbImagesRoot must not be null");
        if (!Files.exists(gtsrbImagesRoot)) {
            throw new IllegalArgumentException("GTSRB root directory does not exist: " + gtsrbImagesRoot.toAbsolutePath());
        }

        List<ImageSample> samples = new ArrayList<>();

        // Loop over the 43 standard GTSRB class folders: 00000 to 00042
        for (int c = 0; c < TrafficSignClass.totalClasses(); c++) {
            String folderName = String.format("%05d", c);
            Path classDir = gtsrbImagesRoot.resolve(folderName);

            if (!Files.isDirectory(classDir)) {
                // Some archives place Images directly inside or inside Final_Training/Images
                Path altClassDir = gtsrbImagesRoot.resolve("Final_Training").resolve("Images").resolve(folderName);
                if (Files.isDirectory(altClassDir)) {
                    classDir = altClassDir;
                } else {
                    continue;
                }
            }

            // Look for GT-*.csv annotation file
            String csvName = String.format("GT-%05d.csv", c);
            Path csvPath = classDir.resolve(csvName);

            if (Files.exists(csvPath)) {
                samples.addAll(parseClassCsv(csvPath, classDir, c));
            } else {
                // Fallback: search directory for any .ppm files
                final int currentClass = c;
                try (var stream = Files.list(classDir)) {
                    stream.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".ppm"))
                            .forEach(ppm -> {
                                String fname = ppm.getFileName().toString();
                                String trackId = extractTrackId(fname);
                                samples.add(ImageSample.ofSimple(fname, ppm, currentClass, trackId, 0, 0));
                            });
                }
            }
        }

        return samples;
    }

    public static List<ImageSample> parseClassCsv(Path csvPath, Path parentDir, int expectedClassId) throws IOException {
        List<ImageSample> samples = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            String header = reader.readLine();
            if (header == null) {
                return samples;
            }

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                // Format: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                String[] tokens = line.split(";");
                if (tokens.length < 8) {
                    logger.warn("Malformed annotation row in {}: {}", csvPath, line);
                    continue;
                }

                String filename = tokens[0].trim();
                int width = Integer.parseInt(tokens[1].trim());
                int height = Integer.parseInt(tokens[2].trim());
                int roiX1 = Integer.parseInt(tokens[3].trim());
                int roiY1 = Integer.parseInt(tokens[4].trim());
                int roiX2 = Integer.parseInt(tokens[5].trim());
                int roiY2 = Integer.parseInt(tokens[6].trim());
                int classId = Integer.parseInt(tokens[7].trim());

                if (classId != expectedClassId) {
                    logger.warn("ClassId mismatch in {}: row says {} but folder is {}", csvPath, classId, expectedClassId);
                }

                Path imagePath = parentDir.resolve(filename);
                String trackId = extractTrackId(filename);

                samples.add(new ImageSample(
                        filename,
                        imagePath,
                        classId,
                        trackId,
                        width,
                        height,
                        roiX1,
                        roiY1,
                        roiX2,
                        roiY2
                ));
            }
        }
        return samples;
    }

    public static String extractTrackId(String filename) {
        int underscoreIdx = filename.indexOf('_');
        if (underscoreIdx > 0) {
            return filename.substring(0, underscoreIdx);
        }
        int dotIdx = filename.lastIndexOf('.');
        return (dotIdx > 0) ? filename.substring(0, dotIdx) : filename;
    }

    public static DatasetValidationSummary validateDataset(List<ImageSample> samples) {
        Objects.requireNonNull(samples, "samples must not be null");

        Map<Integer, Integer> perClass = new TreeMap<>();
        Set<String> tracks = new HashSet<>();
        List<String> errors = new ArrayList<>();

        for (ImageSample s : samples) {
            perClass.put(s.classId(), perClass.getOrDefault(s.classId(), 0) + 1);
            tracks.add(s.trackId());

            if (s.roiX1() > s.roiX2() || s.roiY1() > s.roiY2()) {
                errors.add("Inverted ROI bounding box for sample: " + s.sampleId());
            }
            if (s.classId() < 0 || s.classId() >= TrafficSignClass.totalClasses()) {
                errors.add("Invalid class ID: " + s.classId());
            }
        }

        boolean valid = errors.isEmpty() && !samples.isEmpty();
        return new DatasetValidationSummary(
                samples.size(),
                tracks.size(),
                perClass.size(),
                perClass,
                errors,
                valid
        );
    }
}
