package com.trafficsign.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

/**
 * Dataset loader supporting standard ImageFolder directory structures
 * (commonly used in Hugging Face, Kaggle, and torchvision formats).
 * Maps subdirectories to traffic sign taxonomy classes, skips corrupt files,
 * records unmapped classes, and tracks sample frequencies.
 */
public class ImageFolderDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(ImageFolderDatasetLoader.class);

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "png", "jpg", "jpeg", "ppm", "bmp", "webp"
    );

    private final TrafficSignTaxonomy taxonomy;

    public record LoadResult(
            List<ImageSample> samples,
            ClassFrequencyReport frequencyReport,
            List<Path> corruptOrUnreadableFiles,
            int totalFilesScanned
    ) {
    }

    public ImageFolderDatasetLoader(TrafficSignTaxonomy taxonomy) {
        this.taxonomy = Objects.requireNonNull(taxonomy, "taxonomy must not be null");
    }

    public LoadResult load(Path datasetDir, boolean verifyImageHeaders) throws IOException {
        Objects.requireNonNull(datasetDir, "datasetDir must not be null");
        if (!Files.exists(datasetDir) || !Files.isDirectory(datasetDir)) {
            throw new IllegalArgumentException("Dataset directory does not exist or is not a directory: " + datasetDir);
        }

        List<ImageSample> samples = new ArrayList<>();
        List<Path> corruptFiles = new ArrayList<>();
        Map<String, Integer> unmappedLabels = new TreeMap<>();
        int totalScanned = 0;

        try (Stream<Path> classDirs = Files.list(datasetDir)) {
            List<Path> sortedDirs = classDirs.filter(Files::isDirectory).sorted().toList();

            for (Path classDir : sortedDirs) {
                String folderName = classDir.getFileName().toString();
                Optional<Integer> classIdOpt = taxonomy.findClassIdByLabel(folderName);

                if (classIdOpt.isEmpty()) {
                    unmappedLabels.put(folderName, unmappedLabels.getOrDefault(folderName, 0) + 1);
                    logger.warn("Encountered unmapped class directory in taxonomy: '{}'", folderName);
                }

                int classId = classIdOpt.orElse(-1);

                try (Stream<Path> files = Files.list(classDir)) {
                    List<Path> imageFiles = files.filter(Files::isRegularFile)
                            .filter(this::isSupportedExtension)
                            .sorted()
                            .toList();

                    for (Path imgPath : imageFiles) {
                        totalScanned++;
                        long size = Files.size(imgPath);
                        if (size == 0) {
                            corruptFiles.add(imgPath);
                            continue;
                        }

                        int width = 0;
                        int height = 0;

                        if (verifyImageHeaders) {
                            try {
                                BufferedImage bimg = ImageIO.read(imgPath.toFile());
                                if (bimg == null) {
                                    corruptFiles.add(imgPath);
                                    continue;
                                }
                                width = bimg.getWidth();
                                height = bimg.getHeight();
                            } catch (Exception ex) {
                                corruptFiles.add(imgPath);
                                continue;
                            }
                        }

                        // Generate deterministic sample and group IDs
                        String sampleId = folderName + "_" + imgPath.getFileName().toString();
                        String groupId = folderName; // Group by class folder or capture batch

                        // Only ingest mapped valid classes into the classification pipeline
                        if (classId >= 0) {
                            samples.add(new ImageSample(sampleId, imgPath, classId, groupId, width, height, 0, 0, width, height));
                        }
                    }
                }
            }
        }

        ClassFrequencyReport report = ClassFrequencyReport.generate(samples, taxonomy, unmappedLabels);

        return new LoadResult(
                Collections.unmodifiableList(samples),
                report,
                Collections.unmodifiableList(corruptFiles),
                totalScanned
        );
    }

    private boolean isSupportedExtension(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) {
            return false;
        }
        String ext = name.substring(dot + 1);
        return SUPPORTED_EXTENSIONS.contains(ext);
    }
}
