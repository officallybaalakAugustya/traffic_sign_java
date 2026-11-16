package com.trafficsign.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

public class ClassificationDatasetValidator {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "png", "jpg", "jpeg", "ppm", "bmp", "webp"
    );

    private final TrafficSignTaxonomy taxonomy;
    private final ObjectMapper objectMapper;

    public ClassificationDatasetValidator(TrafficSignTaxonomy taxonomy) {
        this.taxonomy = Objects.requireNonNull(taxonomy, "taxonomy must not be null");
        this.objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public ClassificationValidationResult validate(
            Path datasetRoot,
            long randomSeed,
            double valRatio,
            boolean verifyImageHeaders
    ) throws IOException {
        Objects.requireNonNull(datasetRoot, "datasetRoot must not be null");
        if (!Files.exists(datasetRoot) || !Files.isDirectory(datasetRoot)) {
            throw new IllegalArgumentException("Dataset root does not exist: " + datasetRoot);
        }

        Path trainDir = datasetRoot.resolve("train");
        Path testDir = datasetRoot.resolve("test");

        if (!Files.exists(trainDir) || !Files.isDirectory(trainDir)) {
            throw new IllegalArgumentException("Train directory does not exist: " + trainDir);
        }
        if (!Files.exists(testDir) || !Files.isDirectory(testDir)) {
            throw new IllegalArgumentException("Test directory does not exist: " + testDir);
        }

        List<String> anomalies = new ArrayList<>();
        List<Path> unexpectedExtensionFiles = new ArrayList<>();
        List<Path> unreadableFiles = new ArrayList<>();

        ScanResult trainScan = scanPartition(trainDir, verifyImageHeaders, unexpectedExtensionFiles, unreadableFiles, anomalies, "train");
        ScanResult testScan = scanPartition(testDir, verifyImageHeaders, unexpectedExtensionFiles, unreadableFiles, anomalies, "test");

        Set<String> trainClasses = trainScan.classCounts().keySet();
        Set<String> testClasses = testScan.classCounts().keySet();

        List<String> classesOnlyInTrain = trainClasses.stream().filter(c -> !testClasses.contains(c)).sorted().toList();
        List<String> classesOnlyInTest = testClasses.stream().filter(c -> !trainClasses.contains(c)).sorted().toList();
        List<String> emptyClasses = new ArrayList<>();
        trainScan.emptyClasses().forEach(emptyClasses::add);
        testScan.emptyClasses().forEach(emptyClasses::add);

        if (!classesOnlyInTrain.isEmpty()) {
            anomalies.add("Classes present in train but missing in test: " + classesOnlyInTrain);
        }
        if (!classesOnlyInTest.isEmpty()) {
            anomalies.add("Classes present in test but missing in train: " + classesOnlyInTest);
        }
        if (!emptyClasses.isEmpty()) {
            anomalies.add("Empty class directories detected: " + emptyClasses);
        }

        DatasetIntegrityChecker.IntegrityAuditReport trainAudit = DatasetIntegrityChecker.auditDirectory(trainScan.imagePaths());
        DatasetIntegrityChecker.IntegrityAuditReport testAudit = DatasetIntegrityChecker.auditDirectory(testScan.imagePaths());

        if (!trainAudit.duplicateGroups().isEmpty()) {
            anomalies.add("Duplicate content hashes detected within train: " + trainAudit.duplicateGroups().size() + " duplicate sets");
        }
        if (!testAudit.duplicateGroups().isEmpty()) {
            anomalies.add("Duplicate content hashes detected within test: " + testAudit.duplicateGroups().size() + " duplicate sets");
        }

        Map<Integer, List<ImageSample>> classToTrainSamples = new TreeMap<>();
        for (ImageSample s : trainScan.samples()) {
            classToTrainSamples.computeIfAbsent(s.classId(), k -> new ArrayList<>()).add(s);
        }

        Random rng = new Random(randomSeed);
        List<ImageSample> trainPart = new ArrayList<>();
        List<ImageSample> valPart = new ArrayList<>();

        for (List<ImageSample> classSamples : classToTrainSamples.values()) {
            List<ImageSample> shuffled = new ArrayList<>(classSamples);
            Collections.shuffle(shuffled, rng);
            int n = shuffled.size();
            int valCount = (int) Math.round(n * valRatio);
            if (n > 1 && valCount == 0) {
                valCount = 1;
            } else if (n > 1 && valCount >= n) {
                valCount = n - 1;
            }
            int trainCount = n - valCount;
            for (int i = 0; i < trainCount; i++) {
                trainPart.add(shuffled.get(i));
            }
            for (int i = trainCount; i < n; i++) {
                valPart.add(shuffled.get(i));
            }
        }

        int crossSplitLeakage = DatasetIntegrityChecker.checkCrossSplitHashLeakage(trainPart, valPart);
        if (crossSplitLeakage > 0) {
            anomalies.add("Cross-split hash leakage detected between generated train and val: " + crossSplitLeakage + " images");
        }

        int trainTestLeakage = DatasetIntegrityChecker.checkCrossSplitHashLeakage(trainScan.samples(), testScan.samples());
        if (trainTestLeakage > 0) {
            anomalies.add("Content hash leakage detected between raw train and raw test sets: " + trainTestLeakage + " images");
        }

        return new ClassificationValidationResult(
                datasetRoot,
                trainScan.samples().size(),
                testScan.samples().size(),
                trainClasses.size(),
                testClasses.size(),
                classesOnlyInTrain,
                classesOnlyInTest,
                emptyClasses,
                trainScan.classCounts(),
                testScan.classCounts(),
                trainScan.minSupport(),
                trainScan.maxSupport(),
                trainScan.imbalanceRatio(),
                testScan.minSupport(),
                testScan.maxSupport(),
                testScan.imbalanceRatio(),
                Collections.unmodifiableList(unexpectedExtensionFiles),
                Collections.unmodifiableList(unreadableFiles),
                trainAudit.duplicateGroups(),
                testAudit.duplicateGroups(),
                trainPart.size(),
                valPart.size(),
                randomSeed,
                crossSplitLeakage + trainTestLeakage,
                Collections.unmodifiableList(anomalies)
        );
    }

    public String generateMarkdownReport(ClassificationValidationResult r) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Classification Dataset Validation Report\n\n");
        sb.append("**Dataset Root:** `").append(r.datasetRoot()).append("`\n\n");
        sb.append("## Summary Statistics\n\n");
        sb.append("| Metric | Train Partition | Test Partition |\n");
        sb.append("| :--- | :---: | :---: |\n");
        sb.append("| **Total Images** | ").append(r.totalTrainImages()).append(" | ").append(r.totalTestImages()).append(" |\n");
        sb.append("| **Distinct Classes** | ").append(r.trainClassesCount()).append(" | ").append(r.testClassesCount()).append(" |\n");
        sb.append("| **Minimum Support** | ").append(r.trainMinSupport()).append(" | ").append(r.testMinSupport()).append(" |\n");
        sb.append("| **Maximum Support** | ").append(r.trainMaxSupport()).append(" | ").append(r.testMaxSupport()).append(" |\n");
        sb.append("| **Imbalance Ratio** | ").append(String.format("%.2f", r.trainImbalanceRatio())).append(" | ").append(String.format("%.2f", r.testImbalanceRatio())).append(" |\n\n");

        sb.append("## Class Consistency\n\n");
        sb.append("- **Class Consistency Across Splits:** ").append(r.isClassConsistent() ? "PASSED (Exact class match)" : "FAILED").append("\n");
        if (!r.classesOnlyInTrain().isEmpty()) {
            sb.append("- **Classes only in Train:** ").append(r.classesOnlyInTrain()).append("\n");
        }
        if (!r.classesOnlyInTest().isEmpty()) {
            sb.append("- **Classes only in Test:** ").append(r.classesOnlyInTest()).append("\n");
        }
        if (!r.emptyClasses().isEmpty()) {
            sb.append("- **Empty Class Folders:** ").append(r.emptyClasses()).append("\n");
        }

        sb.append("\n## Data Hygiene & Integrity\n\n");
        sb.append("- **Unreadable / Corrupt Images:** ").append(r.unreadableFiles().size()).append("\n");
        sb.append("- **Unexpected File Extensions:** ").append(r.unexpectedExtensionFiles().size()).append("\n");
        sb.append("- **Duplicate Hash Sets in Train:** ").append(r.trainDuplicates().size()).append("\n");
        sb.append("- **Duplicate Hash Sets in Test:** ").append(r.testDuplicates().size()).append("\n");
        sb.append("- **Cross-Split Hash Overlap:** ").append(r.crossSplitLeakageCount()).append("\n\n");

        sb.append("## Deterministic Train/Validation Partitioning\n\n");
        sb.append("- **Random Seed:** `").append(r.trainValSplitSeed()).append("`\n");
        sb.append("- **Derived Train Count:** ").append(r.trainValSplitTrainCount()).append("\n");
        sb.append("- **Derived Validation Count:** ").append(r.trainValSplitValCount()).append("\n");
        sb.append("- **Preserved Held-Out Test Count:** ").append(r.totalTestImages()).append(" (Strictly untouched)\n\n");

        if (!r.recordedAnomalies().isEmpty()) {
            sb.append("## Recorded Anomalies & Warnings\n\n");
            for (String anomaly : r.recordedAnomalies()) {
                sb.append("- ").append(anomaly).append("\n");
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    public String generateJsonReport(ClassificationValidationResult r) throws IOException {
        return objectMapper.writeValueAsString(r);
    }

    private record ScanResult(
            List<ImageSample> samples,
            List<Path> imagePaths,
            Map<String, Integer> classCounts,
            List<String> emptyClasses,
            int minSupport,
            int maxSupport,
            double imbalanceRatio
    ) {
    }

    private ScanResult scanPartition(
            Path partitionDir,
            boolean verifyImageHeaders,
            List<Path> unexpectedExtensionFiles,
            List<Path> unreadableFiles,
            List<String> anomalies,
            String partitionName
    ) throws IOException {
        List<ImageSample> samples = new ArrayList<>();
        List<Path> imagePaths = new ArrayList<>();
        Map<String, Integer> classCounts = new TreeMap<>();
        List<String> emptyClasses = new ArrayList<>();

        try (Stream<Path> dirs = Files.list(partitionDir)) {
            List<Path> classDirs = dirs.filter(Files::isDirectory).sorted().toList();

            for (Path cdir : classDirs) {
                String className = cdir.getFileName().toString();
                int count = 0;

                try (Stream<Path> files = Files.list(cdir)) {
                    List<Path> fileList = files.filter(Files::isRegularFile).sorted().toList();

                    for (Path f : fileList) {
                        String name = f.getFileName().toString().toLowerCase();
                        int dot = name.lastIndexOf('.');
                        String ext = dot > 0 ? name.substring(dot + 1) : "";

                        if (!SUPPORTED_EXTENSIONS.contains(ext)) {
                            unexpectedExtensionFiles.add(f);
                            anomalies.add("Unexpected file extension in " + partitionName + "/" + className + ": " + f.getFileName());
                            continue;
                        }

                        if (Files.size(f) == 0) {
                            unreadableFiles.add(f);
                            anomalies.add("Zero-byte file in " + partitionName + "/" + className + ": " + f.getFileName());
                            continue;
                        }

                        int width = 0;
                        int height = 0;
                        if (verifyImageHeaders) {
                            try {
                                BufferedImage bimg = ImageIO.read(f.toFile());
                                if (bimg == null) {
                                    unreadableFiles.add(f);
                                    anomalies.add("ImageIO decode failure in " + partitionName + "/" + className + ": " + f.getFileName());
                                    continue;
                                }
                                width = bimg.getWidth();
                                height = bimg.getHeight();
                            } catch (Exception ex) {
                                unreadableFiles.add(f);
                                anomalies.add("Exception decoding image in " + partitionName + "/" + className + ": " + f.getFileName() + " (" + ex.getMessage() + ")");
                                continue;
                            }
                        }

                        int classId = taxonomy.findClassIdByLabel(className).orElse(-1);
                        String sampleId = partitionName + "_" + className + "_" + f.getFileName();
                        samples.add(new ImageSample(sampleId, f, Math.max(0, classId), sampleId, width, height, 0, 0, width, height));
                        imagePaths.add(f);
                        count++;
                    }
                }

                if (count == 0) {
                    emptyClasses.add(className);
                }
                classCounts.put(className, count);
            }
        }

        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int c : classCounts.values()) {
            if (c > 0 && c < min) min = c;
            if (c > max) max = c;
        }
        if (min == Integer.MAX_VALUE) {
            min = 0;
        }
        double ratio = min > 0 ? (double) max / min : (max > 0 ? Double.POSITIVE_INFINITY : 1.0);

        return new ScanResult(samples, imagePaths, classCounts, emptyClasses, min, max, ratio);
    }
}
