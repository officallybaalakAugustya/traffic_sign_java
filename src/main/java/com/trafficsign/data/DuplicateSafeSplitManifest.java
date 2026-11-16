package com.trafficsign.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class DuplicateSafeSplitManifest {

    public enum PartitionCategory {
        TRAIN,
        VALIDATION,
        CLEAN_TEST,
        EXCLUDED_TRAIN_OVERLAP
    }

    public record PartitionEntry(
            String sampleId,
            String relativePath,
            int classId,
            String classLabel,
            String contentHash,
            PartitionCategory partition
    ) {
    }

    public record ManifestSummary(
            int totalTrainFilesScanned,
            int totalTestFilesScanned,
            int uniqueTrainHashes,
            int uniqueTestHashes,
            int overlappingUniqueHashes,
            int trainFilesAssigned,
            int valFilesAssigned,
            int cleanTestFilesAssigned,
            int excludedTestFilesAssigned,
            long randomSeed,
            double valRatio,
            List<PartitionEntry> entries
    ) {
    }

    public static ManifestSummary generate(
            Path datasetRoot,
            double valRatio,
            long randomSeed,
            IndianTrafficSignTaxonomy taxonomy
    ) throws IOException {
        Path trainDir = datasetRoot.resolve("train");
        Path testDir = datasetRoot.resolve("test");

        if (!Files.exists(trainDir) || !Files.isDirectory(trainDir)) {
            throw new IllegalArgumentException("Train directory does not exist: " + trainDir);
        }
        if (!Files.exists(testDir) || !Files.isDirectory(testDir)) {
            throw new IllegalArgumentException("Test directory does not exist: " + testDir);
        }

        Map<String, List<FileItem>> trainHashToFiles = new LinkedHashMap<>();
        int trainCount = scanDirectory(trainDir, taxonomy, trainHashToFiles);

        Map<String, List<FileItem>> testHashToFiles = new LinkedHashMap<>();
        int testCount = scanDirectory(testDir, taxonomy, testHashToFiles);

        Set<String> trainHashSet = trainHashToFiles.keySet();
        Set<String> testHashSet = testHashToFiles.keySet();

        Set<String> overlappingHashes = new HashSet<>(testHashSet);
        overlappingHashes.retainAll(trainHashSet);

        Map<Integer, List<String>> classToTrainHashes = new TreeMap<>();
        for (Map.Entry<String, List<FileItem>> entry : trainHashToFiles.entrySet()) {
            int classId = entry.getValue().getFirst().classId;
            classToTrainHashes.computeIfAbsent(classId, k -> new ArrayList<>()).add(entry.getKey());
        }

        Random rng = new Random(randomSeed);
        Set<String> assignedTrainHashes = new HashSet<>();
        Set<String> assignedValHashes = new HashSet<>();

        for (Map.Entry<Integer, List<String>> entry : classToTrainHashes.entrySet()) {
            List<String> hashes = new ArrayList<>(entry.getValue());
            Collections.sort(hashes);
            Collections.shuffle(hashes, rng);

            int n = hashes.size();
            int valHCount = (int) Math.round(n * valRatio);
            if (n > 1 && valHCount == 0) {
                valHCount = 1;
            } else if (n > 1 && valHCount >= n) {
                valHCount = n - 1;
            }

            int trainHCount = n - valHCount;
            for (int i = 0; i < trainHCount; i++) {
                assignedTrainHashes.add(hashes.get(i));
            }
            for (int i = trainHCount; i < n; i++) {
                assignedValHashes.add(hashes.get(i));
            }
        }

        List<PartitionEntry> allEntries = new ArrayList<>();
        int trainAssignedCount = 0;
        int valAssignedCount = 0;
        int cleanTestCount = 0;
        int excludedCount = 0;

        for (Map.Entry<String, List<FileItem>> entry : trainHashToFiles.entrySet()) {
            String hash = entry.getKey();
            PartitionCategory cat = assignedTrainHashes.contains(hash)
                    ? PartitionCategory.TRAIN
                    : PartitionCategory.VALIDATION;

            for (FileItem item : entry.getValue()) {
                if (cat == PartitionCategory.TRAIN) {
                    trainAssignedCount++;
                } else {
                    valAssignedCount++;
                }
                allEntries.add(new PartitionEntry(
                        item.sampleId,
                        item.relativePath,
                        item.classId,
                        item.classLabel,
                        hash,
                        cat
                ));
            }
        }

        for (Map.Entry<String, List<FileItem>> entry : testHashToFiles.entrySet()) {
            String hash = entry.getKey();
            boolean isOverlap = overlappingHashes.contains(hash);
            PartitionCategory cat = isOverlap
                    ? PartitionCategory.EXCLUDED_TRAIN_OVERLAP
                    : PartitionCategory.CLEAN_TEST;

            for (FileItem item : entry.getValue()) {
                if (isOverlap) {
                    excludedCount++;
                } else {
                    cleanTestCount++;
                }
                allEntries.add(new PartitionEntry(
                        item.sampleId,
                        item.relativePath,
                        item.classId,
                        item.classLabel,
                        hash,
                        cat
                ));
            }
        }

        return new ManifestSummary(
                trainCount,
                testCount,
                trainHashSet.size(),
                testHashSet.size(),
                overlappingHashes.size(),
                trainAssignedCount,
                valAssignedCount,
                cleanTestCount,
                excludedCount,
                randomSeed,
                valRatio,
                Collections.unmodifiableList(allEntries)
        );
    }

    private static int scanDirectory(
            Path baseDir,
            IndianTrafficSignTaxonomy taxonomy,
            Map<String, List<FileItem>> hashToFiles
    ) throws IOException {
        int count = 0;
        try (var stream = Files.walk(baseDir)) {
            List<Path> files = stream.filter(Files::isRegularFile)
                    .filter(p -> isSupported(p.getFileName().toString()))
                    .sorted()
                    .toList();

            for (Path f : files) {
                count++;
                String hash = DatasetIntegrityChecker.computeSha256(f);
                String folderName = f.getParent().getFileName().toString();
                int classId = taxonomy.findClassIdByLabel(folderName).orElse(0);
                String relPath = baseDir.getParent().relativize(f).toString().replace('\\', '/');
                String sampleId = folderName + "_" + f.getFileName().toString();

                hashToFiles.computeIfAbsent(hash, k -> new ArrayList<>())
                        .add(new FileItem(sampleId, relPath, classId, folderName));
            }
        }
        return count;
    }

    private static boolean isSupported(String name) {
        String lower = name.toLowerCase();
        return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg");
    }

    private record FileItem(String sampleId, String relativePath, int classId, String classLabel) {
    }
}
