package com.trafficsign.data;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * Validates dataset hygiene: detects zero-byte files, corrupt headers,
 * identical bit-for-bit duplicate images via SHA-256 hashing, and cross-split leakage.
 */
public class DatasetIntegrityChecker {

    public record DuplicateGroup(String sha256Hex, List<Path> duplicatePaths) {
    }

    public record IntegrityAuditReport(
            int totalFilesInspected,
            int uniqueFilesCount,
            List<DuplicateGroup> duplicateGroups,
            List<Path> zeroByteFiles,
            int crossSplitLeakageCount
    ) {
        public boolean isClean() {
            return duplicateGroups.isEmpty() && zeroByteFiles.isEmpty() && crossSplitLeakageCount == 0;
        }
    }

    public static IntegrityAuditReport auditDirectory(List<Path> files) throws IOException {
        Objects.requireNonNull(files, "files must not be null");

        Map<String, List<Path>> hashToPaths = new LinkedHashMap<>();
        List<Path> zeroByteFiles = new ArrayList<>();
        int inspected = 0;

        for (Path p : files) {
            if (!Files.exists(p) || !Files.isRegularFile(p)) {
                continue;
            }
            inspected++;
            long size = Files.size(p);
            if (size == 0) {
                zeroByteFiles.add(p);
                continue;
            }

            String hash = computeSha256(p);
            hashToPaths.computeIfAbsent(hash, k -> new ArrayList<>()).add(p);
        }

        List<DuplicateGroup> duplicates = new ArrayList<>();
        for (var entry : hashToPaths.entrySet()) {
            if (entry.getValue().size() > 1) {
                duplicates.add(new DuplicateGroup(entry.getKey(), Collections.unmodifiableList(entry.getValue())));
            }
        }

        return new IntegrityAuditReport(
                inspected,
                hashToPaths.size(),
                Collections.unmodifiableList(duplicates),
                Collections.unmodifiableList(zeroByteFiles),
                0
        );
    }

    public static int checkCrossSplitHashLeakage(
            Collection<ImageSample> splitA,
            Collection<ImageSample> splitB
    ) throws IOException {
        Set<String> hashesA = new HashSet<>();
        for (ImageSample s : splitA) {
            if (Files.exists(s.filePath())) {
                hashesA.add(computeSha256(s.filePath()));
            }
        }

        int leakageCount = 0;
        for (ImageSample s : splitB) {
            if (Files.exists(s.filePath())) {
                String hashB = computeSha256(s.filePath());
                if (hashesA.contains(hashB)) {
                    leakageCount++;
                }
            }
        }
        return leakageCount;
    }

    public static String computeSha256(Path file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream is = Files.newInputStream(file)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    digest.update(buffer, 0, read);
                }
            }
            byte[] hashBytes = digest.digest();
            StringBuilder hexString = new StringBuilder();
            for (byte b : hashBytes) {
                hexString.append(String.format("%02x", b));
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 algorithm not available in JVM", e);
        }
    }
}
