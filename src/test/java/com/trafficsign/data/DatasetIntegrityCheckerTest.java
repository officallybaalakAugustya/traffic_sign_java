package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class DatasetIntegrityCheckerTest {

    @Test
    @DisplayName("Should detect identical duplicates via SHA-256 and identify zero-byte corrupt files")
    void shouldAuditFilesAndDetectDuplicates(@TempDir Path tempDir) throws IOException {
        Path file1 = tempDir.resolve("original.png");
        Files.write(file1, new byte[]{1, 2, 3, 4, 5, 6, 7, 8});

        // Exact bit-for-bit duplicate
        Path file2 = tempDir.resolve("duplicate.png");
        Files.write(file2, new byte[]{1, 2, 3, 4, 5, 6, 7, 8});

        // Unique file
        Path file3 = tempDir.resolve("unique.png");
        Files.write(file3, new byte[]{9, 10, 11, 12});

        // Zero byte file
        Path emptyFile = tempDir.resolve("empty.png");
        Files.createFile(emptyFile);

        List<Path> files = List.of(file1, file2, file3, emptyFile);

        DatasetIntegrityChecker.IntegrityAuditReport report = DatasetIntegrityChecker.auditDirectory(files);

        assertThat(report.totalFilesInspected()).isEqualTo(4);
        assertThat(report.zeroByteFiles()).contains(emptyFile);
        assertThat(report.duplicateGroups()).hasSize(1);
        assertThat(report.duplicateGroups().getFirst().duplicatePaths()).contains(file1, file2);
        assertThat(report.isClean()).isFalse();
    }

    @Test
    @DisplayName("Should detect cross-split hash leakage between train and test partitions")
    void shouldDetectCrossSplitLeakage(@TempDir Path tempDir) throws IOException {
        Path trainImg = tempDir.resolve("train_img.jpg");
        Files.write(trainImg, new byte[]{42, 43, 44});

        Path testImgLeaked = tempDir.resolve("test_img_leaked.jpg");
        Files.write(testImgLeaked, new byte[]{42, 43, 44}); // identical content

        Path testImgUnique = tempDir.resolve("test_img_unique.jpg");
        Files.write(testImgUnique, new byte[]{99, 100});

        List<ImageSample> trainSplit = List.of(
                ImageSample.ofSimple("tr1", trainImg, 0, "g1", 32, 32)
        );

        List<ImageSample> testSplit = List.of(
                ImageSample.ofSimple("te1", testImgLeaked, 0, "g2", 32, 32),
                ImageSample.ofSimple("te2", testImgUnique, 1, "g3", 32, 32)
        );

        int leakage = DatasetIntegrityChecker.checkCrossSplitHashLeakage(trainSplit, testSplit);
        assertThat(leakage).isEqualTo(1);
    }
}
