package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class DuplicateSafeSplitManifestTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("DuplicateSafeSplitManifest isolates duplicates and prevents cross-split leakage")
    void testDuplicateSafePartitioning() throws IOException {
        Path root = tempDir.resolve("dataset");
        Path trainDir = root.resolve("train");
        Path testDir = root.resolve("test");

        Path trainClass0 = trainDir.resolve("STOP");
        Path trainClass1 = trainDir.resolve("GIVE_WAY");
        Path testClass0 = testDir.resolve("STOP");
        Path testClass1 = testDir.resolve("GIVE_WAY");

        Files.createDirectories(trainClass0);
        Files.createDirectories(trainClass1);
        Files.createDirectories(testClass0);
        Files.createDirectories(testClass1);

        BufferedImage imgA = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        BufferedImage imgB = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        BufferedImage imgC = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        imgB.setRGB(0, 0, 0xFF0000);
        imgC.setRGB(0, 0, 0x00FF00);

        ImageIO.write(imgA, "PNG", trainClass0.resolve("imgA_1.png").toFile());
        ImageIO.write(imgA, "PNG", trainClass0.resolve("imgA_2.png").toFile());
        ImageIO.write(imgB, "PNG", trainClass0.resolve("imgB.png").toFile());
        ImageIO.write(imgC, "PNG", trainClass1.resolve("imgC.png").toFile());

        ImageIO.write(imgA, "PNG", testClass0.resolve("imgA_test.png").toFile());
        BufferedImage imgD = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        imgD.setRGB(0, 0, 0x0000FF);
        ImageIO.write(imgD, "PNG", testClass1.resolve("imgD_test.png").toFile());

        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();
        DuplicateSafeSplitManifest.ManifestSummary summary = DuplicateSafeSplitManifest.generate(
                root, 0.50, 42L, taxonomy
        );

        assertThat(summary.totalTrainFilesScanned()).isEqualTo(4);
        assertThat(summary.totalTestFilesScanned()).isEqualTo(2);
        assertThat(summary.overlappingUniqueHashes()).isEqualTo(1);
        assertThat(summary.excludedTestFilesAssigned()).isEqualTo(1);
        assertThat(summary.cleanTestFilesAssigned()).isEqualTo(1);

        Set<String> trainHashes = new HashSet<>();
        Set<String> valHashes = new HashSet<>();
        Set<String> cleanTestHashes = new HashSet<>();

        for (var entry : summary.entries()) {
            switch (entry.partition()) {
                case TRAIN -> trainHashes.add(entry.contentHash());
                case VALIDATION -> valHashes.add(entry.contentHash());
                case CLEAN_TEST -> cleanTestHashes.add(entry.contentHash());
                case EXCLUDED_TRAIN_OVERLAP -> {
                    assertThat(trainHashes.contains(entry.contentHash()) || valHashes.contains(entry.contentHash())).isTrue();
                }
            }
        }

        Set<String> trainValIntersect = new HashSet<>(trainHashes);
        trainValIntersect.retainAll(valHashes);
        assertThat(trainValIntersect).isEmpty();

        Set<String> trainTestIntersect = new HashSet<>(trainHashes);
        trainTestIntersect.retainAll(cleanTestHashes);
        assertThat(trainTestIntersect).isEmpty();

        Set<String> valTestIntersect = new HashSet<>(valHashes);
        valTestIntersect.retainAll(cleanTestHashes);
        assertThat(valTestIntersect).isEmpty();
    }
}
