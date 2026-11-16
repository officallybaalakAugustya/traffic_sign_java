package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class ClassificationDatasetValidatorTest {

    @Test
    @DisplayName("Should validate class consistency, detect imbalance, unexpected extensions, and partition train/val")
    void shouldValidateClassificationDataset(@TempDir Path tempDir) throws IOException {
        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();
        ClassificationDatasetValidator validator = new ClassificationDatasetValidator(taxonomy);

        Path trainDir = tempDir.resolve("train");
        Path testDir = tempDir.resolve("test");
        Files.createDirectories(trainDir);
        Files.createDirectories(testDir);

        createClassWithImages(trainDir, "STOP", 5);
        createClassWithImages(trainDir, "CATTLE", 2);
        createClassWithImages(trainDir, "SPEED_LIMIT_40", 1);
        Files.createDirectories(trainDir.resolve("EMPTY_CLASS"));

        Files.writeString(trainDir.resolve("STOP").resolve("notes.txt"), "unexpected text file");

        createClassWithImages(testDir, "STOP", 2);
        createClassWithImages(testDir, "CATTLE", 1);
        createClassWithImages(testDir, "SPEED_LIMIT_40", 1);
        createClassWithImages(testDir, "GIVE_WAY", 1);

        ClassificationValidationResult result = validator.validate(tempDir, 42L, 0.25, true);

        assertThat(result.totalTrainImages()).isEqualTo(8);
        assertThat(result.totalTestImages()).isEqualTo(5);
        assertThat(result.emptyClasses()).contains("EMPTY_CLASS");
        assertThat(result.classesOnlyInTrain()).contains("EMPTY_CLASS");
        assertThat(result.classesOnlyInTest()).contains("GIVE_WAY");
        assertThat(result.isClassConsistent()).isFalse();

        assertThat(result.unexpectedExtensionFiles()).hasSize(1);
        assertThat(result.trainMinSupport()).isEqualTo(1);
        assertThat(result.trainMaxSupport()).isEqualTo(5);
        assertThat(result.trainImbalanceRatio()).isEqualTo(5.0);

        assertThat(result.trainValSplitTrainCount()).isGreaterThan(0);
        assertThat(result.trainValSplitValCount()).isGreaterThan(0);
        assertThat(result.trainValSplitTrainCount() + result.trainValSplitValCount()).isEqualTo(8);

        String markdown = validator.generateMarkdownReport(result);
        assertThat(markdown).contains("# Classification Dataset Validation Report");
        assertThat(markdown).contains("EMPTY_CLASS");
        assertThat(markdown).contains("GIVE_WAY");

        String json = validator.generateJsonReport(result);
        assertThat(json).contains("\"totalTrainImages\" : 8");
    }

    @Test
    @DisplayName("Should detect identical content duplicates within partitions")
    void shouldDetectDuplicatesWithinPartitions(@TempDir Path tempDir) throws IOException {
        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();
        ClassificationDatasetValidator validator = new ClassificationDatasetValidator(taxonomy);

        Path trainDir = tempDir.resolve("train");
        Path testDir = tempDir.resolve("test");
        Files.createDirectories(trainDir);
        Files.createDirectories(testDir);

        Path stopDir = trainDir.resolve("STOP");
        Files.createDirectories(stopDir);

        byte[] duplicateBytes = new byte[]{10, 20, 30, 40};
        Files.write(stopDir.resolve("img1.png"), duplicateBytes);
        Files.write(stopDir.resolve("img2.png"), duplicateBytes);

        createClassWithImages(testDir, "STOP", 1);

        ClassificationValidationResult result = validator.validate(tempDir, 42L, 0.20, false);
        assertThat(result.trainDuplicates()).hasSize(1);
    }

    private void createClassWithImages(Path partitionDir, String className, int count) throws IOException {
        Path classDir = partitionDir.resolve(className);
        Files.createDirectories(classDir);
        for (int i = 0; i < count; i++) {
            BufferedImage img = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
            img.setRGB(0, 0, (i + 1) * 1000);
            ImageIO.write(img, "png", classDir.resolve("img_" + i + ".png").toFile());
        }
    }
}
