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

class ImageFolderDatasetLoaderTest {

    @Test
    @DisplayName("Should scan directory structure, load valid images, skip corrupt files, and flag unmapped folders")
    void shouldLoadImageFolderDataset(@TempDir Path tempDir) throws IOException {
        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();
        ImageFolderDatasetLoader loader = new ImageFolderDatasetLoader(taxonomy);

        // 1. Create mapped class directory: "CATTLE"
        Path cattleDir = tempDir.resolve("CATTLE");
        Files.createDirectories(cattleDir);

        BufferedImage img1 = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(img1, "png", cattleDir.resolve("cattle_01.png").toFile());

        BufferedImage img2 = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(img2, "jpg", cattleDir.resolve("cattle_02.jpg").toFile());

        // Zero-byte corrupt file
        Files.createFile(cattleDir.resolve("corrupt_zero.png"));

        // 2. Create mapped class directory: "STOP"
        Path stopDir = tempDir.resolve("STOP");
        Files.createDirectories(stopDir);

        BufferedImage stopImg = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(stopImg, "png", stopDir.resolve("stop_01.png").toFile());

        // 3. Create unmapped class directory: "UNMAPPED_FOREIGN_SIGN"
        Path unmappedDir = tempDir.resolve("UNMAPPED_FOREIGN_SIGN");
        Files.createDirectories(unmappedDir);

        BufferedImage foreignImg = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(foreignImg, "png", unmappedDir.resolve("foreign_01.png").toFile());

        // 4. Non-image file in class directory (.txt)
        Files.writeString(stopDir.resolve("readme.txt"), "ignorable annotation text");

        // Execute load
        ImageFolderDatasetLoader.LoadResult result = loader.load(tempDir, true);

        assertThat(result.totalFilesScanned()).isGreaterThanOrEqualTo(4);
        assertThat(result.samples()).hasSize(3); // 2 cattle + 1 stop
        assertThat(result.corruptOrUnreadableFiles()).hasSize(1);
        assertThat(result.corruptOrUnreadableFiles().getFirst().getFileName().toString()).isEqualTo("corrupt_zero.png");

        // Verify frequency report
        ClassFrequencyReport report = result.frequencyReport();
        assertThat(report.totalSamples()).isEqualTo(3);
        assertThat(report.unmappedLabelCounts()).containsKey("UNMAPPED_FOREIGN_SIGN");

        int cattleId = taxonomy.findClassIdByLabel("CATTLE").orElseThrow();
        int stopId = taxonomy.findClassIdByLabel("STOP").orElseThrow();

        assertThat(report.classFrequencies().get(cattleId)).isEqualTo(2);
        assertThat(report.classFrequencies().get(stopId)).isEqualTo(1);
    }
}
