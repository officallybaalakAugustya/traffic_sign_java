package com.trafficsign.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DatasetImportManagerTest {
    @Test
    void exactDuplicateIsSkippedIndependentOfPath() throws Exception {
        Path root = Files.createTempDirectory("import-test");
        Path raw = root.resolve("data/raw/existing");
        Files.createDirectories(raw);
        byte[] image = imageBytes();
        Files.write(raw.resolve("old-name.jpg"), image);
        Path zip = zip(root.resolve("incoming.zip"), new Item("nested/new-name.jpg", image));
        DatasetImportManager.Report report = DatasetImportManager.audit(root, zip);
        assertThat(report.proposedImports()).isZero();
        assertThat(report.exactDuplicates()).isEqualTo(1);
        assertThat(report.archives().get(0).files().get(0).reason()).contains("data/raw");
    }

    @Test
    void duplicateContentAcrossArchivesIsSkipped() throws Exception {
        Path root = Files.createTempDirectory("import-test");
        byte[] image = imageBytes();
        Path first = zip(root.resolve("a.zip"), new Item("a.jpg", image));
        Path second = zip(root.resolve("b.zip"), new Item("b.jpg", image));
        DatasetImportManager.Report report = DatasetImportManager.audit(root, root);
        assertThat(report.archives()).hasSize(2);
        assertThat(report.proposedImports()).isEqualTo(1);
        assertThat(report.exactDuplicates()).isEqualTo(1);
        assertThat(first).exists();
        assertThat(second).exists();
    }

    @Test
    void traversalAndCorruptArchivesAreRejected() throws Exception {
        Path root = Files.createTempDirectory("import-test");
        zip(root.resolve("unsafe.zip"), new Item("../escape.jpg", imageBytes()));
        Files.writeString(root.resolve("corrupt.zip"), "not a zip", StandardCharsets.UTF_8);
        DatasetImportManager.Report report = DatasetImportManager.audit(root, root);
        assertThat(report.archives()).extracting(DatasetImportManager.Archive::classification).containsOnly("REJECTED");
    }

    @Test
    void reviewAnnotationsAreNotProposedForImport() throws Exception {
        Path root = Files.createTempDirectory("import-test");
        Path zip = zip(root.resolve("yolo.zip"), new Item("Dataset/images/x.jpg", imageBytes()), new Item("Dataset/labels/x.txt", "0 0.5 0.5 0.2 0.2".getBytes(StandardCharsets.UTF_8)));
        DatasetImportManager.Report report = DatasetImportManager.audit(root, zip);
        assertThat(report.archives().get(0).classification()).isEqualTo("NEEDS_REVIEW");
        assertThat(report.proposedImports()).isZero();
        assertThat(report.markdown()).contains("License and class mapping");
    }

    @Test
    void importIsIdempotentAndPreservesExistingCollision() throws Exception {
        Path root = Files.createTempDirectory("import-test");
        byte[] image = imageBytes();
        Path zip = zip(root.resolve("incoming.zip"), new Item("images/new.jpg", image));
        DatasetImportManager.Report report = DatasetImportManager.audit(root, zip);
        Path plan = root.resolve("plan.json");
        new ObjectMapper().writeValue(plan.toFile(), report);
        DatasetImportManager.ImportResult first = DatasetImportManager.importPlan(root, plan);
        DatasetImportManager.ImportResult second = DatasetImportManager.importPlan(root, plan);
        assertThat(first.imported()).isEqualTo(1);
        assertThat(second.imported()).isZero();
        assertThat(Files.exists(root.resolve("data/raw/imported/incoming.zip/images/new.jpg"))).isTrue();
    }

    private static Path zip(Path path, Item... items) throws IOException {
        try (ZipOutputStream out = new ZipOutputStream(Files.newOutputStream(path))) {
            for (Item item : items) {
                out.putNextEntry(new ZipEntry(item.name()));
                out.write(item.bytes());
                out.closeEntry();
            }
        }
        return path;
    }

    private static byte[] imageBytes() throws IOException {
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", out);
        return out.toByteArray();
    }

    private record Item(String name, byte[] bytes) {
    }
}
