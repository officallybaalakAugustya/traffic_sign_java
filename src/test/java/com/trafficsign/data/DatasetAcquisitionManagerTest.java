package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Dataset Acquisition & Safe Extraction Security Tests")
class DatasetAcquisitionManagerTest {

    @Test
    @DisplayName("Dry-run inspection should verify expected archive metadata")
    void testDryRunInspection(@TempDir Path tempDir) {
        DatasetAcquisitionManager.AcquisitionDryRunReport report =
                DatasetAcquisitionManager.performDryRun(tempDir);

        assertNotNull(report);
        assertEquals(tempDir, report.targetRawDirectory());
        assertTrue(report.requiredDiskSpaceBytes() > 0);
        assertNotNull(report.trainingArchive());
        assertEquals(276_294_756L, report.trainingArchive().expectedBytes());
        assertEquals(88_978_620L, report.testImagesArchive().expectedBytes());
        assertEquals(99_620L, report.testGtArchive().expectedBytes());
    }

    @Test
    @DisplayName("SafeZipExtract should successfully unpack legitimate archive entries")
    void testSafeExtractValidZip(@TempDir Path tempDir) throws IOException {
        Path zipFile = tempDir.resolve("test_valid.zip");
        Path extractDir = tempDir.resolve("extracted");

        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipFile))) {
            ZipEntry entry = new ZipEntry("subdir/hello.txt");
            zos.putNextEntry(entry);
            zos.write("GTSRB Validation".getBytes());
            zos.closeEntry();
        }

        int count = DatasetAcquisitionManager.safeExtractZip(zipFile, extractDir);
        assertEquals(1, count);
        Path unpacked = extractDir.resolve("subdir").resolve("hello.txt");
        assertTrue(Files.exists(unpacked));
        assertEquals("GTSRB Validation", Files.readString(unpacked));
    }

    @Test
    @DisplayName("SafeZipExtract should actively reject Zip Slip path traversal attempts with SecurityException")
    void testZipSlipRejection(@TempDir Path tempDir) throws IOException {
        Path maliciousZip = tempDir.resolve("malicious.zip");
        Path extractDir = tempDir.resolve("extracted");

        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(maliciousZip))) {
            // Malicious entry attempting to escape extraction directory
            ZipEntry entry = new ZipEntry("../evil_outside.txt");
            zos.putNextEntry(entry);
            zos.write("Malicious payload".getBytes());
            zos.closeEntry();
        }

        assertThrows(SecurityException.class, () ->
                DatasetAcquisitionManager.safeExtractZip(maliciousZip, extractDir));
    }
}
