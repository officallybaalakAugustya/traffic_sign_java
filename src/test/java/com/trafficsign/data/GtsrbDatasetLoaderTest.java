package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GTSRB Dataset Loader & Annotation Parsing Tests")
class GtsrbDatasetLoaderTest {

    @Test
    @DisplayName("Should extract correct trackId from standard GTSRB filename patterns")
    void testExtractTrackId() {
        assertEquals("00000", GtsrbDatasetLoader.extractTrackId("00000_00014.ppm"));
        assertEquals("00042", GtsrbDatasetLoader.extractTrackId("00042_00029.ppm"));
        assertEquals("trackA", GtsrbDatasetLoader.extractTrackId("trackA_00001.png"));
        assertEquals("single", GtsrbDatasetLoader.extractTrackId("single.ppm"));
    }

    @Test
    @DisplayName("Should parse standard GTSRB semicolon-separated CSV annotation file")
    void testParseClassCsv(@TempDir Path tempDir) throws IOException {
        Path csvFile = tempDir.resolve("GT-00001.csv");
        String csvContent = """
                Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                00001_00000.ppm;38;38;5;5;33;33;1
                00001_00001.ppm;39;40;6;5;34;35;1
                """;
        Files.writeString(csvFile, csvContent);

        List<ImageSample> samples = GtsrbDatasetLoader.parseClassCsv(csvFile, tempDir, 1);
        assertEquals(2, samples.size());

        ImageSample s0 = samples.get(0);
        assertEquals("00001_00000.ppm", s0.sampleId());
        assertEquals(1, s0.classId());
        assertEquals("00001", s0.trackId());
        assertEquals(38, s0.width());
        assertEquals(38, s0.height());
        assertEquals(5, s0.roiX1());
        assertEquals(5, s0.roiY1());
        assertEquals(33, s0.roiX2());
        assertEquals(33, s0.roiY2());

        // Validate summary calculation
        GtsrbDatasetLoader.DatasetValidationSummary summary = GtsrbDatasetLoader.validateDataset(samples);
        assertTrue(summary.isFullyValid());
        assertEquals(2, summary.totalSamples());
        assertEquals(1, summary.totalTracks());
        assertEquals(1, summary.classCount());
    }

    @Test
    @DisplayName("Should detect inverted ROI bounding box as validation error")
    void testInvertedRoiValidation(@TempDir Path tempDir) throws IOException {
        Path csvFile = tempDir.resolve("GT-00002.csv");
        String invalidContent = """
                Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                00002_00000.ppm;40;40;30;30;10;10;2
                """;
        Files.writeString(csvFile, invalidContent);

        List<ImageSample> samples = GtsrbDatasetLoader.parseClassCsv(csvFile, tempDir, 2);
        GtsrbDatasetLoader.DatasetValidationSummary summary = GtsrbDatasetLoader.validateDataset(samples);

        assertFalse(summary.isFullyValid());
        assertFalse(summary.validationErrors().isEmpty());
        assertTrue(summary.validationErrors().get(0).contains("Inverted ROI"));
    }
}
