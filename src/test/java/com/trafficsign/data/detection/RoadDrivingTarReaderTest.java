package com.trafficsign.data.detection;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RoadDrivingTarReaderTest {

    @Test
    @DisplayName("Should parse valid tar archive, match JPG/JSON stems, ignore PNGs, and filter generic signs")
    void shouldParseValidTarArchive(@TempDir Path tempDir) throws IOException {
        Path tarPath = tempDir.resolve("test_archive.tar");
        try (OutputStream fos = Files.newOutputStream(tarPath);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             TarArchiveOutputStream tos = new TarArchiveOutputStream(bos)) {

            writeTarEntry(tos, "frame_0001.jpg", new byte[]{1, 2, 3, 4});
            writeTarEntry(tos, "frame_0001.png", new byte[]{5, 6, 7, 8});

            String validJson = """
                    {
                      "name": "frame_0001.jpg",
                      "labels": [
                        {
                          "id": 1,
                          "category": "traffic sign",
                          "box2d": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 80.0},
                          "confidence": 0.88,
                          "attributes": {"occluded": false, "truncated": false},
                          "track_id": 101
                        },
                        {
                          "id": 2,
                          "category": "car",
                          "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
                          "confidence": 0.95,
                          "attributes": {"occluded": false, "truncated": false},
                          "track_id": 102
                        }
                      ]
                    }
                    """;
            writeTarEntry(tos, "frame_0001.json", validJson.getBytes(StandardCharsets.UTF_8));
        }

        RoadDrivingTarReader reader = new RoadDrivingTarReader();
        TarInspectionSummary summary = reader.inspectArchives(List.of(tarPath));

        assertThat(summary.totalArchivesProcessed()).isEqualTo(1);
        assertThat(summary.totalFrames()).isEqualTo(1);
        assertThat(summary.totalMatchedPairs()).isEqualTo(1);
        assertThat(summary.totalPngIgnored()).isEqualTo(1);
        assertThat(summary.totalObjects()).isEqualTo(2);
        assertThat(summary.totalTrafficSigns()).isEqualTo(1);

        assertThat(summary.categoryDistribution()).containsEntry("traffic sign", 1);
        assertThat(summary.categoryDistribution()).containsEntry("car", 1);

        FrameDetectionRecord frame = summary.sampleFrames().getFirst();
        assertThat(frame.frameStem()).isEqualTo("frame_0001");
        assertThat(frame.hasImage()).isTrue();
        assertThat(frame.hasJson()).isTrue();

        List<DetectionAnnotation> signAnnotations = frame.getTrafficSignAnnotations();
        assertThat(signAnnotations).hasSize(1);
        DetectionAnnotation sign = signAnnotations.getFirst();
        assertThat(sign.isGenericTrafficSign()).isTrue();
        assertThat(sign.box().x1()).isEqualTo(10.0);
        assertThat(sign.box().y1()).isEqualTo(20.0);
        assertThat(sign.box().x2()).isEqualTo(50.0);
        assertThat(sign.box().y2()).isEqualTo(80.0);
    }

    @Test
    @DisplayName("Should detect missing pairs, malformed json, out of bounds coordinates, and empty categories")
    void shouldHandleAnomaliesAndCorruptions(@TempDir Path tempDir) throws IOException {
        Path tarPath = tempDir.resolve("anomalies.tar");
        try (OutputStream fos = Files.newOutputStream(tarPath);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             TarArchiveOutputStream tos = new TarArchiveOutputStream(bos)) {

            writeTarEntry(tos, "orphan_image.jpg", new byte[]{1, 2, 3});

            String orphanJson = """
                    {"name": "orphan_json.jpg", "labels": []}
                    """;
            writeTarEntry(tos, "orphan_json.json", orphanJson.getBytes(StandardCharsets.UTF_8));

            writeTarEntry(tos, "corrupt_json.jpg", new byte[]{1, 2, 3});
            writeTarEntry(tos, "corrupt_json.json", "{ malformed json content ...".getBytes(StandardCharsets.UTF_8));

            writeTarEntry(tos, "bad_box.jpg", new byte[]{1, 2, 3});
            String badBoxJson = """
                    {
                      "name": "bad_box.jpg",
                      "labels": [
                        {
                          "id": 1,
                          "category": "",
                          "box2d": {"x1": 50.0, "y1": 20.0, "x2": 10.0, "y2": 80.0}
                        },
                        {
                          "id": 2,
                          "category": "traffic sign",
                          "box2d": {"x1": 10.0, "y1": 20.0, "x2": 5000.0, "y2": 80.0}
                        }
                      ]
                    }
                    """;
            writeTarEntry(tos, "bad_box.json", badBoxJson.getBytes(StandardCharsets.UTF_8));
        }

        RoadDrivingTarReader reader = new RoadDrivingTarReader();
        TarInspectionSummary summary = reader.inspectArchives(List.of(tarPath));

        assertThat(summary.totalJpgOnly()).isEqualTo(1);
        assertThat(summary.totalJsonOnly()).isEqualTo(1);
        assertThat(summary.annotationErrors()).isNotEmpty();

        boolean foundMalformed = summary.annotationErrors().stream().anyMatch(e -> e.contains("Malformed JSON"));
        boolean foundInverted = summary.annotationErrors().stream().anyMatch(e -> e.contains("Inverted or negative"));
        boolean foundOutOfBounds = summary.annotationErrors().stream().anyMatch(e -> e.contains("Severe out-of-bounds"));
        boolean foundEmptyCategory = summary.annotationErrors().stream().anyMatch(e -> e.contains("Empty or null category"));

        assertThat(foundMalformed).isTrue();
        assertThat(foundInverted).isTrue();
        assertThat(foundOutOfBounds).isTrue();
        assertThat(foundEmptyCategory).isTrue();
    }

    @Test
    @DisplayName("Should respect configurable maxArchives limit when scanning directory")
    void shouldRespectMaxArchivesLimit(@TempDir Path tempDir) throws IOException {
        for (int i = 0; i < 5; i++) {
            Path p = tempDir.resolve("archive_" + i + ".tar");
            try (OutputStream fos = Files.newOutputStream(p);
                 BufferedOutputStream bos = new BufferedOutputStream(fos);
                 TarArchiveOutputStream tos = new TarArchiveOutputStream(bos)) {
                writeTarEntry(tos, "dummy_" + i + ".jpg", new byte[]{1});
                writeTarEntry(tos, "dummy_" + i + ".json", "{\"name\":\"d\",\"labels\":[]}".getBytes(StandardCharsets.UTF_8));
            }
        }

        RoadDrivingTarReader reader = new RoadDrivingTarReader();
        TarInspectionSummary summary = reader.inspectDirectory(tempDir, 2);

        assertThat(summary.totalArchivesProcessed()).isEqualTo(2);
        assertThat(summary.archiveNames()).hasSize(2);
    }

    private void writeTarEntry(TarArchiveOutputStream tos, String name, byte[] content) throws IOException {
        TarArchiveEntry entry = new TarArchiveEntry(name);
        entry.setSize(content.length);
        tos.putArchiveEntry(entry);
        tos.write(content);
        tos.closeArchiveEntry();
    }
}
