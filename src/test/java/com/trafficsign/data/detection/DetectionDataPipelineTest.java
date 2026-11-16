package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;
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
import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class DetectionDataPipelineTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("DetectionDataPipeline correctly extracts generic traffic signs, excludes degenerates, and partitions by sequence ID")
    void testPipelineAndSequenceIsolation() throws IOException {
        Path tarPath = tempDir.resolve("sample-00000.tar");

        try (OutputStream fos = Files.newOutputStream(tarPath);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             TarArchiveOutputStream tos = new TarArchiveOutputStream(bos)) {

            addTarEntry(tos, "seqA_0000.jpg", new byte[]{1, 2, 3});
            String jsonA0 = """
                    {
                      "labels": [
                        { "category": "traffic sign", "box2d": { "x1": 100, "y1": 150, "x2": 200, "y2": 250 } },
                        { "category": "car", "box2d": { "x1": 300, "y1": 400, "x2": 500, "y2": 600 } },
                        { "category": "traffic sign", "box2d": { "x1": 50, "y1": 50, "x2": 50, "y2": 80 } }
                      ]
                    }
                    """;
            addTarEntry(tos, "seqA_0000.json", jsonA0.getBytes(StandardCharsets.UTF_8));

            addTarEntry(tos, "seqA_0001.jpg", new byte[]{4, 5, 6});
            String jsonA1 = """
                    {
                      "labels": [
                        { "category": "traffic sign", "box2d": { "x1": 110, "y1": 155, "x2": 210, "y2": 255 } }
                      ]
                    }
                    """;
            addTarEntry(tos, "seqA_0001.json", jsonA1.getBytes(StandardCharsets.UTF_8));

            addTarEntry(tos, "seqB_0000.jpg", new byte[]{7, 8, 9});
            String jsonB0 = """
                    {
                      "labels": [
                        { "category": "person", "box2d": { "x1": 10, "y1": 20, "x2": 30, "y2": 40 } }
                      ]
                    }
                    """;
            addTarEntry(tos, "seqB_0000.json", jsonB0.getBytes(StandardCharsets.UTF_8));
        }

        DetectionDataPipeline pipeline = new DetectionDataPipeline();
        DetectionDataPipeline.ManifestSummary summary = pipeline.buildManifest(
                tempDir, 1, 0.50, 42L
        );

        assertThat(summary.totalFramesScanned()).isEqualTo(3);
        assertThat(summary.totalMatchedJpgJsonPairs()).isEqualTo(3);
        assertThat(summary.totalSequences()).isEqualTo(2);
        assertThat(summary.excludedAnnotationsCount()).isEqualTo(1);

        Set<String> trainSeqs = new HashSet<>();
        Set<String> valSeqs = new HashSet<>();

        for (var sample : summary.samples()) {
            if (sample.partition() == DetectionDataPipeline.SplitPartition.TRAIN) {
                trainSeqs.add(sample.sequenceId());
            } else {
                valSeqs.add(sample.sequenceId());
            }
        }

        assertThat(trainSeqs).hasSize(1);
        assertThat(valSeqs).hasSize(1);
        trainSeqs.retainAll(valSeqs);
        assertThat(trainSeqs).isEmpty();
    }

    private void addTarEntry(TarArchiveOutputStream tos, String name, byte[] data) throws IOException {
        TarArchiveEntry entry = new TarArchiveEntry(name);
        entry.setSize(data.length);
        tos.putArchiveEntry(entry);
        tos.write(data);
        tos.closeArchiveEntry();
    }
}
