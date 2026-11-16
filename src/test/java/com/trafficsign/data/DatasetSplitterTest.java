package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

class DatasetSplitterTest {

    @Test
    @DisplayName("Should generate reproducible track-aware splits with zero track leakage")
    void shouldCreateZeroLeakageSplits() {
        // Create 20 synthetic tracks across 4 classes, each track having 10 frames (200 samples)
        List<ImageSample> samples = new ArrayList<>();
        for (int c = 0; c < 4; c++) {
            for (int t = 0; t < 5; t++) {
                String trackId = String.format("class_%02d_track_%02d", c, t);
                for (int f = 0; f < 10; f++) {
                    String sampleId = String.format("%s_frame_%02d", trackId, f);
                    Path fakePath = Path.of("raw", String.format("%02d", c), sampleId + ".ppm");
                    samples.add(ImageSample.ofSimple(sampleId, fakePath, c, trackId, 32, 32));
                }
            }
        }

        DatasetSplitter splitter = new DatasetSplitter(42L, 0.70, 0.15, 0.15);
        SplitManifest manifest = splitter.split("SyntheticGTSRB", samples);

        assertThat(manifest.totalSamples()).isEqualTo(200);
        assertThat(manifest.totalTracks()).isEqualTo(20);

        // Verify that every sample from a given track is assigned to the SAME split partition
        Map<String, Set<SplitType>> trackToSplits = manifest.records().stream()
                .collect(Collectors.groupingBy(
                        SplitRecord::trackId,
                        Collectors.mapping(SplitRecord::splitType, Collectors.toSet())
                ));

        for (Map.Entry<String, Set<SplitType>> entry : trackToSplits.entrySet()) {
            assertThat(entry.getValue())
                    .as("Track " + entry.getKey() + " must strictly map to exactly one split")
                    .hasSize(1);
        }

        // Verify LeakageDetector confirms clean partition
        LeakageDetector detector = new LeakageDetector();
        LeakageDetector.LeakageReport report = detector.inspect(manifest);
        assertThat(report.isClean()).isTrue();
        assertThat(report.errors()).isEmpty();
    }

    @Test
    @DisplayName("Should serialize and deserialize SplitManifest to JSON without loss")
    void shouldSerializeAndDeserializeManifest(@TempDir Path tempDir) throws IOException {
        List<ImageSample> samples = List.of(
                ImageSample.ofSimple("s1", Path.of("1.ppm"), 0, "tr1", 32, 32),
                ImageSample.ofSimple("s2", Path.of("2.ppm"), 0, "tr1", 32, 32),
                ImageSample.ofSimple("s3", Path.of("3.ppm"), 1, "tr2", 32, 32)
        );

        DatasetSplitter splitter = DatasetSplitter.defaultStandard();
        SplitManifest original = splitter.split("TestSet", samples);

        Path manifestFile = tempDir.resolve("split_manifest.json");
        splitter.saveManifest(original, manifestFile);

        assertThat(manifestFile).exists();

        SplitManifest loaded = splitter.loadManifest(manifestFile);
        assertThat(loaded.datasetName()).isEqualTo(original.datasetName());
        assertThat(loaded.totalSamples()).isEqualTo(original.totalSamples());
        assertThat(loaded.totalTracks()).isEqualTo(original.totalTracks());
        assertThat(loaded.records()).hasSameSizeAs(original.records());
    }

    @Test
    @DisplayName("Should detect and flag artificial track leakage when injected")
    void shouldDetectInjectedLeakage() {
        List<SplitRecord> leakyRecords = List.of(
                new SplitRecord("s1", "track_A", 1, SplitType.TRAIN, "path1.ppm"),
                new SplitRecord("s2", "track_A", 1, SplitType.TEST, "path2.ppm"), // Leak! track_A in both Train and Test
                new SplitRecord("s3", "track_B", 2, SplitType.VALIDATION, "path3.ppm")
        );

        LeakageDetector detector = new LeakageDetector();
        LeakageDetector.LeakageReport report = detector.inspect(leakyRecords);

        assertThat(report.isClean()).isFalse();
        assertThat(report.hasErrors()).isTrue();
        assertThat(report.errors()).anyMatch(err -> err.contains("Data Leakage Error") && err.contains("track_A"));
    }
}
