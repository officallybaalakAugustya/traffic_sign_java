package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class GroupedDatasetSplitterTest {

    @Test
    @DisplayName("Should partition samples into train/val/test splits without leaking group IDs across splits")
    void shouldPartitionWithoutGroupLeakage() {
        GroupedDatasetSplitter splitter = new GroupedDatasetSplitter(42L, 0.70, 0.15, 0.15);

        // Create 20 groups of samples across 2 classes
        List<ImageSample> samples = new ArrayList<>();
        for (int c = 0; c < 2; c++) {
            for (int g = 0; g < 10; g++) {
                String groupId = "class" + c + "_group" + g;
                for (int i = 0; i < 5; i++) {
                    String sampleId = groupId + "_img" + i;
                    samples.add(ImageSample.ofSimple(sampleId, Path.of(sampleId + ".png"), c, groupId, 32, 32));
                }
            }
        }

        GroupedDatasetSplitter.PartitionResult result = splitter.partition(samples);

        assertThat(result.totalSamples()).isEqualTo(100);
        assertThat(result.trainSamples()).isNotEmpty();
        assertThat(result.valSamples()).isNotEmpty();
        assertThat(result.testSamples()).isNotEmpty();

        // Verify zero group leakage
        Set<String> trainGroups = new HashSet<>();
        for (ImageSample s : result.trainSamples()) {
            trainGroups.add(s.trackId());
        }

        Set<String> valGroups = new HashSet<>();
        for (ImageSample s : result.valSamples()) {
            valGroups.add(s.trackId());
        }

        Set<String> testGroups = new HashSet<>();
        for (ImageSample s : result.testSamples()) {
            testGroups.add(s.trackId());
        }

        // Intersections must be empty
        assertThat(trainGroups).doesNotContainAnyElementsOf(valGroups);
        assertThat(trainGroups).doesNotContainAnyElementsOf(testGroups);
        assertThat(valGroups).doesNotContainAnyElementsOf(testGroups);
    }

    @Test
    @DisplayName("Should generate identical partitions for identical random seeds")
    void shouldBeDeterministicWithSameSeed() {
        List<ImageSample> samples = List.of(
                ImageSample.ofSimple("s1", Path.of("1.png"), 0, "g1", 32, 32),
                ImageSample.ofSimple("s2", Path.of("2.png"), 0, "g2", 32, 32),
                ImageSample.ofSimple("s3", Path.of("3.png"), 0, "g3", 32, 32),
                ImageSample.ofSimple("s4", Path.of("4.png"), 0, "g4", 32, 32)
        );

        GroupedDatasetSplitter s1 = new GroupedDatasetSplitter(12345L, 0.5, 0.25, 0.25);
        GroupedDatasetSplitter s2 = new GroupedDatasetSplitter(12345L, 0.5, 0.25, 0.25);

        GroupedDatasetSplitter.PartitionResult r1 = s1.partition(samples);
        GroupedDatasetSplitter.PartitionResult r2 = s2.partition(samples);

        assertThat(r1.trainSamples()).isEqualTo(r2.trainSamples());
        assertThat(r1.valSamples()).isEqualTo(r2.valSamples());
        assertThat(r1.testSamples()).isEqualTo(r2.testSamples());
    }

    @Test
    @DisplayName("Should validate split ratios sum to 1.0")
    void shouldRejectInvalidRatios() {
        assertThatThrownBy(() -> new GroupedDatasetSplitter(42L, 0.6, 0.2, 0.5))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must sum to 1.0");
    }
}
