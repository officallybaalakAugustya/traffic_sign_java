package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class ClassFrequencyReportTest {

    @Test
    @DisplayName("Should accurately report class counts, missing classes, and calculate imbalance ratio")
    void shouldCalculateClassFrequencies() {
        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();

        // Create mock samples: 10 samples of class 0, 5 samples of class 1, 2 samples of class 2
        List<ImageSample> samples = List.of(
                ImageSample.ofSimple("s1", Path.of("img1.jpg"), 0, "g1", 32, 32),
                ImageSample.ofSimple("s2", Path.of("img2.jpg"), 0, "g1", 32, 32),
                ImageSample.ofSimple("s3", Path.of("img3.jpg"), 0, "g1", 32, 32),
                ImageSample.ofSimple("s4", Path.of("img4.jpg"), 1, "g2", 32, 32),
                ImageSample.ofSimple("s5", Path.of("img5.jpg"), 1, "g2", 32, 32),
                ImageSample.ofSimple("s6", Path.of("img6.jpg"), 2, "g3", 32, 32)
        );

        Map<String, Integer> unmapped = Map.of("FOREIGN_SIGN_TEST", 2);

        ClassFrequencyReport report = ClassFrequencyReport.generate(samples, taxonomy, unmapped);

        assertThat(report.totalSamples()).isEqualTo(6);
        assertThat(report.classFrequencies().get(0)).isEqualTo(3);
        assertThat(report.classFrequencies().get(1)).isEqualTo(2);
        assertThat(report.classFrequencies().get(2)).isEqualTo(1);

        // Min support among populated classes = 1, max support = 3 -> ratio = 3.0
        assertThat(report.minSupport()).isEqualTo(1);
        assertThat(report.maxSupport()).isEqualTo(3);
        assertThat(report.imbalanceRatio()).isEqualTo(3.0);

        // Missing classes should contain class 3, 4, etc.
        assertThat(report.missingClassIds()).contains(3, 4, 5);
        assertThat(report.unmappedLabelCounts()).containsKey("FOREIGN_SIGN_TEST");

        assertThat(report.hasSevereImbalance(2.0)).isTrue();
        assertThat(report.hasSevereImbalance(5.0)).isFalse();
    }
}
