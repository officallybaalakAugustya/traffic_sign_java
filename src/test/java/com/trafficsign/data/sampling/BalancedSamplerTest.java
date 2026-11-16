package com.trafficsign.data.sampling;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class BalancedSamplerTest {

    @Test
    @DisplayName("BalancedSampler generates balanced index distribution and respects class caps")
    void testBalancedSamplerAndCappedWeights() {
        int[] labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        int[] sampled = BalancedSampler.generateBalancedIndices(labels, 2, 42L);

        assertThat(sampled).hasSize(labels.length);

        int count0 = 0;
        int count1 = 0;
        for (int idx : sampled) {
            if (labels[idx] == 0) count0++;
            else count1++;
        }

        assertThat(count1).isGreaterThanOrEqualTo(4);

        float[] capped = BalancedSampler.computeCappedWeights(labels, 2, 2.0f);
        assertThat(capped[1]).isLessThanOrEqualTo(3.0f);
    }

    @Test
    @DisplayName("BalancedSampler generateSmoothedIndices produces valid smoothed distribution")
    void testSmoothedSampler() {
        int[] labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        int[] sampled = BalancedSampler.generateSmoothedIndices(labels, 2, 0.5, 42L);

        assertThat(sampled).hasSize(labels.length);

        int count1 = 0;
        for (int idx : sampled) {
            if (labels[idx] == 1) count1++;
        }
        assertThat(count1).isBetween(1, 5);
    }
}
