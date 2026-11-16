package com.trafficsign.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class MajorityClassBaselineTest {

    @Test
    @DisplayName("MajorityClassBaseline correctly selects most frequent training class and calculates metrics")
    void testMajorityClassBaseline() {
        int[] train = {1, 1, 1, 0, 2};
        int[] val = {1, 1, 0, 2};

        MajorityClassBaseline.BaselineResult res = MajorityClassBaseline.evaluate(train, val, 3);
        assertThat(res.majorityClassId()).isEqualTo(1);
        assertThat(res.accuracy()).isEqualTo(0.5);
    }
}
