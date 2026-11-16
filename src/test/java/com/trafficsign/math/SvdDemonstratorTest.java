package com.trafficsign.math;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SvdDemonstratorTest {

    @Test
    @DisplayName("Should perform SVD and reconstruct original matrix at full rank with near-zero error")
    void shouldPerformFullRankReconstruction() {
        Matrix input = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
        }); // 4 x 3 matrix, max rank is 3

        SvdDemonstrator.SvdResult fullResult = SvdDemonstrator.decomposeAndReconstruct(input, 3);

        assertThat(fullResult.u().rows()).isEqualTo(4);
        assertThat(fullResult.u().cols()).isEqualTo(3); // Thin SVD produces m x min(m, n) matrix
        assertThat(fullResult.v().rows()).isEqualTo(3);
        assertThat(fullResult.v().cols()).isEqualTo(3);
        assertThat(fullResult.singularValues()).hasSize(3);

        // Singular values must be sorted descending
        assertThat(fullResult.singularValues()[0]).isGreaterThanOrEqualTo(fullResult.singularValues()[1]);
        assertThat(fullResult.singularValues()[1]).isGreaterThanOrEqualTo(fullResult.singularValues()[2]);

        // Full rank reconstruction error should be near numerical precision (< 1e-10)
        assertThat(fullResult.frobeniusError()).isLessThan(1e-10);
        assertThat(fullResult.energyPreservedRatio()).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    @DisplayName("Lower rank approximations exhibit monotonically decreasing Frobenius error as rank increases")
    void shouldDemonstrateMonotonicErrorReduction() {
        Matrix input = new Matrix(new double[][]{
                {5.0, 2.0, 1.0, 0.0},
                {2.0, 4.0, 0.0, 1.0},
                {1.0, 0.0, 3.0, 2.0},
                {0.0, 1.0, 2.0, 5.0}
        });

        SvdDemonstrator.SvdResult rank1 = SvdDemonstrator.decomposeAndReconstruct(input, 1);
        SvdDemonstrator.SvdResult rank2 = SvdDemonstrator.decomposeAndReconstruct(input, 2);
        SvdDemonstrator.SvdResult rank3 = SvdDemonstrator.decomposeAndReconstruct(input, 3);
        SvdDemonstrator.SvdResult rank4 = SvdDemonstrator.decomposeAndReconstruct(input, 4);

        assertThat(rank1.frobeniusError()).isGreaterThan(rank2.frobeniusError());
        assertThat(rank2.frobeniusError()).isGreaterThan(rank3.frobeniusError());
        assertThat(rank3.frobeniusError()).isGreaterThan(rank4.frobeniusError());

        assertThat(rank1.energyPreservedRatio()).isLessThan(rank2.energyPreservedRatio());
        assertThat(rank4.energyPreservedRatio()).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    @DisplayName("Should throw when requested rank k is invalid")
    void shouldThrowOnInvalidRank() {
        Matrix input = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertThatThrownBy(() -> SvdDemonstrator.decomposeAndReconstruct(input, 0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> SvdDemonstrator.decomposeAndReconstruct(input, 3)) // max rank is 2
                .isInstanceOf(IllegalArgumentException.class);
    }
}
