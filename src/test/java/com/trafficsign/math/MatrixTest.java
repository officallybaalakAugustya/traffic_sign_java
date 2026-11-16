package com.trafficsign.math;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MatrixTest {

    @Test
    @DisplayName("Should construct matrix and preserve defensive copy of input arrays")
    void shouldConstructWithDefensiveCopy() {
        double[][] raw = {{1.0, 2.0}, {3.0, 4.0}};
        Matrix m = new Matrix(raw);

        // Mutate original array
        raw[0][0] = 999.0;
        assertThat(m.get(0, 0)).isEqualTo(1.0);

        // Mutate returned array
        double[][] extracted = m.toArray();
        extracted[0][0] = 888.0;
        assertThat(m.get(0, 0)).isEqualTo(1.0);
    }

    @Test
    @DisplayName("Should validate dimensions on construction and throw on invalid sizes")
    void shouldThrowOnInvalidDimensions() {
        assertThatThrownBy(() -> new Matrix(0, 5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Matrix(3, -1))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Matrix(new double[][]{}))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Matrix(new double[][]{{1.0, 2.0}, {3.0}})) // Ragged array
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should correctly perform matrix addition and subtraction")
    void shouldAddAndSubtract() {
        Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Matrix b = new Matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

        Matrix sum = a.add(b);
        assertThat(sum.get(0, 0)).isEqualTo(6.0);
        assertThat(sum.get(1, 1)).isEqualTo(12.0);

        Matrix diff = b.subtract(a);
        assertThat(diff.get(0, 0)).isEqualTo(4.0);
        assertThat(diff.get(1, 1)).isEqualTo(4.0);
    }

    @Test
    @DisplayName("Should correctly multiply matrices and throw on dimension mismatch")
    void shouldMultiplyMatrices() {
        // (2x3) * (3x2) -> (2x2)
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        Matrix b = new Matrix(new double[][]{
                {7.0, 8.0},
                {9.0, 1.0},
                {2.0, 3.0}
        });

        Matrix c = a.multiply(b);
        assertThat(c.rows()).isEqualTo(2);
        assertThat(c.cols()).isEqualTo(2);
        // c[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        assertThat(c.get(0, 0)).isEqualTo(31.0);
        // c[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        assertThat(c.get(0, 1)).isEqualTo(19.0);
        // c[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        assertThat(c.get(1, 0)).isEqualTo(85.0);
        // c[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        assertThat(c.get(1, 1)).isEqualTo(55.0);

        // Mismatched dimensions (2x3) * (2x2) -> error
        Matrix invalidB = new Matrix(2, 2);
        assertThatThrownBy(() -> a.multiply(invalidB))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Incompatible dimensions");
    }

    @Test
    @DisplayName("Should correctly transpose matrix")
    void shouldTransposeMatrix() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        Matrix at = a.transpose();
        assertThat(at.rows()).isEqualTo(3);
        assertThat(at.cols()).isEqualTo(2);
        assertThat(at.get(0, 0)).isEqualTo(1.0);
        assertThat(at.get(0, 1)).isEqualTo(4.0);
        assertThat(at.get(2, 0)).isEqualTo(3.0);
        assertThat(at.get(2, 1)).isEqualTo(6.0);
    }

    @Test
    @DisplayName("Should compute vector dot product and throw on length mismatch")
    void shouldComputeDotProduct() {
        double[] u = {1.0, 3.0, -5.0};
        double[] v = {4.0, -2.0, -1.0};
        // 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3
        assertThat(Matrix.dot(u, v)).isEqualTo(3.0);

        double[] w = {1.0, 2.0};
        assertThatThrownBy(() -> Matrix.dot(u, w))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should sum matrix elements and column-wise sums")
    void shouldComputeSums() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });
        assertThat(a.sum()).isEqualTo(10.0);
        assertThat(a.sumCols()).containsExactly(4.0, 6.0);
    }
}
