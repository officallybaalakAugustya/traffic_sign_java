package com.trafficsign.math;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TensorTest {

    @Test
    @DisplayName("Should validate tensor shapes, ranks, and element count")
    void shouldValidateTensorShapes() {
        Tensor t = new Tensor(2, 3, 4);
        assertThat(t.rank()).isEqualTo(3);
        assertThat(t.shape()).containsExactly(2, 3, 4);
        assertThat(t.size()).isEqualTo(24);

        assertThatThrownBy(() -> new Tensor(2, -1, 4))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should correctly index, get, and set tensor elements")
    void shouldIndexAndSetElements() {
        Tensor t = new Tensor(2, 3);
        t.set(42.0, 1, 2);
        assertThat(t.get(1, 2)).isEqualTo(42.0);
        assertThat(t.get(0, 0)).isEqualTo(0.0);

        assertThatThrownBy(() -> t.get(2, 0)) // Index out of bounds
                .isInstanceOf(IndexOutOfBoundsException.class);
        assertThatThrownBy(() -> t.get(0)) // Rank mismatch
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should correctly reshape tensor with identical element count")
    void shouldReshapeTensor() {
        Tensor t = new Tensor(new int[]{2, 3}, new double[]{1, 2, 3, 4, 5, 6});
        Tensor reshaped = t.reshape(3, 2);

        assertThat(reshaped.shape()).containsExactly(3, 2);
        assertThat(reshaped.get(0, 0)).isEqualTo(1.0);
        assertThat(reshaped.get(2, 1)).isEqualTo(6.0);

        assertThatThrownBy(() -> t.reshape(4, 2)) // 8 != 6
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should support explicit bias broadcasting over batch dimension")
    void shouldPerformSupportedBiasBroadcasting() {
        // [Batch=2, Features=3] + [Features=3]
        Tensor batch = new Tensor(new int[]{2, 3}, new double[]{
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
        });
        Tensor bias = new Tensor(new int[]{3}, new double[]{1.0, 2.0, 3.0});

        Tensor result = batch.add(bias);
        assertThat(result.shape()).containsExactly(2, 3);
        assertThat(result.get(0, 0)).isEqualTo(11.0);
        assertThat(result.get(0, 2)).isEqualTo(33.0);
        assertThat(result.get(1, 0)).isEqualTo(41.0);
        assertThat(result.get(1, 2)).isEqualTo(63.0);
    }

    @Test
    @DisplayName("Should throw on unsupported tensor broadcasting shapes")
    void shouldThrowOnUnsupportedBroadcasting() {
        Tensor a = new Tensor(2, 3, 4);
        Tensor b = new Tensor(2, 3);

        assertThatThrownBy(() -> a.add(b))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unsupported broadcasting");
    }

    @Test
    @DisplayName("Should slice batch elements correctly")
    void shouldSliceBatch() {
        Tensor t = new Tensor(new int[]{4, 2}, new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8
        });

        Tensor slice = t.sliceBatch(1, 3); // samples 1 and 2
        assertThat(slice.shape()).containsExactly(2, 2);
        assertThat(slice.get(0, 0)).isEqualTo(3.0);
        assertThat(slice.get(1, 1)).isEqualTo(6.0);
    }
}
