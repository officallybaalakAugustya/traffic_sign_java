package com.trafficsign.neural.djl;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class HighResSignDetectorBlockTest {

    @Test
    @DisplayName("HighResSignDetectorBlock builds and produces 5x16x16 grid tensor")
    void testBlockOutputShape() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Block block = HighResSignDetectorBlock.build();
            long params = HighResSignDetectorBlock.countParameters(block, manager, new Shape(1, 3, 128, 128));

            assertThat(params).isGreaterThan(200_000L).isLessThan(300_000L);

            ParameterStore ps = new ParameterStore(manager, false);
            NDArray x = manager.ones(new Shape(1, 3, 128, 128));
            NDArray out = block.forward(ps, new NDList(x), false).head();

            assertThat(out.getShape()).isEqualTo(new Shape(1, 5, 16, 16));
        }
    }
}
