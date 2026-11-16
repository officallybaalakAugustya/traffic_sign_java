package com.trafficsign.neural.djl;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;

public final class CompactSignDetectorBlock {

    public static final int GRID_ROWS = 8;
    public static final int GRID_COLS = 8;
    public static final int CHANNELS_PER_CELL = 5;

    private CompactSignDetectorBlock() {
    }

    public static Block build() {
        SequentialBlock block = new SequentialBlock();

        block.add(Conv2d.builder()
                .setFilters(32)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation.reluBlock());
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Conv2d.builder()
                .setFilters(64)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation.reluBlock());
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Conv2d.builder()
                .setFilters(128)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation.reluBlock());
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Conv2d.builder()
                .setFilters(128)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation.reluBlock());
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Conv2d.builder()
                .setFilters(CHANNELS_PER_CELL)
                .setKernelShape(new Shape(1, 1))
                .build());

        return block;
    }

    public static long countParameters(Block block, NDManager manager, Shape inputShape) {
        if (!block.isInitialized()) {
            block.initialize(manager, DataType.FLOAT32, inputShape);
        }

        ParameterStore ps = new ParameterStore(manager, false);
        long total = 0;
        for (var entry : block.getParameters()) {
            Shape s = entry.getValue().getArray().getShape();
            long elements = 1;
            for (int i = 0; i < s.dimension(); i++) {
                elements *= s.get(i);
            }
            total += elements;
        }
        return total;
    }
}
