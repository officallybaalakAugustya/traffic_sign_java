package com.trafficsign.neural.djl;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;

public final class IndianTrafficSignCnnBlock {

    public static final int DEFAULT_NUM_CLASSES = 85;

    private IndianTrafficSignCnnBlock() {
    }

    public static Block build(int numClasses) {
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

        block.add(Blocks.batchFlattenBlock());
        block.add(Dropout.builder().optRate(0.4f).build());
        block.add(Linear.builder().setUnits(256).build());
        block.add(Activation.reluBlock());
        block.add(Dropout.builder().optRate(0.3f).build());
        block.add(Linear.builder().setUnits(numClasses).build());

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
