package com.trafficsign.neural.djl;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;

/**
 * Minimal lightweight CNN architecture designed for rapid unit tests and synthetic data smoke tests.
 */
public final class TinyCnnBlock {

    private TinyCnnBlock() {
    }

    public static Block build(int numClasses) {
        SequentialBlock block = new SequentialBlock();
        // Conv 3 -> 8 channels, 3x3 kernel, padding 1
        block.add(Conv2d.builder()
                .setFilters(8)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build());
        block.add(Activation.reluBlock());
        // MaxPool 2x2 with stride 2 -> 16x16
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        // Flatten 8 * 16 * 16 = 2048 features
        block.add(ai.djl.nn.Blocks.batchFlattenBlock());
        // Linear classifier to output classes
        block.add(Linear.builder().setUnits(numClasses).build());
        return block;
    }
}
