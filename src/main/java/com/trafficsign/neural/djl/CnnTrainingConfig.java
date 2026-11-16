package com.trafficsign.neural.djl;

import ai.djl.Device;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Strongly-typed configuration record for CNN training runs.
 */
public record CnnTrainingConfig(
        int batchSize,
        int epochs,
        float learningRate,
        String optimizerName,
        long randomSeed,
        Path checkpointsDir,
        String modelName,
        Device device
) {
    public static final int DEFAULT_BATCH_SIZE = 64;
    public static final int DEFAULT_EPOCHS = 10;
    public static final float DEFAULT_LEARNING_RATE = 0.001f;

    public CnnTrainingConfig {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive: " + batchSize);
        }
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epoch count must be positive: " + epochs);
        }
        if (learningRate <= 0.0f || Float.isNaN(learningRate)) {
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        }
        Objects.requireNonNull(optimizerName, "optimizerName must not be null");
        Objects.requireNonNull(checkpointsDir, "checkpointsDir must not be null");
        Objects.requireNonNull(modelName, "modelName must not be null");
        Objects.requireNonNull(device, "device must not be null");
    }

    public static CnnTrainingConfig defaultSmoke() {
        return new CnnTrainingConfig(
                DEFAULT_BATCH_SIZE,
                DEFAULT_EPOCHS,
                DEFAULT_LEARNING_RATE,
                "adam",
                42L,
                Path.of("models", "checkpoints"),
                "gtsrb_cnn",
                Device.cpu()
        );
    }
}
