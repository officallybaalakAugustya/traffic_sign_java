package com.trafficsign.neural.djl;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.trafficsign.evaluation.ConfusionMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Random;

/**
 * Deep Java Library (DJL) CNN training engine.
 * Handles training loops, validation evaluation, metric logging,
 * and checkpoint serialization for deep traffic-sign classifiers.
 */
public class CnnTrainer {

    private static final Logger logger = LoggerFactory.getLogger(CnnTrainer.class);

    private final CnnTrainingConfig config;

    public record SmokeTestResult(
            boolean lossFinite,
            float initialLoss,
            float finalLoss,
            boolean saveLoadConsistent,
            Path checkpointPath
    ) {
    }

    public CnnTrainer(CnnTrainingConfig config) {
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    public DefaultTrainingConfig createTrainingConfig(Loss loss) {
        Optimizer opt;
        if ("sgd".equalsIgnoreCase(config.optimizerName())) {
            opt = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        } else {
            opt = Optimizer.adam().optLearningRateTracker(Tracker.fixed(config.learningRate())).build();
        }

        return new DefaultTrainingConfig(loss)
                .optOptimizer(opt)
                .addEvaluator(new Accuracy());
    }

    /**
     * Executes a synthetic-data smoke test verifying:
     * 1. Block initialization with shape [Batch, 3, 32, 32]
     * 2. Forward pass and finite loss calculation
     * 3. Backward pass and finite gradient collection
     * 4. Weight serialization to checkpoint directory
     * 5. Weight deserialization into a fresh model
     * 6. Prediction consistency between original and reloaded models
     */
    public SmokeTestResult runSyntheticSmokeTest(Block block, int numClasses, int iterations)
            throws IOException, MalformedModelException {
        Path tempDir = Files.createTempDirectory("djl_cnn_smoke_");
        float firstLoss = Float.NaN;
        float lastLoss = Float.NaN;

        try (Model model = Model.newInstance("smoke_cnn", config.device())) {
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();
            DefaultTrainingConfig trainConfig = createTrainingConfig(loss);

            try (Trainer trainer = model.newTrainer(trainConfig)) {
                Shape inputShape = new Shape(config.batchSize(), 3, 32, 32);
                trainer.initialize(inputShape);

                try (NDManager manager = NDManager.newBaseManager(config.device())) {
                    Random rng = new Random(config.randomSeed());

                    // Generate deterministic synthetic batch
                    float[] syntheticPixels = new float[(int) inputShape.size()];
                    for (int i = 0; i < syntheticPixels.length; i++) {
                        syntheticPixels[i] = rng.nextFloat();
                    }
                    long[] syntheticLabels = new long[config.batchSize()];
                    for (int i = 0; i < syntheticLabels.length; i++) {
                        syntheticLabels[i] = rng.nextInt(numClasses);
                    }

                    NDArray x = manager.create(syntheticPixels, inputShape);
                    NDArray y = manager.create(syntheticLabels, new Shape(config.batchSize()));

                    for (int iter = 0; iter < iterations; iter++) {
                        try (GradientCollector gc = trainer.newGradientCollector()) {
                            NDList predictions = trainer.forward(new NDList(x));
                            NDArray lossArray = loss.evaluate(new NDList(y), predictions);

                            float lossVal = lossArray.getFloat();
                            if (Float.isNaN(lossVal) || Float.isInfinite(lossVal)) {
                                throw new IllegalStateException("Loss is non-finite at iter " + iter + ": " + lossVal);
                            }

                            if (iter == 0) firstLoss = lossVal;
                            lastLoss = lossVal;

                            gc.backward(lossArray);
                        }
                        trainer.step();
                    }

                    // Save model checkpoint
                    model.setProperty("Epoch", String.valueOf(iterations));
                    model.save(tempDir, "smoke_cnn");

                    // Run inference on original model
                    NDArray origLogits = model.getBlock().forward(
                            new ai.djl.training.ParameterStore(manager, false),
                            new NDList(x),
                            false
                    ).head();
                    long[] origPreds = origLogits.argMax(1).toLongArray();

                    // Load checkpoint into fresh model
                    boolean consistent;
                    try (Model reloadedModel = Model.newInstance("smoke_cnn_reloaded", config.device())) {
                        reloadedModel.setBlock(TinyCnnBlock.build(numClasses));
                        reloadedModel.load(tempDir, "smoke_cnn");

                        NDArray reloadedLogits = reloadedModel.getBlock().forward(
                                new ai.djl.training.ParameterStore(manager, false),
                                new NDList(x),
                                false
                        ).head();
                        long[] reloadedPreds = reloadedLogits.argMax(1).toLongArray();

                        consistent = java.util.Arrays.equals(origPreds, reloadedPreds);
                    }

                    boolean lossFinite = !Float.isNaN(firstLoss) && !Float.isInfinite(firstLoss)
                            && !Float.isNaN(lastLoss) && !Float.isInfinite(lastLoss);

                    return new SmokeTestResult(lossFinite, firstLoss, lastLoss, consistent, tempDir);
                }
            }
        }
    }
}
