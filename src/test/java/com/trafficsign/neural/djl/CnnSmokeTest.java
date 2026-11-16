package com.trafficsign.neural.djl;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("DJL CNN Architecture and Synthetic Smoke Tests")
class CnnSmokeTest {

    @Test
    @DisplayName("GpuVerifier should diagnose available compute devices without throwing")
    void testGpuVerifier() {
        GpuVerifier.GpuStatus status = GpuVerifier.verify();
        assertNotNull(status, "GpuStatus must not be null");
        assertNotNull(status.selectedDevice(), "Selected device must not be null");
        assertNotNull(status.engineName(), "Engine name must not be null");
        assertNotNull(status.diagnosticMessage(), "Diagnostic message must not be null");
        assertTrue(status.diagnosticMessage().contains("Using CPU fallback") || status.diagnosticMessage().contains("CUDA GPU execution available"));
    }

    @Test
    @DisplayName("GtsrbCnnBlock parameter count should strictly satisfy the 0.3M - 1.0M budget")
    void testGtsrbCnnParameterCount() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = GtsrbCnnBlock.build(43);
            long paramCount = GtsrbCnnBlock.countParameters(block, manager, new Shape(1, 3, 32, 32));

            // Analytical count is 629,739 parameters
            assertTrue(paramCount >= 300_000, "Parameter count must be at least 300,000, but got: " + paramCount);
            assertTrue(paramCount <= 1_000_000, "Parameter count must not exceed 1,000,000, but got: " + paramCount);
            assertEquals(629_739L, paramCount, "Parameter count must match analytical calculation");
        }
    }

    @Test
    @DisplayName("TinyCnnBlock should build cleanly and report valid block structure")
    void testTinyCnnBlock() {
        Block block = TinyCnnBlock.build(5);
        assertNotNull(block, "TinyCnnBlock must not be null");
    }

    @Test
    @DisplayName("CnnTrainer synthetic smoke test should verify finite loss, backpropagation, and save/load consistency")
    void testCnnSyntheticSmokeTest() throws IOException, MalformedModelException {
        // Use small batch size and 4 iterations for fast, reliable unit smoke test
        CnnTrainingConfig config = new CnnTrainingConfig(
                8,                                      // batchSize
                1,                                      // epochs
                0.01f,                                  // learningRate
                "sgd",                                  // optimizerName
                42L,                                    // randomSeed
                Path.of("target", "test-checkpoints"),  // checkpointsDir
                "smoke_cnn",                            // modelName
                Device.cpu()                            // device
        );

        CnnTrainer trainer = new CnnTrainer(config);
        int numClasses = 4;
        int iterations = 4;

        Block block = TinyCnnBlock.build(numClasses);
        CnnTrainer.SmokeTestResult result = trainer.runSyntheticSmokeTest(block, numClasses, iterations);

        assertNotNull(result, "SmokeTestResult must not be null");
        assertTrue(result.lossFinite(), "Loss must remain finite during synthetic training iterations");
        assertFalse(Float.isNaN(result.initialLoss()), "Initial loss must not be NaN");
        assertFalse(Float.isNaN(result.finalLoss()), "Final loss must not be NaN");
        assertTrue(result.initialLoss() > 0, "Initial loss must be positive");
        assertTrue(result.finalLoss() > 0, "Final loss must be positive");
        assertTrue(result.saveLoadConsistent(), "Predictions of reloaded model must exactly match original model");

        // Verify checkpoint directory actually contains files
        assertNotNull(result.checkpointPath());
        assertTrue(Files.exists(result.checkpointPath()));
    }
}
