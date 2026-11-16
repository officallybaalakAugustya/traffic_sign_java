package com.trafficsign.neural.djl;

import ai.djl.Device;
import ai.djl.engine.Engine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Diagnostic utility for Deep Java Library (DJL) and GPU/CUDA verification.
 * Tests engine initialization, GPU count, device selection, and safe CPU fallback.
 */
public class GpuVerifier {

    private static final Logger logger = LoggerFactory.getLogger(GpuVerifier.class);

    public record GpuStatus(
            String engineName,
            String engineVersion,
            boolean hasGpu,
            int gpuCount,
            String selectedDevice,
            boolean isCudaAvailable,
            String diagnosticMessage
    ) {
    }

    public static GpuStatus verify() {
        try {
            Engine engine = Engine.getInstance();
            String engineName = engine.getEngineName();
            String engineVersion = engine.getVersion();
            int gpuCount = engine.getGpuCount();
            boolean hasGpu = gpuCount > 0;

            Device device;
            boolean cudaAvailable = false;
            String diagnostic;

            if (hasGpu) {
                try {
                    device = Device.gpu(0);
                    cudaAvailable = true;
                    diagnostic = "CUDA GPU execution available: " + device + " (Total GPUs detected: " + gpuCount + ")";
                    logger.info(diagnostic);
                } catch (Exception e) {
                    device = Device.cpu();
                    diagnostic = "GPU detected by engine (" + gpuCount + ") but CUDA device initialization failed: "
                            + e.getMessage() + ". Falling back to CPU.";
                    logger.warn(diagnostic);
                }
            } else {
                device = Device.cpu();
                diagnostic = "No CUDA GPU detected or PyTorch CUDA native binaries not present for Windows. Using CPU fallback: " + device;
                logger.info(diagnostic);
            }

            return new GpuStatus(
                    engineName,
                    engineVersion,
                    hasGpu,
                    gpuCount,
                    device.toString(),
                    cudaAvailable,
                    diagnostic
            );
        } catch (Throwable t) {
            String errorMsg = "DJL Engine initialization failed: " + t.getMessage();
            logger.error(errorMsg, t);
            return new GpuStatus(
                    "Unavailable",
                    "N/A",
                    false,
                    0,
                    Device.cpu().toString(),
                    false,
                    errorMsg
            );
        }
    }

    public static void main(String[] args) {
        System.out.println("=== DJL GPU & Engine Verification ===");
        GpuStatus status = verify();
        System.out.println("Engine Name:       " + status.engineName());
        System.out.println("Engine Version:    " + status.engineVersion());
        System.out.println("Has GPU:           " + status.hasGpu());
        System.out.println("GPU Count:         " + status.gpuCount());
        System.out.println("CUDA Available:    " + status.isCudaAvailable());
        System.out.println("Selected Device:   " + status.selectedDevice());
        System.out.println("Diagnostic:        " + status.diagnosticMessage());
    }
}
