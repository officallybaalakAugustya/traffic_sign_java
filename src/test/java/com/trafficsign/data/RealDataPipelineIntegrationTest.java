package com.trafficsign.data;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.training.loss.Loss;
import com.trafficsign.neural.djl.GtsrbCnnBlock;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Real-Data & Realistic Pipeline Integration Tests")
class RealDataPipelineIntegrationTest {

    private static final Path REAL_GTSRB_PATH = Path.of("data", "raw", "GTSRB", "Final_Training", "Images");

    @Test
    @DisplayName("Should execute end-to-end data pipeline: PPM load, crop, resize, split, and CNN forward pass")
    void testEndToEndPipelineVerification(@TempDir Path tempDir) throws IOException {
        Path datasetRoot;
        boolean isRealDataPresent = Files.isDirectory(REAL_GTSRB_PATH);

        if (isRealDataPresent) {
            System.out.println("[INFO] Real GTSRB dataset detected at " + REAL_GTSRB_PATH.toAbsolutePath());
            datasetRoot = REAL_GTSRB_PATH;
        } else {
            System.out.println("[INFO] Real GTSRB dataset not downloaded yet. Generating realistic mock GTSRB directory fixture.");
            datasetRoot = createRealisticMockDataset(tempDir);
        }

        // 1. Load dataset samples via GtsrbDatasetLoader
        List<ImageSample> samples = GtsrbDatasetLoader.loadTrainingDataset(datasetRoot);
        assertFalse(samples.isEmpty(), "Samples must not be empty");

        // Take a small configurable batch (up to 6 samples across distinct tracks)
        List<ImageSample> batchSamples = samples.subList(0, Math.min(6, samples.size()));

        // 2. Preprocess images into standardized [3, 32, 32] tensors
        ImagePreprocessor preprocessor = new ImagePreprocessor(32, 32, true);
        float[][][][] batchTensors = new float[batchSamples.size()][3][32][32];
        int[] batchLabels = new int[batchSamples.size()];

        for (int i = 0; i < batchSamples.size(); i++) {
            ImageSample s = batchSamples.get(i);
            batchLabels[i] = s.classId();
            assertTrue(s.classId() >= 0 && s.classId() < 43, "Class ID must be in [0, 42]");

            BufferedImage img = PpmImageReader.read(s.filePath());
            assertNotNull(img, "Decoded image must not be null");

            BufferedImage cropped = preprocessor.cropRoi(img, s.roiX1(), s.roiY1(), s.roiX2(), s.roiY2());
            float[][][] chw = preprocessor.toTensorCHW(cropped);
            assertEquals(3, chw.length, "Channel count must be 3 (RGB)");
            assertEquals(32, chw[0].length, "Height must be 32");
            assertEquals(32, chw[0][0].length, "Width must be 32");

            batchTensors[i] = chw;
        }

        // 3. Track-aware split and Leakage Detection
        DatasetSplitter splitter = new DatasetSplitter(42L, 0.50, 0.25, 0.25);
        SplitManifest manifest = splitter.split("GTSRB-Integration-Test", samples);
        assertNotNull(manifest);

        LeakageDetector leakageDetector = new LeakageDetector();
        LeakageDetector.LeakageReport leakageReport = leakageDetector.inspect(manifest);
        assertTrue(leakageReport.isClean(), "Splits must be strictly track-isolated without leakage: " + leakageReport.errors());

        // 4. CNN Forward Pass and Loss Computation using DJL GtsrbCnnBlock
        try (NDManager manager = NDManager.newBaseManager();
             Model model = Model.newInstance("gtsrb_cnn_test")) {

            Block block = GtsrbCnnBlock.build(43);
            model.setBlock(block);

            // Flatten batch tensor into 1D float array for NDArray creation
            float[] flatBatch = new float[batchSamples.size() * 3 * 32 * 32];
            int idx = 0;
            for (int b = 0; b < batchSamples.size(); b++) {
                for (int c = 0; c < 3; c++) {
                    for (int h = 0; h < 32; h++) {
                        for (int w = 0; w < 32; w++) {
                            flatBatch[idx++] = batchTensors[b][c][h][w];
                        }
                    }
                }
            }

            Shape inputShape = new Shape(batchSamples.size(), 3, 32, 32);
            block.initialize(manager, DataType.FLOAT32, inputShape);

            NDArray inputNd = manager.create(flatBatch, inputShape);
            NDArray targetLabels = manager.create(batchLabels, new Shape(batchSamples.size()));

            ParameterStore ps = new ParameterStore(manager, false);
            NDList predictions = block.forward(ps, new NDList(inputNd), false);
            NDArray logits = predictions.head();

            // Verify logits shape: [batchSize, 43]
            assertEquals(new Shape(batchSamples.size(), 43), logits.getShape(), "Logits shape must match [batchSize, 43]");

            Loss loss = Loss.softmaxCrossEntropyLoss();
            NDArray lossVal = loss.evaluate(new NDList(targetLabels), new NDList(logits));
            float scalarLoss = lossVal.getFloat();

            assertTrue(scalarLoss > 0, "Loss must be strictly positive");
            assertFalse(Float.isNaN(scalarLoss), "Loss must not be NaN");
            assertFalse(Float.isInfinite(scalarLoss), "Loss must not be Infinite");
        }
    }

    private Path createRealisticMockDataset(Path tempDir) throws IOException {
        Path imagesRoot = tempDir.resolve("Final_Training").resolve("Images");
        Files.createDirectories(imagesRoot);

        // Create 2 class folders: 00000 and 00001
        for (int c = 0; c < 2; c++) {
            String folderName = String.format("%05d", c);
            Path classDir = imagesRoot.resolve(folderName);
            Files.createDirectories(classDir);

            // Create 2 tracks per class, 2 frames per track
            List<String> csvLines = new ArrayList<>();
            csvLines.add("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId");

            for (int t = 0; t < 2; t++) {
                String trackId = String.format("%05d", (c * 10) + t);
                for (int f = 0; f < 2; f++) {
                    String fname = String.format("%s_%05d.ppm", trackId, f);
                    Path ppmFile = classDir.resolve(fname);
                    writeTestPpm(ppmFile, 36, 36, c == 0 ? 255 : 0, c == 1 ? 255 : 0, 128);
                    csvLines.add(String.format("%s;36;36;4;4;32;32;%d", fname, c));
                }
            }

            Path csvFile = classDir.resolve(String.format("GT-%05d.csv", c));
            Files.write(csvFile, csvLines);
        }

        return imagesRoot;
    }

    private void writeTestPpm(Path path, int width, int height, int r, int g, int b) throws IOException {
        try (OutputStream fos = Files.newOutputStream(path);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            String header = "P6\n" + width + " " + height + "\n255\n";
            bos.write(header.getBytes());
            byte[] pixel = new byte[]{(byte) r, (byte) g, (byte) b};
            for (int i = 0; i < width * height; i++) {
                bos.write(pixel);
            }
        }
    }
}
