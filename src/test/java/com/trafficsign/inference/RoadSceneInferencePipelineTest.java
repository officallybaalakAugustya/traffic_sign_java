package com.trafficsign.inference;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.data.IndianTrafficSignTaxonomy;
import com.trafficsign.neural.djl.CompactSignDetectorBlock;
import com.trafficsign.neural.djl.IndianTrafficSignCnnBlock;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class RoadSceneInferencePipelineTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("RoadSceneInferencePipeline loads detector and classifier and produces structured detections")
    void testEndToEndInferencePipeline() throws IOException, MalformedModelException {
        Path detDir = tempDir.resolve("detector");
        Path clsDir = tempDir.resolve("classifier");
        Files.createDirectories(detDir);
        Files.createDirectories(clsDir);

        createMockModels(detDir, clsDir, "sign_detector", "indian_cnn", 32);

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {
            BufferedImage testImg = new BufferedImage(1920, 994, BufferedImage.TYPE_INT_RGB);

            RoadSceneInferencePipeline.PipelineResult result = pipeline.processImage(testImg);

            assertThat(result.imageWidth()).isEqualTo(1920);
            assertThat(result.imageHeight()).isEqualTo(994);
            assertThat(result.detectionLatencyMs()).isGreaterThanOrEqualTo(0);
            assertThat(result.classificationLatencyMs()).isGreaterThanOrEqualTo(0);
            assertThat(result.totalLatencyMs()).isGreaterThanOrEqualTo(0);
            assertThat(result.throughputFps()).isGreaterThanOrEqualTo(0.0);
        }
    }

    @Test
    @DisplayName("RoadSceneInferencePipeline correctly operates with 48x48 resolution champion")
    void test48x48Pipeline() throws IOException, MalformedModelException {
        Path detDir = tempDir.resolve("detector48");
        Path clsDir = tempDir.resolve("classifier48");
        Files.createDirectories(detDir);
        Files.createDirectories(clsDir);

        createMockModels(detDir, clsDir, "sign_detector", "exp_b_48x48", 48);

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(
                detDir, "sign_detector", clsDir, "exp_b_48x48", 48, Device.cpu()
        )) {
            assertThat(pipeline.getClassifierResolution()).isEqualTo(48);

            BufferedImage testImg = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
            var res = pipeline.processImage(testImg);

            assertThat(res.imageWidth()).isEqualTo(640);
            assertThat(res.imageHeight()).isEqualTo(480);
        }
    }

    @Test
    @DisplayName("RoadSceneInferencePipeline safely handles empty and degenerate crops")
    void testDegenerateCropSafeHandling() throws IOException, MalformedModelException {
        Path detDir = tempDir.resolve("detector_crop");
        Path clsDir = tempDir.resolve("classifier_crop");
        Files.createDirectories(detDir);
        Files.createDirectories(clsDir);

        createMockModels(detDir, clsDir, "sign_detector", "indian_cnn", 32);

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {
            BufferedImage smallImg = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
            var res = pipeline.processImage(smallImg);

            assertThat(res.imageWidth()).isEqualTo(4);
            assertThat(res.imageHeight()).isEqualTo(4);
        }
    }

    @Test
    @DisplayName("RoadSceneInferencePipeline drawDetections preserves original frame and overlays annotations")
    void testDrawDetectionsPreservesOriginal() throws IOException, MalformedModelException {
        Path detDir = tempDir.resolve("detector_draw");
        Path clsDir = tempDir.resolve("classifier_draw");
        Files.createDirectories(detDir);
        Files.createDirectories(clsDir);

        createMockModels(detDir, clsDir, "sign_detector", "indian_cnn", 32);

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {
            BufferedImage sourceImg = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
            sourceImg.setRGB(50, 50, 0x112233);

            RoadSceneInferencePipeline.RoadSceneDetection mockDet = new RoadSceneInferencePipeline.RoadSceneDetection(
                    BoundingBox.ofPixels(10, 10, 40, 40, 0, "traffic sign"),
                    BoundingBox.ofNormalized(0.1, 0.1, 0.4, 0.4, 0, "traffic sign"),
                    0.85f,
                    51,
                    "Speed Limit (40 km/h)",
                    "MANDATORY",
                    "Speed limit 40 km/h.",
                    0.92f,
                    RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
            );

            RoadSceneInferencePipeline.PipelineResult mockRes = new RoadSceneInferencePipeline.PipelineResult(
                    100, 100, 15, 10, List.of(mockDet)
            );

            BufferedImage annotated = pipeline.drawDetections(sourceImg, mockRes);

            assertThat(annotated).isNotSameAs(sourceImg);
            assertThat(sourceImg.getRGB(50, 50) & 0xFFFFFF).isEqualTo(0x112233);
            assertThat(annotated.getWidth()).isEqualTo(100);
            assertThat(annotated.getHeight()).isEqualTo(100);
        }
    }

    @Test
    @DisplayName("RoadSceneInferencePipeline supports threshold configuration and uncertainty categorization")
    void testThresholdConfiguration() throws IOException, MalformedModelException {
        Path detDir = tempDir.resolve("detector_thresh");
        Path clsDir = tempDir.resolve("classifier_thresh");
        Files.createDirectories(detDir);
        Files.createDirectories(clsDir);

        createMockModels(detDir, clsDir, "sign_detector", "indian_cnn", 32);

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {
            pipeline.setDetectionConfidenceThreshold(0.60f);
            pipeline.setClassificationConfidenceThreshold(0.70f);
            pipeline.setNmsIouThreshold(0.35);

            assertThat(pipeline.getDetectionConfidenceThreshold()).isEqualTo(0.60f);
            assertThat(pipeline.getClassificationConfidenceThreshold()).isEqualTo(0.70f);
            assertThat(pipeline.getNmsIouThreshold()).isEqualTo(0.35);
        }
    }

    @Test
    @DisplayName("RoadSceneInferencePipeline throws exception when checkpoint paths do not exist")
    void testMissingCheckpointHandling() {
        Path nonExistentDet = tempDir.resolve("no_det");
        Path nonExistentCls = tempDir.resolve("no_cls");

        assertThatThrownBy(() -> new RoadSceneInferencePipeline(nonExistentDet, nonExistentCls, Device.cpu()))
                .isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("IndianTrafficSignTaxonomy correctly maps all 85 classes")
    void testTaxonomyMapping() {
        IndianTrafficSignTaxonomy taxonomy = new IndianTrafficSignTaxonomy();
        assertThat(taxonomy.totalClasses()).isGreaterThanOrEqualTo(85);

        var info51 = taxonomy.getClassInfo(51);
        assertThat(info51).isPresent();
        assertThat(info51.get().canonicalName()).contains("40");
        assertThat(info51.get().speedLimitKmH()).isEqualTo(40);
        assertThat(info51.get().category()).isEqualTo("MANDATORY");
    }

    private void createMockModels(Path detDir, Path clsDir, String detName, String clsName, int clsRes) throws IOException {
        try (ai.djl.ndarray.NDManager manager = ai.djl.ndarray.NDManager.newBaseManager(Device.cpu())) {
            try (Model detModel = Model.newInstance(detName, Device.cpu())) {
                ai.djl.nn.Block b = CompactSignDetectorBlock.build();
                b.initialize(manager, ai.djl.ndarray.types.DataType.FLOAT32, new ai.djl.ndarray.types.Shape(1, 3, 128, 128));
                detModel.setBlock(b);
                detModel.save(detDir, detName);
            }

            try (Model clsModel = Model.newInstance(clsName, Device.cpu())) {
                ai.djl.nn.Block b = IndianTrafficSignCnnBlock.build(85);
                b.initialize(manager, ai.djl.ndarray.types.DataType.FLOAT32, new ai.djl.ndarray.types.Shape(1, 3, clsRes, clsRes));
                clsModel.setBlock(b);
                clsModel.save(clsDir, clsName);
            }
        }
    }
}
