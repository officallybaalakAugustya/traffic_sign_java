package com.trafficsign.ui;

import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.safety.SafetyEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class RoadSafetyMonitorAppIntegrationTest {

    private RoadSafetyMonitorApp app;

    @BeforeEach
    void setUp() {
        app = new RoadSafetyMonitorApp();
    }

    @Test
    void testProcessSingleFramePopulatesTableAndHud(@TempDir Path tempDir) throws IOException {
        app.getSafetyEngine().setEvidenceDirectory(tempDir.resolve("evidence"));

        BufferedImage frame = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        BoundingBox normBox1 = BoundingBox.ofNormalized(0.15, 0.20, 0.35, 0.40, 51, "SPEED_LIMIT_40");
        BoundingBox absBox1 = normBox1.toAbsolute(800, 600);

        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                absBox1, normBox1, 0.82f, 51, "SPEED_LIMIT_40", "MANDATORY", "Speed limit 40 km/h", 0.89f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        BoundingBox normBox2 = BoundingBox.ofNormalized(0.60, 0.20, 0.80, 0.40, 59, "STOP");
        BoundingBox absBox2 = normBox2.toAbsolute(800, 600);
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                absBox2, normBox2, 0.88f, 59, "STOP", "MANDATORY", "Stop", 0.94f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        RoadSceneInferencePipeline.PipelineResult result = new RoadSceneInferencePipeline.PipelineResult(
                800, 600, 45, 30, List.of(det1, det2)
        );

        app.getSafetyEngine().processFrame(1, 1000L, frame, result);
        app.getTableModel().setEvents(app.getSafetyEngine().getEventLog());

        assertThat(app.getTableModel().getRowCount()).isEqualTo(2);

        SafetyEvent event0 = app.getTableModel().getEventAt(0);
        assertThat(event0).isNotNull();
        assertThat(event0.classId()).isEqualTo(51);

        SafetyEvent event1 = app.getTableModel().getEventAt(1);
        assertThat(event1).isNotNull();
        assertThat(event1.classId()).isEqualTo(59);

        Path csvPath = tempDir.resolve("test_export.csv");
        Path jsonPath = tempDir.resolve("test_export.json");
        app.getSafetyEngine().exportEventsToCsv(csvPath);
        app.getSafetyEngine().exportEventsToJson(jsonPath);

        assertThat(Files.exists(csvPath)).isTrue();
        assertThat(Files.exists(jsonPath)).isTrue();
        assertThat(Files.readAllLines(csvPath).size()).isEqualTo(3);

        if (!GraphicsEnvironment.isHeadless()) {
            BufferedImage snapshot = app.captureUiSnapshot(1280, 800);
            assertThat(snapshot).isNotNull();
            assertThat(snapshot.getWidth()).isEqualTo(1280);
            assertThat(snapshot.getHeight()).isEqualTo(800);
            Path snapPath = tempDir.resolve("ui_snapshot.png");
            ImageIO.write(snapshot, "png", snapPath.toFile());
            assertThat(Files.exists(snapPath)).isTrue();
        }
    }
}
