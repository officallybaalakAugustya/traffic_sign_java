package com.trafficsign.safety;

import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class SafetyEventEngineTest {

    private SafetyEventEngine engine;

    @BeforeEach
    void setUp(@TempDir Path tempDir) {
        engine = new SafetyEventEngine();
        engine.setEvidenceDirectory(tempDir.resolve("evidence"));
    }

    @Test
    void testSingleDetectionGeneratesTrackAndEvent() {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox normBox = BoundingBox.ofNormalized(0.2, 0.2, 0.4, 0.4, 51, "SPEED_LIMIT_40");
        BoundingBox absBox = normBox.toAbsolute(640, 480);

        RoadSceneInferencePipeline.RoadSceneDetection det = new RoadSceneInferencePipeline.RoadSceneDetection(
                absBox,
                normBox,
                0.85f,
                51,
                "SPEED_LIMIT_40",
                "MANDATORY",
                "Speed limit 40 km/h",
                0.92f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        RoadSceneInferencePipeline.PipelineResult result = new RoadSceneInferencePipeline.PipelineResult(
                640, 480, 50, 20, List.of(det)
        );

        List<SafetyEvent> events = engine.processFrame(1, 1000L, frame, result);
        assertThat(events).hasSize(1);
        SafetyEvent evt = events.get(0);
        assertThat(evt.classId()).isEqualTo(51);
        assertThat(evt.canonicalLabel()).isEqualTo("SPEED_LIMIT_40");
        assertThat(evt.numericalValue()).isEqualTo(40.0);
        assertThat(evt.severity()).isEqualTo(SignRuleCatalogue.SeverityLevel.WARNING);
        assertThat(evt.eventType()).isEqualTo(SignRuleCatalogue.EventType.SPEED_LIMIT_POSTED);

        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(40);
        assertThat(engine.getActiveTracks()).hasSize(1);
        assertThat(engine.getEventLog()).hasSize(1);
    }

    @Test
    void testConsecutiveFramesSuppressDuplicateEvents() {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox normBox1 = BoundingBox.ofNormalized(0.2, 0.2, 0.4, 0.4, 59, "STOP");
        BoundingBox absBox1 = normBox1.toAbsolute(640, 480);
        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                absBox1, normBox1, 0.88f, 59, "STOP", "MANDATORY", "Stop", 0.95f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        RoadSceneInferencePipeline.PipelineResult result1 = new RoadSceneInferencePipeline.PipelineResult(
                640, 480, 50, 20, List.of(det1)
        );

        List<SafetyEvent> frame1Events = engine.processFrame(1, 1000L, frame, result1);
        assertThat(frame1Events).hasSize(1);
        assertThat(engine.getEventLog()).hasSize(1);

        BoundingBox normBox2 = BoundingBox.ofNormalized(0.21, 0.21, 0.41, 0.41, 59, "STOP");
        BoundingBox absBox2 = normBox2.toAbsolute(640, 480);
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                absBox2, normBox2, 0.89f, 59, "STOP", "MANDATORY", "Stop", 0.96f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        RoadSceneInferencePipeline.PipelineResult result2 = new RoadSceneInferencePipeline.PipelineResult(
                640, 480, 50, 20, List.of(det2)
        );

        List<SafetyEvent> frame2Events = engine.processFrame(2, 1033L, frame, result2);
        assertThat(frame2Events).isEmpty();
        assertThat(engine.getEventLog()).hasSize(1);
        assertThat(engine.getActiveTracks().get(0).getObservationCount()).isEqualTo(2);
    }

    @Test
    void testDistinctSignsCreateSeparateTracksAndEvents() {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox normBox1 = BoundingBox.ofNormalized(0.1, 0.1, 0.25, 0.25, 59, "STOP");
        BoundingBox normBox2 = BoundingBox.ofNormalized(0.7, 0.7, 0.85, 0.85, 44, "SCHOOL_AHEAD");

        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                normBox1.toAbsolute(640, 480), normBox1, 0.90f, 59, "STOP", "MANDATORY", "Stop", 0.95f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                normBox2.toAbsolute(640, 480), normBox2, 0.85f, 44, "SCHOOL_AHEAD", "CAUTIONARY", "School", 0.90f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        RoadSceneInferencePipeline.PipelineResult result = new RoadSceneInferencePipeline.PipelineResult(
                640, 480, 50, 40, List.of(det1, det2)
        );

        List<SafetyEvent> events = engine.processFrame(1, 1000L, frame, result);
        assertThat(events).hasSize(2);
        assertThat(engine.getActiveTracks()).hasSize(2);
        assertThat(engine.getEventLog()).hasSize(2);
    }

    @Test
    void testStateUpdatesRestrictionsAndSpeedLimit() {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox box1 = BoundingBox.ofNormalized(0.2, 0.2, 0.3, 0.3, 34, "OVERTAKING_PROHIBITED");
        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box1.toAbsolute(640, 480), box1, 0.88f, 34, "OVERTAKING_PROHIBITED", "MANDATORY", "No overtaking", 0.92f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        engine.processFrame(1, 1000L, frame, new RoadSceneInferencePipeline.PipelineResult(640, 480, 40, 20, List.of(det1)));

        assertThat(engine.getSafetyState().getActiveRestrictions()).contains("Overtaking Prohibited");

        BoundingBox box2 = BoundingBox.ofNormalized(0.5, 0.5, 0.6, 0.6, 40, "RESTRICTION_ENDS");
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box2.toAbsolute(640, 480), box2, 0.82f, 40, "RESTRICTION_ENDS", "MANDATORY", "End restrictions", 0.85f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        engine.processFrame(2, 2000L, frame, new RoadSceneInferencePipeline.PipelineResult(640, 480, 40, 20, List.of(det2)));

        assertThat(engine.getSafetyState().getActiveRestrictions()).isEmpty();
    }

    @Test
    void testOperatorReviewUpdate() {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox box = BoundingBox.ofNormalized(0.2, 0.2, 0.3, 0.3, 50, "SPEED_LIMIT_30");
        RoadSceneInferencePipeline.RoadSceneDetection det = new RoadSceneInferencePipeline.RoadSceneDetection(
                box.toAbsolute(640, 480), box, 0.80f, 50, "SPEED_LIMIT_30", "MANDATORY", "Speed limit 30", 0.85f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        List<SafetyEvent> events = engine.processFrame(1, 1000L, frame, new RoadSceneInferencePipeline.PipelineResult(640, 480, 40, 20, List.of(det)));
        String eventId = events.get(0).eventId();

        boolean updated = engine.updateOperatorReview(eventId, SafetyEvent.OperatorStatus.CONFIRMED, "Verified by operator in urban zone.");
        assertThat(updated).isTrue();

        SafetyEvent reviewedEvt = engine.getEventLog().get(0);
        assertThat(reviewedEvt.operatorStatus()).isEqualTo(SafetyEvent.OperatorStatus.CONFIRMED);
        assertThat(reviewedEvt.operatorNote()).isEqualTo("Verified by operator in urban zone.");
    }

    @Test
    void testExportCsvAndJson(@TempDir Path tempDir) throws IOException {
        BufferedImage frame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        BoundingBox box = BoundingBox.ofNormalized(0.2, 0.2, 0.3, 0.3, 50, "SPEED_LIMIT_30");
        RoadSceneInferencePipeline.RoadSceneDetection det = new RoadSceneInferencePipeline.RoadSceneDetection(
                box.toAbsolute(640, 480), box, 0.80f, 50, "SPEED_LIMIT_30", "MANDATORY", "Speed limit 30", 0.85f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        engine.processFrame(1, 1000L, frame, new RoadSceneInferencePipeline.PipelineResult(640, 480, 40, 20, List.of(det)));

        Path csvPath = tempDir.resolve("events.csv");
        Path jsonPath = tempDir.resolve("events.json");

        engine.exportEventsToCsv(csvPath);
        engine.exportEventsToJson(jsonPath);

        assertThat(Files.exists(csvPath)).isTrue();
        assertThat(Files.exists(jsonPath)).isTrue();

        List<String> csvLines = Files.readAllLines(csvPath);
        assertThat(csvLines.size()).isGreaterThanOrEqualTo(2);
        assertThat(csvLines.get(0)).contains("eventId,frameIndex,timestampMs");
        assertThat(csvLines.get(1)).contains("SPEED_LIMIT_30");

        String jsonContent = Files.readString(jsonPath);
        assertThat(jsonContent).contains("SPEED_LIMIT_30");
        assertThat(jsonContent).contains("Speed Limit 30 km/h");
    }
}
