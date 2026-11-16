package com.trafficsign.safety;

import ai.djl.Device;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.ui.RoadSafetyMonitorApp;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class Task17StateAndFalsePositiveTest {

    private static RoadSceneInferencePipeline pipeline;

    @BeforeAll
    static void setUp() throws Exception {
        Path detDir = Path.of("models/checkpoints/detector_task15_champion");
        Path clsDir = Path.of("models/checkpoints/task7_champion");
        pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu());
        pipeline.setDetectionConfidenceThreshold(0.20f);
        pipeline.setClassificationConfidenceThreshold(0.20f);
        pipeline.setNmsIouThreshold(0.25);
        pipeline.setMaxProposalsPerFrame(25);
    }

    @AfterAll
    static void tearDown() {
        if (pipeline != null) {
            pipeline.close();
        }
    }

    @Test
    void testClearSpeedLimit40SignDetectionAndState() throws Exception {
        File speed40File = new File("data/raw/traffic_sign_classification_dataset/test/SPEED_LIMIT_40/66000.jpg");
        assertThat(speed40File).exists();
        BufferedImage patch = ImageIO.read(speed40File);

        BufferedImage scene = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = scene.createGraphics();
        g.setColor(new java.awt.Color(50, 70, 90));
        g.fillRect(0, 0, 800, 300);
        g.setColor(new java.awt.Color(70, 70, 70));
        g.fillRect(0, 300, 800, 300);
        g.drawImage(patch, 400, 200, 100, 100, null);
        g.dispose();

        pipeline.setDetectionConfidenceThreshold(0.20f);
        pipeline.setClassificationConfidenceThreshold(0.20f);
        RoadSceneInferencePipeline.PipelineResult result = pipeline.processImage(scene);
        assertThat(result.detections()).isNotEmpty();

        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.20f);

        List<SafetyEvent> events = engine.processFrame(1, 1000L, scene, result);
        assertThat(events).isNotEmpty();

        boolean hasSpeed40 = events.stream().anyMatch(e ->
                e.eventType() == SignRuleCatalogue.EventType.SPEED_LIMIT_POSTED
                        && e.numericalValue() != null
                        && Math.round(e.numericalValue()) == 40
        );
        assertThat(hasSpeed40).isTrue();
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(40);
    }

    @Test
    void testEmptyRoadImageProducesNoAuthoritativeSpeedLimitOrHazard() throws Exception {
        File emptyRoadFile = new File("data/raw/indian-road-driving-sample/inspection/000db725-8180-4770-8b6a-4eb74aeda9f9_0004.jpg");
        assertThat(emptyRoadFile).exists();
        BufferedImage emptyRoadImg = ImageIO.read(emptyRoadFile);

        pipeline.setDetectionConfidenceThreshold(0.55f);
        pipeline.setClassificationConfidenceThreshold(0.35f);
        RoadSceneInferencePipeline.PipelineResult result = pipeline.processImage(emptyRoadImg);

        assertThat(result.detections()).isEmpty();

        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.55f);
        engine.setMinClassificationConfidence(0.35f);

        List<SafetyEvent> events = engine.processFrame(1, 1000L, emptyRoadImg, result);
        assertThat(events).isEmpty();

        SafetyState state = engine.getSafetyState();
        assertThat(state.getCurrentPostedSpeedLimit()).isNull();
        assertThat(state.getActiveRestrictions()).isEmpty();
        assertThat(state.getActiveHazards()).isEmpty();
    }

    @Test
    void testImageSwitchClearsStaleSpeedLimitAndHazards() {
        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.20f);

        SignRuleCatalogue catalogue = engine.getCatalogue();
        SignRuleCatalogue.SignRule speed40Rule = catalogue.getRule(51);

        BoundingBox box = BoundingBox.ofNormalized(0.4, 0.4, 0.5, 0.5, 51, "SPEED_LIMIT_40");
        SignObservation obs1 = new SignObservation(
                1, 1000L, box.toAbsolute(800, 600), box, 0.85f, 51, "Speed Limit (40 km/h)", 0.88f,
                speed40Rule, RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box.toAbsolute(800, 600), box, 0.85f, 51, "Speed Limit (40 km/h)", "REGULATORY", "Obey 40 km/h", 0.88f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        engine.processFrame(1, 1000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det1)));
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(40);

        engine.clear();
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isNull();
        assertThat(engine.getActiveTracks()).isEmpty();
        assertThat(engine.getEventLog()).isEmpty();

        BoundingBox emptyBox = BoundingBox.ofNormalized(0.1, 0.1, 0.2, 0.2, 11, "DANGEROUS_DIP");
        SignObservation obs2 = new SignObservation(
                2, 2000L, emptyBox.toAbsolute(800, 600), emptyBox, 0.35f, 11, "Dangerous Dip", 0.12f,
                catalogue.getRule(11), RoadSceneInferencePipeline.UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE
        );
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                emptyBox.toAbsolute(800, 600), emptyBox, 0.35f, 11, "Dangerous Dip", "CAUTIONARY", "Dip ahead", 0.12f,
                RoadSceneInferencePipeline.UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE
        );

        engine.processFrame(2, 2000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det2)));
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isNull();
        assertThat(engine.getSafetyState().getActiveHazards()).isEmpty();
    }

    @Test
    void testLowClassificationConfidenceDoesNotCorruptSafetyState() {
        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.20f);

        BoundingBox box = BoundingBox.ofNormalized(0.3, 0.3, 0.4, 0.4, 51, "SPEED_LIMIT_40");
        RoadSceneInferencePipeline.RoadSceneDetection det = new RoadSceneInferencePipeline.RoadSceneDetection(
                box.toAbsolute(800, 600), box, 0.85f, 51, "Speed Limit (40 km/h)", "REGULATORY", "Obey 40 km/h", 0.14f,
                RoadSceneInferencePipeline.UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE
        );

        List<SafetyEvent> events = engine.processFrame(1, 1000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det)));
        assertThat(events).hasSize(1);
        SafetyEvent event = events.get(0);
        assertThat(event.eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);
        assertThat(event.severity()).isEqualTo(SignRuleCatalogue.SeverityLevel.INFO);
        assertThat(event.numericalValue()).isNull();
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isNull();
    }

    @Test
    void testSliderThresholdSynchronization() {
        System.setProperty("java.awt.headless", "true");
        RoadSafetyMonitorApp app = new RoadSafetyMonitorApp();

        assertThat(app.getDetConfidenceSlider().getValue()).isEqualTo(20);
        assertThat(app.getClsConfidenceSlider().getValue()).isEqualTo(25);
        assertThat(app.getNmsIouSlider().getValue()).isEqualTo(25);
        assertThat(app.getMaxProposalsSlider().getValue()).isEqualTo(25);

        assertThat(app.getDetConfValueLabel().getText()).isEqualTo("0.20");
        assertThat(app.getClsConfValueLabel().getText()).isEqualTo("0.25");
        assertThat(app.getNmsIouValueLabel().getText()).isEqualTo("0.25");
        assertThat(app.getProposalCapValueLabel().getText()).isEqualTo("25");

        app.getDetConfidenceSlider().setValue(45);
        assertThat(app.getDetConfValueLabel().getText()).isEqualTo("0.45");
        assertThat(app.getSafetyEngine().getMinDetectionConfidence()).isEqualTo(0.45f);
        if (app.getPipeline() != null) {
            assertThat(app.getPipeline().getDetectionConfidenceThreshold()).isEqualTo(0.45f);
        }

        app.getClsConfidenceSlider().setValue(30);
        assertThat(app.getClsConfValueLabel().getText()).isEqualTo("0.30");
        assertThat(app.getSafetyEngine().getMinClassificationConfidence()).isEqualTo(0.30f);
        if (app.getPipeline() != null) {
            assertThat(app.getPipeline().getClassificationConfidenceThreshold()).isEqualTo(0.30f);
        }

        app.getNmsIouSlider().setValue(50);
        assertThat(app.getNmsIouValueLabel().getText()).isEqualTo("0.50");
        assertThat(app.getSafetyEngine().getTrackIouThreshold()).isEqualTo(0.50);
        if (app.getPipeline() != null) {
            assertThat(app.getPipeline().getNmsIouThreshold()).isEqualTo(0.50);
        }

        app.getMaxProposalsSlider().setValue(40);
        assertThat(app.getProposalCapValueLabel().getText()).isEqualTo("40");
        if (app.getPipeline() != null) {
            assertThat(app.getPipeline().getMaxProposalsPerFrame()).isEqualTo(40);
        }

        app.dispose();
    }

    @Test
    void testInvalidAndUnmappedClassFallback() {
        SafetyEventEngine engine = new SafetyEventEngine();
        SignRuleCatalogue catalogue = engine.getCatalogue();

        SignRuleCatalogue.SignRule fallbackRuleNeg = catalogue.getRule(-5);
        assertThat(fallbackRuleNeg.classId()).isEqualTo(-1);
        assertThat(fallbackRuleNeg.canonicalLabel()).isEqualTo("UNKNOWN_SIGN");
        assertThat(fallbackRuleNeg.eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);
        assertThat(fallbackRuleNeg.severity()).isEqualTo(SignRuleCatalogue.SeverityLevel.INFO);
        assertThat(fallbackRuleNeg.numericalValue()).isNull();

        SignRuleCatalogue.SignRule fallbackRuleOut = catalogue.getRule(999);
        assertThat(fallbackRuleOut.classId()).isEqualTo(-1);
        assertThat(fallbackRuleOut.canonicalLabel()).isEqualTo("UNKNOWN_SIGN");
        assertThat(fallbackRuleOut.eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);

        BoundingBox box = BoundingBox.ofNormalized(0.2, 0.2, 0.3, 0.3, 999, "UNKNOWN");
        RoadSceneInferencePipeline.RoadSceneDetection det = new RoadSceneInferencePipeline.RoadSceneDetection(
                box.toAbsolute(800, 600), box, 0.70f, 999, "UNKNOWN", "UNKNOWN", "", 0.80f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        List<SafetyEvent> events = engine.processFrame(1, 1000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det)));
        assertThat(events).hasSize(1);
        assertThat(events.get(0).eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);
        assertThat(events.get(0).severity()).isEqualTo(SignRuleCatalogue.SeverityLevel.INFO);
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isNull();
    }

    @Test
    void testDeduplicationAndElevationOnTrack() {
        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.20f);

        BoundingBox box1 = BoundingBox.ofNormalized(0.40, 0.40, 0.50, 0.50, 51, "SPEED_LIMIT_40");
        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box1.toAbsolute(800, 600), box1, 0.60f, 51, "Speed Limit (40 km/h)", "REGULATORY", "Obey 40 km/h", 0.15f,
                RoadSceneInferencePipeline.UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE
        );

        List<SafetyEvent> f1Events = engine.processFrame(1, 1000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det1)));
        assertThat(f1Events).hasSize(1);
        assertThat(f1Events.get(0).eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isNull();

        BoundingBox box2 = BoundingBox.ofNormalized(0.41, 0.40, 0.51, 0.51, 51, "SPEED_LIMIT_40");
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box2.toAbsolute(800, 600), box2, 0.85f, 51, "Speed Limit (40 km/h)", "REGULATORY", "Obey 40 km/h", 0.88f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        List<SafetyEvent> f2Events = engine.processFrame(2, 1100L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det2)));
        assertThat(f2Events).hasSize(1);
        assertThat(f2Events.get(0).eventType()).isEqualTo(SignRuleCatalogue.EventType.SPEED_LIMIT_POSTED);
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(40);

        BoundingBox box3 = BoundingBox.ofNormalized(0.42, 0.41, 0.52, 0.52, 51, "SPEED_LIMIT_40");
        RoadSceneInferencePipeline.RoadSceneDetection det3 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box3.toAbsolute(800, 600), box3, 0.88f, 51, "Speed Limit (40 km/h)", "REGULATORY", "Obey 40 km/h", 0.90f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );

        List<SafetyEvent> f3Events = engine.processFrame(3, 1200L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det3)));
        assertThat(f3Events).isEmpty();
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(40);
        assertThat(engine.getEventLog()).hasSize(2);
    }

    @Test
    void testNonSpeedLimitEventDoesNotOverwriteActiveSpeedLimit() {
        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.20f);

        BoundingBox box1 = BoundingBox.ofNormalized(0.2, 0.2, 0.3, 0.3, 52, "SPEED_LIMIT_50");
        RoadSceneInferencePipeline.RoadSceneDetection det1 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box1.toAbsolute(800, 600), box1, 0.90f, 52, "Speed Limit (50 km/h)", "REGULATORY", "Obey 50 km/h", 0.95f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        engine.processFrame(1, 1000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det1)));
        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(50);

        BoundingBox box2 = BoundingBox.ofNormalized(0.6, 0.2, 0.7, 0.3, 36, "PEDESTRIAN_CROSSING");
        RoadSceneInferencePipeline.RoadSceneDetection det2 = new RoadSceneInferencePipeline.RoadSceneDetection(
                box2.toAbsolute(800, 600), box2, 0.88f, 36, "Pedestrian Crossing Ahead", "CAUTIONARY", "Watch for pedestrians", 0.91f,
                RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
        );
        engine.processFrame(2, 2000L, null, new RoadSceneInferencePipeline.PipelineResult(800, 600, 20, 10, List.of(det2)));

        assertThat(engine.getSafetyState().getCurrentPostedSpeedLimit()).isEqualTo(50);
        assertThat(engine.getSafetyState().getActiveHazards()).contains("Pedestrian Crossing Ahead");
    }
}
