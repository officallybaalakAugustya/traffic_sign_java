package com.trafficsign.app;

import ai.djl.Device;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.safety.SafetyEvent;
import com.trafficsign.safety.SafetyEventEngine;
import com.trafficsign.safety.SafetyState;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Task17AuditRunner {

    public record ModelCheckpointInfo(
            String modelType,
            String checkpointPath,
            String sha256Hex,
            long fileSizeBytes
    ) {
    }

    public record EvaluationCase(
            String caseName,
            String inputPath,
            int imageWidth,
            int imageHeight,
            int groundTruthSignCount,
            Map<String, ConfigResult> configResults
    ) {
    }

    public record ConfigResult(
            float detConfidenceThreshold,
            float clsConfidenceThreshold,
            double nmsIouThreshold,
            int maxProposalsPerFrame,
            int rawProposalsCount,
            int confidentDetectionsCount,
            int uncertainDetectionsCount,
            List<DetectionSummary> detections,
            List<EventSummary> emittedEvents,
            HudStateSummary hudState
    ) {
    }

    public record DetectionSummary(
            int classId,
            String className,
            float detConfidence,
            float clsConfidence,
            String uncertaintyState,
            double[] normalizedBox
    ) {
    }

    public record EventSummary(
            String eventId,
            String eventType,
            String severity,
            String displayName,
            Double numericalValue,
            String driverAdvisory
    ) {
    }

    public record HudStateSummary(
            Integer currentPostedSpeedLimit,
            List<String> activeRestrictions,
            List<String> activeHazards
    ) {
    }

    public record AuditReportPayload(
            String title,
            String timestamp,
            List<ModelCheckpointInfo> checkpoints,
            List<EvaluationCase> evaluationCases
    ) {
    }

    public static void main(String[] args) throws Exception {
        Path detParams = Path.of("models/checkpoints/detector_task15_champion/sign_detector-0004.params");
        Path clsParams = Path.of("models/checkpoints/task7_champion/exp_b_48x48-0008.params");

        List<ModelCheckpointInfo> checkpoints = List.of(
                computeCheckpointInfo("HighResSignDetector (Task 15 Champion)", detParams),
                computeCheckpointInfo("IndianTrafficSignCnnBlock (Task 7 Champion)", clsParams)
        );

        Path detDir = Path.of("models/checkpoints/detector_task15_champion");
        Path clsDir = Path.of("models/checkpoints/task7_champion");

        List<EvaluationCase> cases = new ArrayList<>();

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {

            File speed40File = new File("data/raw/traffic_sign_classification_dataset/test/SPEED_LIMIT_40/66000.jpg");
            BufferedImage speed40Patch = ImageIO.read(speed40File);
            BufferedImage speed40Scene = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = speed40Scene.createGraphics();
            g.setColor(new java.awt.Color(50, 70, 90));
            g.fillRect(0, 0, 800, 300);
            g.setColor(new java.awt.Color(70, 70, 70));
            g.fillRect(0, 300, 800, 300);
            g.drawImage(speed40Patch, 400, 200, 100, 100, null);
            g.dispose();

            cases.add(evaluateImage(
                    "Clear Speed-Limit 40 Sign",
                    "data/raw/traffic_sign_classification_dataset/test/SPEED_LIMIT_40/66000.jpg (embedded in road scene)",
                    speed40Scene,
                    1,
                    pipeline
            ));

            File emptyRoadFile = new File("data/raw/indian-road-driving-sample/inspection/000db725-8180-4770-8b6a-4eb74aeda9f9_0004.jpg");
            BufferedImage emptyRoadImg = ImageIO.read(emptyRoadFile);
            cases.add(evaluateImage(
                    "Empty Road Frame (Zero Signs)",
                    emptyRoadFile.getPath(),
                    emptyRoadImg,
                    0,
                    pipeline
            ));

            File multiSignFile = new File("data/raw/indian-road-driving-sample/inspection/000db725-8180-4770-8b6a-4eb74aeda9f9_0015.jpg");
            BufferedImage multiSignImg = ImageIO.read(multiSignFile);
            cases.add(evaluateImage(
                    "Multi-Sign Indian Road Frame",
                    multiSignFile.getPath(),
                    multiSignImg,
                    4,
                    pipeline
            ));
        }

        AuditReportPayload payload = new AuditReportPayload(
                "Task 17 False-Positive and Safety-State Integrity Audit Report",
                java.time.Instant.now().toString(),
                checkpoints,
                cases
        );

        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        Path outJson = Path.of("reports/task17_audit_results.json");
        Files.createDirectories(outJson.getParent());
        mapper.writeValue(outJson.toFile(), payload);
        System.out.println("Exported Task 17 audit payload to: " + outJson.toAbsolutePath());
    }

    private static EvaluationCase evaluateImage(
            String caseName,
            String inputPath,
            BufferedImage img,
            int gtCount,
            RoadSceneInferencePipeline pipeline
    ) {
        Map<String, ConfigResult> results = new LinkedHashMap<>();

        results.put("Config A (Task 16 UI Defaults: Det 0.20, Cls 0.25)",
                runConfig(img, pipeline, 0.20f, 0.25f, 0.25, 25));

        results.put("Config B (Unfiltered Low Threshold: Det 0.20, Cls 0.20)",
                runConfig(img, pipeline, 0.20f, 0.20f, 0.25, 25));

        results.put("Config C (Calibrated Balanced: Det 0.50, Cls 0.25)",
                runConfig(img, pipeline, 0.50f, 0.25f, 0.25, 25));

        results.put("Config D (Calibrated False-Positive Suppression: Det 0.55, Cls 0.35)",
                runConfig(img, pipeline, 0.55f, 0.35f, 0.25, 25));

        return new EvaluationCase(
                caseName,
                inputPath,
                img.getWidth(),
                img.getHeight(),
                gtCount,
                results
        );
    }

    private static ConfigResult runConfig(
            BufferedImage img,
            RoadSceneInferencePipeline pipeline,
            float detConf,
            float clsConf,
            double nmsIou,
            int maxProps
    ) {
        pipeline.setDetectionConfidenceThreshold(detConf);
        pipeline.setClassificationConfidenceThreshold(clsConf);
        pipeline.setNmsIouThreshold(nmsIou);
        pipeline.setMaxProposalsPerFrame(maxProps);

        RoadSceneInferencePipeline.PipelineResult res = pipeline.processImage(img);

        SafetyEventEngine engine = new SafetyEventEngine();
        engine.setMinDetectionConfidence(detConf);
        engine.setMinClassificationConfidence(clsConf);
        engine.setTrackIouThreshold(nmsIou);

        List<SafetyEvent> events = engine.processFrame(1, 1000L, img, res);
        SafetyState state = engine.getSafetyState();

        int confDets = 0;
        int uncertDets = 0;
        List<DetectionSummary> detSummaries = new ArrayList<>();
        if (res.detections() != null) {
            for (var d : res.detections()) {
                if (d.uncertaintyState() == RoadSceneInferencePipeline.UncertaintyState.CONFIDENT) {
                    confDets++;
                } else {
                    uncertDets++;
                }
                var nb = d.normalizedBox();
                detSummaries.add(new DetectionSummary(
                        d.predictedClassId(),
                        d.predictedClassName(),
                        d.detectionConfidence(),
                        d.classificationConfidence(),
                        d.uncertaintyState().name(),
                        new double[]{nb.x1(), nb.y1(), nb.x2(), nb.y2()}
                ));
            }
        }

        List<EventSummary> eventSummaries = new ArrayList<>();
        for (var evt : events) {
            eventSummaries.add(new EventSummary(
                    evt.eventId(),
                    evt.eventType().name(),
                    evt.severity().name(),
                    evt.displayName(),
                    evt.numericalValue(),
                    evt.driverAdvisory()
            ));
        }

        HudStateSummary hud = new HudStateSummary(
                state.getCurrentPostedSpeedLimit(),
                new ArrayList<>(state.getActiveRestrictions()),
                state.getActiveHazards()
        );

        return new ConfigResult(
                detConf,
                clsConf,
                nmsIou,
                maxProps,
                res.detections() != null ? res.detections().size() : 0,
                confDets,
                uncertDets,
                detSummaries,
                eventSummaries,
                hud
        );
    }

    private static ModelCheckpointInfo computeCheckpointInfo(String modelType, Path p) {
        try {
            long size = Files.size(p);
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream is = Files.newInputStream(p)) {
                byte[] buf = new byte[8192];
                int read;
                while ((read = is.read(buf)) != -1) {
                    digest.update(buf, 0, read);
                }
            }
            byte[] hash = digest.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02X", b));
            }
            return new ModelCheckpointInfo(modelType, p.toString(), sb.toString(), size);
        } catch (Exception e) {
            return new ModelCheckpointInfo(modelType, p.toString(), "ERROR: " + e.getMessage(), 0);
        }
    }
}
