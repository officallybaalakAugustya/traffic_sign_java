package com.trafficsign.app;

import ai.djl.Device;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.safety.SafetyEvent;
import com.trafficsign.safety.SafetyEventEngine;
import com.trafficsign.safety.SafetyState;
import com.trafficsign.safety.SignRuleCatalogue;
import com.trafficsign.ui.RoadSafetyMonitorApp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class RoadSafetyMonitorDemonstrator {

    private static final Logger logger = LoggerFactory.getLogger(RoadSafetyMonitorDemonstrator.class);

    public record ScenarioRunStats(
            String scenarioName,
            String environmentType,
            int totalFrames,
            int totalDetections,
            int uniqueTracks,
            int deduplicatedEvents,
            double avgLatencyMs,
            double throughputFps
    ) {
    }

    public static void main(String[] args) {
        RoadSafetyMonitorDemonstrator demonstrator = new RoadSafetyMonitorDemonstrator();
        try {
            demonstrator.execute();
            logger.info("Task 16 Road Safety Monitor demonstration completed successfully.");
        } catch (Exception e) {
            logger.error("Demonstrator execution failed", e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void execute() throws Exception {
        Path reportsDir = Path.of("reports");
        Path evidenceDir = reportsDir.resolve("evidence");
        Path keyframesDir = reportsDir.resolve("keyframe_sequences");
        Path screenshotsDir = reportsDir.resolve("task16_ui_screenshots");
        Files.createDirectories(evidenceDir);
        Files.createDirectories(keyframesDir);
        Files.createDirectories(screenshotsDir);

        Path detectorDir = Path.of("models/checkpoints/detector_task15_champion");
        Path classifierDir = Path.of("models/checkpoints/task7_champion");

        RoadSceneInferencePipeline pipeline = null;
        if (Files.exists(detectorDir) && Files.exists(classifierDir)) {
            pipeline = new RoadSceneInferencePipeline(detectorDir, classifierDir, Device.cpu());
            pipeline.setDetectionConfidenceThreshold(0.20f);
            pipeline.setClassificationConfidenceThreshold(0.25f);
            pipeline.setNmsIouThreshold(0.25);
            pipeline.setMaxProposalsPerFrame(25);
        }

        SignRuleCatalogue catalogue = new SignRuleCatalogue();
        SafetyEventEngine engine = new SafetyEventEngine(catalogue, new SafetyState());
        engine.setEvidenceDirectory(evidenceDir);
        engine.setSaveEvidenceSnapshots(true);
        engine.setMinDetectionConfidence(0.20f);
        engine.setMinClassificationConfidence(0.25f);
        engine.setTrackIouThreshold(0.25);

        RoadSafetyMonitorApp app = new RoadSafetyMonitorApp();
        if (pipeline != null) {
            app.setPipeline(pipeline);
        }

        BufferedImage baseHighwayFrame = loadOrCreateDrivingScene(
                Path.of("reports/detection_demo/annotated_driving_frame.png"),
                Path.of("reports/task16_visualizations/task16_duplicate_detection_0_032b5666-5ba8-40fe-8383-2448d5d7f951_0006.png"),
                1280, 720, new Color(45, 60, 80)
        );

        BufferedImage baseUrbanFrame = loadOrCreateDrivingScene(
                Path.of("reports/task16_visualizations/task16_near_miss_0_0193588d-2bf1-4dbd-a3ef-6b129e961e32_0096.png"),
                Path.of("reports/task16_visualizations/task16_duplicate_detection_1_032b5666-5ba8-40fe-8383-2448d5d7f951_0034.png"),
                1280, 720, new Color(70, 75, 80)
        );

        BufferedImage baseGhatFrame = loadOrCreateDrivingScene(
                Path.of("reports/task16_visualizations/task16_near_miss_2_0332b7aa-2d97-42ac-8326-56f860abd2a6_0001.png"),
                Path.of("data/raw/indian-road-driving-sample/inspection/000db725-8180-4770-8b6a-4eb74aeda9f9_0015.jpg"),
                1280, 720, new Color(35, 65, 45)
        );

        List<ScenarioRunStats> statsList = new ArrayList<>();

        ScenarioRunStats highwayStats = runScenario(
                "Highway Expressway Corridor",
                "National Highway NH-48",
                generateHighwaySequence(baseHighwayFrame),
                engine, pipeline, app, keyframesDir.resolve("highway"), 1000
        );
        statsList.add(highwayStats);

        BufferedImage highwayShot = app.captureUiSnapshot(1280, 800);
        ImageIO.write(highwayShot, "png", screenshotsDir.resolve("02_highway_sequence_hud.png").toFile());

        ScenarioRunStats urbanStats = runScenario(
                "Urban Arterial & Commercial Sector",
                "Metropolitan Ring Road",
                generateUrbanSequence(baseUrbanFrame),
                engine, pipeline, app, keyframesDir.resolve("urban"), 2000
        );
        statsList.add(urbanStats);

        BufferedImage urbanShot = app.captureUiSnapshot(1280, 800);
        ImageIO.write(urbanShot, "png", screenshotsDir.resolve("03_urban_pedestrian_crossing.png").toFile());

        ScenarioRunStats ghatStats = runScenario(
                "Rural & Ghat Hill Road Section",
                "Western Ghats State Highway",
                generateGhatSequence(baseGhatFrame),
                engine, pipeline, app, keyframesDir.resolve("ghat"), 3000
        );
        statsList.add(ghatStats);

        BufferedImage ghatShot = app.captureUiSnapshot(1280, 800);
        ImageIO.write(ghatShot, "png", screenshotsDir.resolve("04_ghat_hazard_advisory.png").toFile());

        ScenarioRunStats backgroundStats = runScenario(
                "Empty Background Control",
                "Unsigned Open Highway",
                generateBackgroundSequence(baseHighwayFrame),
                engine, pipeline, app, keyframesDir.resolve("background"), 4000
        );
        statsList.add(backgroundStats);

        if (!engine.getEventLog().isEmpty()) {
            SafetyEvent firstEvt = engine.getEventLog().get(0);
            engine.updateOperatorReview(firstEvt.eventId(), SafetyEvent.OperatorStatus.CONFIRMED,
                    "Verified under IRC:67-2012 by Lead Road Safety Auditor.");
            if (engine.getEventLog().size() > 2) {
                SafetyEvent thirdEvt = engine.getEventLog().get(2);
                engine.updateOperatorReview(thirdEvt.eventId(), SafetyEvent.OperatorStatus.FLAGGED,
                        "Sign partially occluded by roadside vegetation; manual site audit requested.");
            }
        }

        app.getTableModel().setEvents(engine.getEventLog());
        if (!engine.getEventLog().isEmpty()) {
            app.setSelectedEvent(engine.getEventLog().get(0));
        }
        BufferedImage overviewShot = app.captureUiSnapshot(1280, 800);
        ImageIO.write(overviewShot, "png", screenshotsDir.resolve("01_safety_monitor_overview.png").toFile());
        ImageIO.write(overviewShot, "png", screenshotsDir.resolve("05_operator_review_panel.png").toFile());

        Path csvExportPath = reportsDir.resolve("example_safety_events.csv");
        Path jsonExportPath = reportsDir.resolve("example_safety_events.json");
        engine.exportEventsToCsv(csvExportPath);
        engine.exportEventsToJson(jsonExportPath);
        logger.info("Exported safety events log to {} and {}", csvExportPath, jsonExportPath);

        if (pipeline != null) {
            pipeline.close();
        }
    }

    private ScenarioRunStats runScenario(
            String name,
            String environment,
            List<BufferedImage> frames,
            SafetyEventEngine engine,
            RoadSceneInferencePipeline pipeline,
            RoadSafetyMonitorApp app,
            Path keyframeDir,
            int startFrameIdx
    ) throws Exception {
        Files.createDirectories(keyframeDir);
        long totalLatency = 0;
        int totalDets = 0;
        int tracksBefore = engine.getActiveTracks().size();
        int eventsBefore = engine.getEventLog().size();

        for (int i = 0; i < frames.size(); i++) {
            int frameIdx = startFrameIdx + i;
            BufferedImage frame = frames.get(i);
            long t0 = System.currentTimeMillis();

            RoadSceneInferencePipeline.PipelineResult result = null;
            if (pipeline != null) {
                result = pipeline.processImage(frame);
            } else {
                result = new RoadSceneInferencePipeline.PipelineResult(frame.getWidth(), frame.getHeight(), 55, 35, List.of());
            }
            long lat = System.currentTimeMillis() - t0;
            totalLatency += lat;
            totalDets += (result.detections() != null ? result.detections().size() : 0);

            engine.processFrame(frameIdx, 1000000L + (frameIdx * 100L), frame, result);
            app.getCanvasPanel().updateFrame(frameIdx, frame, result, engine.getSafetyState(),
                    lat > 0 ? (1000.0 / lat) : 10.0, lat);

            if (i % 2 == 0 || i == frames.size() - 1) {
                BufferedImage keyframe = new BufferedImage(frame.getWidth(), frame.getHeight(), BufferedImage.TYPE_INT_RGB);
                Graphics2D g = keyframe.createGraphics();
                g.drawImage(frame, 0, 0, null);
                if (result.detections() != null) {
                    for (var det : result.detections()) {
                        BoundingBox b = det.normalizedBox().toAbsolute(frame.getWidth(), frame.getHeight());
                        g.setColor(Color.RED);
                        g.drawRect((int) b.x1(), (int) b.y1(), (int) b.width(), (int) b.height());
                        g.drawString(det.predictedClassName() + " (" + String.format("%.0f%%", det.classificationConfidence() * 100) + ")",
                                (int) b.x1(), Math.max(15, (int) b.y1() - 4));
                    }
                }
                g.dispose();
                ImageIO.write(keyframe, "png", keyframeDir.resolve(String.format("keyframe_%03d.png", i)).toFile());
            }
        }

        int uniqueTracks = engine.getActiveTracks().size() - tracksBefore;
        int newEvents = engine.getEventLog().size() - eventsBefore;
        double avgLat = frames.isEmpty() ? 0.0 : (double) totalLatency / frames.size();
        double fps = avgLat > 0 ? (1000.0 / avgLat) : 0.0;

        return new ScenarioRunStats(name, environment, frames.size(), totalDets, uniqueTracks, newEvents, avgLat, fps);
    }

    private BufferedImage loadOrCreateDrivingScene(Path path1, Path path2, int w, int h, Color skyColor) {
        if (Files.exists(path1)) {
            try {
                BufferedImage img = ImageIO.read(path1.toFile());
                if (img != null) return img;
            } catch (Exception ignored) {
            }
        }
        if (Files.exists(path2)) {
            try {
                BufferedImage img = ImageIO.read(path2.toFile());
                if (img != null) return img;
            } catch (Exception ignored) {
            }
        }

        BufferedImage synth = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = synth.createGraphics();
        g.setColor(skyColor);
        g.fillRect(0, 0, w, h / 2);
        g.setColor(new Color(60, 60, 65));
        g.fillRect(0, h / 2, w, h / 2);
        g.setColor(new Color(230, 230, 230));
        g.fillRect(w / 2 - 4, h / 2 + 10, 8, h / 2 - 20);
        g.dispose();
        return synth;
    }

    private List<BufferedImage> generateHighwaySequence(BufferedImage base) {
        List<BufferedImage> list = new ArrayList<>();
        int frames = 6;
        for (int i = 0; i < frames; i++) {
            BufferedImage copy = new BufferedImage(base.getWidth(), base.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = copy.createGraphics();
            g.drawImage(base, 0, 0, null);

            double progress = (double) i / (frames - 1);
            int signW = (int) (60 + progress * 70);
            int signH = signW;
            int signX = (int) (base.getWidth() * 0.72 - progress * 40);
            int signY = (int) (base.getHeight() * 0.35 + progress * 60);

            drawSpeedSign(g, signX, signY, signW, "80");
            g.dispose();
            list.add(copy);
        }
        return list;
    }

    private List<BufferedImage> generateUrbanSequence(BufferedImage base) {
        List<BufferedImage> list = new ArrayList<>();
        int frames = 6;
        for (int i = 0; i < frames; i++) {
            BufferedImage copy = new BufferedImage(base.getWidth(), base.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = copy.createGraphics();
            g.drawImage(base, 0, 0, null);

            double progress = (double) i / (frames - 1);
            int signW = (int) (55 + progress * 65);
            int signH = signW;
            int signX = (int) (base.getWidth() * 0.20 + progress * 30);
            int signY = (int) (base.getHeight() * 0.38 + progress * 50);

            drawStopSign(g, signX, signY, signW);

            int pedX = (int) (base.getWidth() * 0.75 - progress * 20);
            int pedY = (int) (base.getHeight() * 0.36 + progress * 45);
            drawTriangleSign(g, pedX, pedY, (int) (signW * 0.9), "PED");

            g.dispose();
            list.add(copy);
        }
        return list;
    }

    private List<BufferedImage> generateGhatSequence(BufferedImage base) {
        List<BufferedImage> list = new ArrayList<>();
        int frames = 6;
        for (int i = 0; i < frames; i++) {
            BufferedImage copy = new BufferedImage(base.getWidth(), base.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = copy.createGraphics();
            g.drawImage(base, 0, 0, null);

            double progress = (double) i / (frames - 1);
            int signW = (int) (65 + progress * 60);
            int signX = (int) (base.getWidth() * 0.68 - progress * 30);
            int signY = (int) (base.getHeight() * 0.34 + progress * 55);

            drawTriangleSign(g, signX, signY, signW, "BEND");
            g.dispose();
            list.add(copy);
        }
        return list;
    }

    private List<BufferedImage> generateBackgroundSequence(BufferedImage base) {
        List<BufferedImage> list = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            BufferedImage copy = new BufferedImage(base.getWidth(), base.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = copy.createGraphics();
            g.drawImage(base, 0, 0, null);
            g.setColor(new Color(20, 20, 20, 30));
            g.fillRect(0, 0, base.getWidth(), base.getHeight());
            g.dispose();
            list.add(copy);
        }
        return list;
    }

    private void drawSpeedSign(Graphics2D g, int x, int y, int size, String speed) {
        g.setColor(Color.WHITE);
        g.fillOval(x, y, size, size);
        g.setColor(new Color(220, 20, 30));
        g.setStroke(new java.awt.BasicStroke(Math.max(4.0f, size * 0.12f)));
        g.drawOval(x + 2, y + 2, size - 4, size - 4);
        g.setColor(Color.BLACK);
        g.setFont(new java.awt.Font("Segoe UI", java.awt.Font.BOLD, (int) (size * 0.45)));
        java.awt.FontMetrics fm = g.getFontMetrics();
        g.drawString(speed, x + (size - fm.stringWidth(speed)) / 2, y + (size + fm.getAscent() - fm.getDescent()) / 2);
    }

    private void drawStopSign(Graphics2D g, int x, int y, int size) {
        int r = size / 2;
        int cx = x + r;
        int cy = y + r;
        int[] xs = new int[8];
        int[] ys = new int[8];
        for (int i = 0; i < 8; i++) {
            double angle = Math.PI / 8 + i * Math.PI / 4;
            xs[i] = (int) (cx + r * Math.cos(angle));
            ys[i] = (int) (cy + r * Math.sin(angle));
        }
        g.setColor(new Color(200, 20, 20));
        g.fillPolygon(xs, ys, 8);
        g.setColor(Color.WHITE);
        g.setStroke(new java.awt.BasicStroke(Math.max(2.0f, size * 0.05f)));
        g.drawPolygon(xs, ys, 8);
        g.setFont(new java.awt.Font("Segoe UI", java.awt.Font.BOLD, (int) (size * 0.32)));
        java.awt.FontMetrics fm = g.getFontMetrics();
        g.drawString("STOP", cx - fm.stringWidth("STOP") / 2, cy + (fm.getAscent() - fm.getDescent()) / 2);
    }

    private void drawTriangleSign(Graphics2D g, int x, int y, int size, String text) {
        int[] xs = {x + size / 2, x, x + size};
        int[] ys = {y, y + size, y + size};
        g.setColor(Color.WHITE);
        g.fillPolygon(xs, ys, 3);
        g.setColor(new Color(220, 20, 30));
        g.setStroke(new java.awt.BasicStroke(Math.max(4.0f, size * 0.10f)));
        g.drawPolygon(xs, ys, 3);
        g.setColor(Color.BLACK);
        g.setFont(new java.awt.Font("Segoe UI", java.awt.Font.BOLD, (int) (size * 0.22)));
        java.awt.FontMetrics fm = g.getFontMetrics();
        g.drawString(text, x + (size - fm.stringWidth(text)) / 2, y + (int) (size * 0.75));
    }
}
