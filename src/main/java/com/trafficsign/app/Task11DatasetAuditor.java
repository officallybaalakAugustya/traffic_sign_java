package com.trafficsign.app;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.data.detection.DetectionDataPipeline;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.utils.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Task11DatasetAuditor {

    private static final Logger logger = LoggerFactory.getLogger(Task11DatasetAuditor.class);

    public record AdditionalShardAudit(
            String archiveName,
            long fileSize,
            int jpgCount,
            int jsonCount,
            int pngCount,
            int matchedPairs,
            int totalSignBoxes,
            int framesWithSigns,
            int emptyFrames
    ) {
    }

    public record PngRoleAnalysis(
            String samplePngName,
            long samplePngSize,
            String detectedRole,
            String reason
    ) {
    }

    public record SequenceLeakageCheck(
            int totalNewSequences,
            int validationSequencesCount,
            int overlappingSequencesCount,
            List<String> overlappingSequences,
            boolean passed
    ) {
    }

    public record ExpandedDatasetAuditReport(
            String timestamp,
            int totalShardsScanned,
            long totalBytesScanned,
            int totalJpgFiles,
            int totalJsonFiles,
            int totalPngFiles,
            int totalMatchedPairs,
            int totalSignBoundingBoxes,
            int totalFramesWithSigns,
            int totalEmptyFrames,
            int distinctNewSequences,
            Map<String, Integer> rejectionReasons,
            PngRoleAnalysis pngRoleAnalysis,
            SequenceLeakageCheck sequenceLeakageCheck,
            List<AdditionalShardAudit> shardAudits,
            String positiveOverlayPath,
            String backgroundOverlayPath
    ) {
    }

    private final ObjectMapper mapper;

    public Task11DatasetAuditor() {
        this.mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public static void main(String[] args) {
        Task11DatasetAuditor auditor = new Task11DatasetAuditor();
        try {
            auditor.execute(
                    Path.of("data/raw/indian-road-driving-sample/additional-shards"),
                    Path.of("data/splits/indian_road_detection_splits.json"),
                    Path.of("reports"),
                    Path.of("reports/task11_visualizations")
            );
        } catch (Exception e) {
            logger.error("Audit failed", e);
            System.err.println("Audit failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    public ExpandedDatasetAuditReport execute(
            Path additionalShardsDir,
            Path existingSplitsManifest,
            Path reportsDir,
            Path visualDir
    ) throws Exception {
        Files.createDirectories(reportsDir);
        Files.createDirectories(visualDir);

        logger.info("Stage 1: Scanning tar archives in {}", additionalShardsDir);
        List<Path> tarFiles;
        try (var s = Files.list(additionalShardsDir)) {
            tarFiles = s.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".tar"))
                    .sorted()
                    .toList();
        }

        if (tarFiles.isEmpty()) {
            throw new IllegalStateException("No tar archives found in " + additionalShardsDir);
        }

        Set<String> existingValSequences = new HashSet<>();
        if (Files.exists(existingSplitsManifest)) {
            DetectionDataPipeline.ManifestSummary existingManifest = mapper.readValue(
                    existingSplitsManifest.toFile(),
                    DetectionDataPipeline.ManifestSummary.class
            );
            for (var rec : existingManifest.samples()) {
                if (rec.partition() == DetectionDataPipeline.SplitPartition.VALIDATION) {
                    existingValSequences.add(rec.sequenceId());
                }
            }
        }
        logger.info("Loaded {} existing sequence IDs assigned to validation partition.", existingValSequences.size());

        List<AdditionalShardAudit> shardAudits = new ArrayList<>();
        Map<String, Integer> rejectionReasons = new LinkedHashMap<>();
        Set<String> allNewSequences = new TreeSet<>();
        Set<String> overlappingSequences = new TreeSet<>();

        int totalJpg = 0;
        int totalJson = 0;
        int totalPng = 0;
        int totalMatched = 0;
        int totalSignBoxes = 0;
        int totalFramesWithSigns = 0;
        int totalEmpty = 0;
        long totalBytes = 0;

        String samplePngName = null;
        long samplePngSize = 0;

        byte[] samplePositiveJpgBytes = null;
        List<BoundingBox> samplePositiveBoxes = null;
        String samplePositiveStem = null;

        byte[] sampleBackgroundJpgBytes = null;
        String sampleBackgroundStem = null;

        for (Path tarPath : tarFiles) {
            String archiveName = tarPath.getFileName().toString();
            long fileSize = Files.size(tarPath);
            totalBytes += fileSize;

            Map<String, String> jpgEntries = new HashMap<>();
            Map<String, byte[]> jsonBytesMap = new HashMap<>();
            Map<String, byte[]> jpgBytesMap = new HashMap<>();

            int shardPngCount = 0;

            try (InputStream fis = Files.newInputStream(tarPath);
                 BufferedInputStream bis = new BufferedInputStream(fis);
                 TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {

                TarArchiveEntry entry;
                while ((entry = tis.getNextEntry()) != null) {
                    if (entry.isDirectory()) continue;
                    String name = entry.getName();
                    String lower = name.toLowerCase();

                    if (lower.endsWith(".png")) {
                        shardPngCount++;
                        totalPng++;
                        if (samplePngName == null) {
                            samplePngName = name;
                            samplePngSize = entry.getSize();
                        }
                        IOUtils.skip(tis, entry.getSize());
                        continue;
                    }

                    String stem = extractStem(name);
                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                        totalJpg++;
                        jpgEntries.put(stem, name);
                        if (samplePositiveJpgBytes == null || sampleBackgroundJpgBytes == null) {
                            byte[] b = tis.readAllBytes();
                            jpgBytesMap.put(stem, b);
                        } else {
                            IOUtils.skip(tis, entry.getSize());
                        }
                    } else if (lower.endsWith(".json")) {
                        totalJson++;
                        byte[] bytes = tis.readAllBytes();
                        jsonBytesMap.put(stem, bytes);
                    } else {
                        rejectionReasons.put("UNSUPPORTED_EXTENSION_" + getExtension(name),
                                rejectionReasons.getOrDefault("UNSUPPORTED_EXTENSION_" + getExtension(name), 0) + 1);
                        IOUtils.skip(tis, entry.getSize());
                    }
                }
            }

            int shardSignBoxes = 0;
            int shardFramesWithSigns = 0;
            int shardEmptyFrames = 0;
            int shardMatched = 0;

            for (String stem : jpgEntries.keySet()) {
                String seqId = DetectionDataPipeline.parseSequenceId(stem);
                allNewSequences.add(seqId);
                if (existingValSequences.contains(seqId)) {
                    overlappingSequences.add(seqId);
                }

                if (!jsonBytesMap.containsKey(stem)) {
                    rejectionReasons.put("MISSING_JSON_ANNOTATION", rejectionReasons.getOrDefault("MISSING_JSON_ANNOTATION", 0) + 1);
                    continue;
                }

                shardMatched++;
                totalMatched++;

                byte[] jsonBytes = jsonBytesMap.get(stem);
                List<BoundingBox> signBoxes = new ArrayList<>();

                try {
                    JsonNode root = mapper.readTree(new ByteArrayInputStream(jsonBytes));
                    JsonNode labelsNode = root.get("labels");
                    if (labelsNode != null && labelsNode.isArray()) {
                        for (JsonNode labelNode : labelsNode) {
                            String category = labelNode.hasNonNull("category") ? labelNode.get("category").asText() : "";
                            if (!"traffic sign".equalsIgnoreCase(category.trim())) {
                                continue;
                            }

                            JsonNode boxNode = labelNode.get("box2d");
                            if (boxNode == null || !boxNode.isObject()) {
                                rejectionReasons.put("MISSING_BOX2D", rejectionReasons.getOrDefault("MISSING_BOX2D", 0) + 1);
                                continue;
                            }

                            double x1 = boxNode.hasNonNull("x1") ? boxNode.get("x1").asDouble() : -1.0;
                            double y1 = boxNode.hasNonNull("y1") ? boxNode.get("y1").asDouble() : -1.0;
                            double x2 = boxNode.hasNonNull("x2") ? boxNode.get("x2").asDouble() : -1.0;
                            double y2 = boxNode.hasNonNull("y2") ? boxNode.get("y2").asDouble() : -1.0;

                            if (x1 < 0.0 || y1 < 0.0 || x2 <= x1 || y2 <= y1) {
                                rejectionReasons.put("DEGENERATE_OR_NEGATIVE_BOX",
                                        rejectionReasons.getOrDefault("DEGENERATE_OR_NEGATIVE_BOX", 0) + 1);
                                continue;
                            }

                            if (x2 > 1920 || y2 > 994) {
                                double cx1 = Math.min(x1, 1919);
                                double cy1 = Math.min(y1, 993);
                                double cx2 = Math.min(x2, 1920);
                                double cy2 = Math.min(y2, 994);
                                if (cx2 <= cx1 || cy2 <= cy1) {
                                    rejectionReasons.put("OUT_OF_BOUNDS_BOX", rejectionReasons.getOrDefault("OUT_OF_BOUNDS_BOX", 0) + 1);
                                    continue;
                                }
                                signBoxes.add(BoundingBox.ofPixels(cx1, cy1, cx2, cy2, 1, "traffic sign"));
                            } else {
                                signBoxes.add(BoundingBox.ofPixels(x1, y1, x2, y2, 1, "traffic sign"));
                            }
                        }
                    }
                } catch (Exception ex) {
                    rejectionReasons.put("JSON_PARSE_ERROR", rejectionReasons.getOrDefault("JSON_PARSE_ERROR", 0) + 1);
                }

                if (!signBoxes.isEmpty()) {
                    shardSignBoxes += signBoxes.size();
                    shardFramesWithSigns++;
                    totalSignBoxes += signBoxes.size();
                    totalFramesWithSigns++;
                    if (samplePositiveJpgBytes == null && jpgBytesMap.containsKey(stem)) {
                        samplePositiveJpgBytes = jpgBytesMap.get(stem);
                        samplePositiveBoxes = new ArrayList<>(signBoxes);
                        samplePositiveStem = stem;
                    }
                } else {
                    shardEmptyFrames++;
                    totalEmpty++;
                    if (sampleBackgroundJpgBytes == null && jpgBytesMap.containsKey(stem)) {
                        sampleBackgroundJpgBytes = jpgBytesMap.get(stem);
                        sampleBackgroundStem = stem;
                    }
                }
            }

            shardAudits.add(new AdditionalShardAudit(
                    archiveName,
                    fileSize,
                    jpgEntries.size(),
                    jsonBytesMap.size(),
                    shardPngCount,
                    shardMatched,
                    shardSignBoxes,
                    shardFramesWithSigns,
                    shardEmptyFrames
            ));
            logger.info("Audited shard {}: JPG={}, JSON={}, PNG={}, Matched={}, SignBoxes={}, FramesWithSigns={}, Empty={}",
                    archiveName, jpgEntries.size(), jsonBytesMap.size(), shardPngCount, shardMatched,
                    shardSignBoxes, shardFramesWithSigns, shardEmptyFrames);
        }

        String pngRole = "Semantic segmentation / instance mask";
        String pngReason = "PNG files in Indian Road Driving tar archives contain pixel-level class ID masks (e.g. for drivable area and road scene segmentation) rather than camera RGB road frames. Object detection and bounding box classification operate on the RGB JPEG frames paired with JSON annotations.";
        PngRoleAnalysis pngAnalysis = new PngRoleAnalysis(
                samplePngName != null ? samplePngName : "none",
                samplePngSize,
                pngRole,
                pngReason
        );

        SequenceLeakageCheck leakageCheck = new SequenceLeakageCheck(
                allNewSequences.size(),
                existingValSequences.size(),
                overlappingSequences.size(),
                new ArrayList<>(overlappingSequences),
                overlappingSequences.isEmpty()
        );
        logger.info("Sequence Leakage Audit: {} new sequences scanned. Overlap with validation sequences: {} (Passed={})",
                allNewSequences.size(), overlappingSequences.size(), leakageCheck.passed());

        Path posOverlayPath = visualDir.resolve("ground_truth_positive_overlay.png");
        Path bgOverlayPath = visualDir.resolve("ground_truth_background_overlay.png");

        if (samplePositiveJpgBytes != null) {
            BufferedImage posImg = ImageIO.read(new ByteArrayInputStream(samplePositiveJpgBytes));
            if (posImg != null) {
                renderGroundTruthOverlay(posImg, samplePositiveBoxes, samplePositiveStem, true);
                ImageIO.write(posImg, "png", posOverlayPath.toFile());
                logger.info("Rendered positive ground-truth overlay to {}", posOverlayPath);
            }
        }

        if (sampleBackgroundJpgBytes != null) {
            BufferedImage bgImg = ImageIO.read(new ByteArrayInputStream(sampleBackgroundJpgBytes));
            if (bgImg != null) {
                renderGroundTruthOverlay(bgImg, List.of(), sampleBackgroundStem, false);
                ImageIO.write(bgImg, "png", bgOverlayPath.toFile());
                logger.info("Rendered background ground-truth overlay to {}", bgOverlayPath);
            }
        }

        ExpandedDatasetAuditReport report = new ExpandedDatasetAuditReport(
                java.time.Instant.now().toString(),
                tarFiles.size(),
                totalBytes,
                totalJpg,
                totalJson,
                totalPng,
                totalMatched,
                totalSignBoxes,
                totalFramesWithSigns,
                totalEmpty,
                allNewSequences.size(),
                rejectionReasons,
                pngAnalysis,
                leakageCheck,
                shardAudits,
                posOverlayPath.toString().replace('\\', '/'),
                bgOverlayPath.toString().replace('\\', '/')
        );

        Path jsonOut = reportsDir.resolve("task11_dataset_audit.json");
        mapper.writeValue(jsonOut.toFile(), report);
        logger.info("Saved expanded dataset audit report to {}", jsonOut);

        return report;
    }

    private void renderGroundTruthOverlay(BufferedImage img, List<BoundingBox> boxes, String stem, boolean isPositive) {
        Graphics2D g = img.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            int w = img.getWidth();
            int h = img.getHeight();

            if (isPositive) {
                for (BoundingBox b : boxes) {
                    int bx = (int) Math.round(b.x1());
                    int by = (int) Math.round(b.y1());
                    int bw = (int) Math.round(b.width());
                    int bh = (int) Math.round(b.height());

                    g.setColor(new Color(0, 255, 68));
                    g.setStroke(new BasicStroke(3.5f));
                    g.drawRect(bx, by, bw, bh);

                    String label = String.format("GROUND TRUTH: %s [%dx%d at (%d,%d)]",
                            b.label(), bw, bh, bx, by);
                    g.setFont(new Font("SansSerif", Font.BOLD, 14));
                    FontMetrics fm = g.getFontMetrics();
                    int textW = fm.stringWidth(label);
                    int textH = fm.getHeight();

                    int pillX = Math.max(0, Math.min(bx, w - textW - 14));
                    int pillY = Math.max(textH + 6, by - 6);

                    g.setColor(new Color(0, 0, 0, 200));
                    g.fillRoundRect(pillX, pillY - textH + 2, textW + 12, textH + 4, 6, 6);
                    g.setColor(new Color(0, 255, 68));
                    g.drawRoundRect(pillX, pillY - textH + 2, textW + 12, textH + 4, 6, 6);
                    g.setColor(Color.WHITE);
                    g.drawString(label, pillX + 6, pillY - 2);
                }
            } else {
                g.setColor(new Color(0, 0, 0, 180));
                g.fillRoundRect(20, 20, 480, 50, 8, 8);
                g.setColor(new Color(51, 153, 255));
                g.setStroke(new BasicStroke(2.0f));
                g.drawRoundRect(20, 20, 480, 50, 8, 8);
                g.setColor(Color.WHITE);
                g.setFont(new Font("SansSerif", Font.BOLD, 18));
                g.drawString("BACKGROUND FRAME — ZERO TRAFFIC SIGNS", 32, 52);
            }

            g.setColor(new Color(0, 0, 0, 160));
            g.fillRect(0, h - 30, w, 30);
            g.setColor(Color.WHITE);
            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            g.drawString("Indian Road Driving Sample Dataset | Frame: " + stem + " (" + w + "x" + h + ") | Annotations: " + boxes.size(), 20, h - 10);
        } finally {
            g.dispose();
        }
    }

    private static String extractStem(String path) {
        String filename = path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path;
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private static String getExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(dot + 1).toLowerCase() : "unknown";
    }
}
