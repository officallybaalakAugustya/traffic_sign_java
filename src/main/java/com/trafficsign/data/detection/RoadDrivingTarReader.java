package com.trafficsign.data.detection;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trafficsign.data.BoundingBox;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.utils.IOUtils;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class RoadDrivingTarReader {

    public static final int DEFAULT_FRAME_WIDTH = 1920;
    public static final int DEFAULT_FRAME_HEIGHT = 994;

    private final ObjectMapper objectMapper;

    public RoadDrivingTarReader() {
        this.objectMapper = new ObjectMapper();
    }

    public TarInspectionSummary inspectDirectory(Path sampleDir, int maxArchives) throws IOException {
        Objects.requireNonNull(sampleDir, "sampleDir must not be null");
        if (!Files.exists(sampleDir) || !Files.isDirectory(sampleDir)) {
            throw new IllegalArgumentException("Directory does not exist: " + sampleDir);
        }

        List<Path> tarFiles;
        try (var stream = Files.list(sampleDir)) {
            tarFiles = stream.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".tar"))
                    .sorted()
                    .toList();
        }

        if (maxArchives > 0 && tarFiles.size() > maxArchives) {
            tarFiles = tarFiles.subList(0, maxArchives);
        }

        return inspectArchives(tarFiles);
    }

    public TarInspectionSummary inspectArchives(List<Path> tarFiles) throws IOException {
        Objects.requireNonNull(tarFiles, "tarFiles must not be null");

        List<String> archiveNames = new ArrayList<>();
        Map<String, Integer> categoryDistribution = new TreeMap<>();
        List<String> annotationErrors = new ArrayList<>();
        List<FrameDetectionRecord> sampleFrames = new ArrayList<>();

        int totalFrames = 0;
        int totalMatched = 0;
        int totalJpgOnly = 0;
        int totalJsonOnly = 0;
        int totalPngIgnored = 0;
        int totalObjects = 0;
        int totalTrafficSigns = 0;

        for (Path tarPath : tarFiles) {
            String archiveName = tarPath.getFileName().toString();
            archiveNames.add(archiveName);

            Map<String, String> jpgEntries = new HashMap<>();
            Map<String, byte[]> jsonBytesMap = new HashMap<>();

            try (InputStream fis = Files.newInputStream(tarPath);
                 BufferedInputStream bis = new BufferedInputStream(fis);
                 TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {

                TarArchiveEntry entry;
                while ((entry = tis.getNextEntry()) != null) {
                    if (entry.isDirectory()) {
                        continue;
                    }

                    String name = entry.getName();
                    String lower = name.toLowerCase();

                    if (lower.endsWith(".png")) {
                        totalPngIgnored++;
                        IOUtils.skip(tis, entry.getSize());
                        continue;
                    }

                    String stem = extractStem(name);

                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                        jpgEntries.put(stem, name);
                        IOUtils.skip(tis, entry.getSize());
                    } else if (lower.endsWith(".json")) {
                        long size = entry.getSize();
                        if (size > 10 * 1024 * 1024) {
                            annotationErrors.add("Excessive JSON entry size (" + size + " bytes) in " + name);
                            IOUtils.skip(tis, size);
                            continue;
                        }
                        byte[] bytes = tis.readAllBytes();
                        jsonBytesMap.put(stem, bytes);
                    } else {
                        IOUtils.skip(tis, entry.getSize());
                    }
                }
            }

            Set<String> allStems = new TreeSet<>();
            allStems.addAll(jpgEntries.keySet());
            allStems.addAll(jsonBytesMap.keySet());

            for (String stem : allStems) {
                totalFrames++;
                boolean hasJpg = jpgEntries.containsKey(stem);
                boolean hasJson = jsonBytesMap.containsKey(stem);

                if (hasJpg && hasJson) {
                    totalMatched++;
                } else if (hasJpg) {
                    totalJpgOnly++;
                    annotationErrors.add("Missing JSON annotation for image frame: " + stem + " in " + archiveName);
                } else {
                    totalJsonOnly++;
                    annotationErrors.add("Missing JPG image for JSON annotation: " + stem + " in " + archiveName);
                }

                List<DetectionAnnotation> annotations = new ArrayList<>();
                if (hasJson) {
                    byte[] bytes = jsonBytesMap.get(stem);
                    try {
                        JsonNode root = objectMapper.readTree(new ByteArrayInputStream(bytes));
                        JsonNode labelsNode = root.get("labels");
                        if (labelsNode != null && labelsNode.isArray()) {
                            for (JsonNode labelNode : labelsNode) {
                                totalObjects++;
                                parseLabel(labelNode, stem, archiveName, annotations, categoryDistribution, annotationErrors);
                            }
                        }
                    } catch (Exception ex) {
                        annotationErrors.add("Malformed JSON in " + stem + " (" + archiveName + "): " + ex.getMessage());
                    }
                }

                int signsInFrame = (int) annotations.stream().filter(DetectionAnnotation::isGenericTrafficSign).count();
                totalTrafficSigns += signsInFrame;

                if (sampleFrames.size() < 1000) {
                    sampleFrames.add(new FrameDetectionRecord(
                            stem,
                            jpgEntries.get(stem),
                            hasJson ? stem + ".json" : null,
                            DEFAULT_FRAME_WIDTH,
                            DEFAULT_FRAME_HEIGHT,
                            annotations,
                            hasJpg,
                            hasJson
                    ));
                }
            }
        }

        return new TarInspectionSummary(
                archiveNames.size(),
                Collections.unmodifiableList(archiveNames),
                totalFrames,
                totalMatched,
                totalJpgOnly,
                totalJsonOnly,
                totalPngIgnored,
                totalObjects,
                totalTrafficSigns,
                Collections.unmodifiableMap(categoryDistribution),
                Collections.unmodifiableList(annotationErrors),
                Collections.unmodifiableList(sampleFrames)
        );
    }

    private void parseLabel(
            JsonNode labelNode,
            String stem,
            String archiveName,
            List<DetectionAnnotation> annotations,
            Map<String, Integer> categoryDistribution,
            List<String> annotationErrors
    ) {
        int id = labelNode.hasNonNull("id") ? labelNode.get("id").asInt() : -1;
        String category = labelNode.hasNonNull("category") ? labelNode.get("category").asText() : "";
        if (category.isBlank()) {
            annotationErrors.add("Empty or null category in " + stem + " (" + archiveName + ")");
            category = "unknown";
        }

        categoryDistribution.put(category, categoryDistribution.getOrDefault(category, 0) + 1);

        Double confidence = labelNode.hasNonNull("confidence") ? labelNode.get("confidence").asDouble() : null;
        Long trackId = labelNode.hasNonNull("track_id") ? labelNode.get("track_id").asLong() : null;

        boolean occluded = false;
        boolean truncated = false;
        if (labelNode.has("attributes")) {
            JsonNode attr = labelNode.get("attributes");
            if (attr.hasNonNull("occluded")) {
                occluded = attr.get("occluded").asBoolean();
            }
            if (attr.hasNonNull("truncated")) {
                truncated = attr.get("truncated").asBoolean();
            }
        }

        JsonNode boxNode = labelNode.get("box2d");
        if (boxNode == null || !boxNode.isObject()) {
            annotationErrors.add("Missing box2d object for object id " + id + " in " + stem);
            return;
        }

        double x1 = boxNode.hasNonNull("x1") ? boxNode.get("x1").asDouble() : -1.0;
        double y1 = boxNode.hasNonNull("y1") ? boxNode.get("y1").asDouble() : -1.0;
        double x2 = boxNode.hasNonNull("x2") ? boxNode.get("x2").asDouble() : -1.0;
        double y2 = boxNode.hasNonNull("y2") ? boxNode.get("y2").asDouble() : -1.0;

        if (x1 < 0.0 || y1 < 0.0 || x2 < x1 || y2 < y1) {
            annotationErrors.add("Inverted or negative box2d [" + x1 + "," + y1 + "," + x2 + "," + y2 + "] in " + stem);
            return;
        }

        if (x2 > DEFAULT_FRAME_WIDTH * 1.5 || y2 > DEFAULT_FRAME_HEIGHT * 1.5) {
            annotationErrors.add("Severe out-of-bounds box2d [" + x1 + "," + y1 + "," + x2 + "," + y2 + "] in " + stem);
            return;
        }

        int classId = "traffic sign".equalsIgnoreCase(category) ? 1 : 0;
        BoundingBox box = BoundingBox.ofPixels(x1, y1, x2, y2, classId, category);
        annotations.add(new DetectionAnnotation(id, category, box, confidence, occluded, truncated, trackId));
    }

    private String extractStem(String path) {
        String filename = path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path;
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
