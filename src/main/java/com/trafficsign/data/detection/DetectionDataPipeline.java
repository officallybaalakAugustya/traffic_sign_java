package com.trafficsign.data.detection;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
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

public class DetectionDataPipeline {

    public enum SplitPartition {
        TRAIN,
        VALIDATION,
        TEST
    }

    public record DetectionSampleRecord(
            @JsonProperty("frameStem") String frameStem,
            @JsonProperty("archiveName") String archiveName,
            @JsonProperty("sequenceId") String sequenceId,
            @JsonProperty("frameIndex") int frameIndex,
            @JsonProperty("imageWidth") int imageWidth,
            @JsonProperty("imageHeight") int imageHeight,
            @JsonProperty("boxes") List<BoundingBox> boxes,
            @JsonProperty("hasSign") boolean hasSign,
            @JsonProperty("partition") SplitPartition partition
    ) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ManifestSummary(
            @JsonProperty("totalArchivesScanned") int totalArchivesScanned,
            @JsonProperty("totalFramesScanned") int totalFramesScanned,
            @JsonProperty("totalMatchedJpgJsonPairs") int totalMatchedJpgJsonPairs,
            @JsonProperty("totalSequences") int totalSequences,
            @JsonProperty("trainSequences") int trainSequences,
            @JsonProperty("valSequences") int valSequences,
            @JsonProperty("trainFrames") int trainFrames,
            @JsonProperty("valFrames") int valFrames,
            @JsonProperty("trainSignBoxes") int trainSignBoxes,
            @JsonProperty("valSignBoxes") int valSignBoxes,
            @JsonProperty("trainFramesWithSign") int trainFramesWithSign,
            @JsonProperty("valFramesWithSign") int valFramesWithSign,
            @JsonProperty("trainEmptyFrames") int trainEmptyFrames,
            @JsonProperty("valEmptyFrames") int valEmptyFrames,
            @JsonProperty("excludedAnnotationsCount") int excludedAnnotationsCount,
            @JsonProperty("exclusionReasons") Map<String, Integer> exclusionReasons,
            @JsonProperty("sequenceDistribution") Map<String, Integer> sequenceDistribution,
            @JsonProperty("samples") List<DetectionSampleRecord> samples
    ) {
    }

    private final ObjectMapper objectMapper;

    public DetectionDataPipeline() {
        this.objectMapper = new ObjectMapper();
    }

    public ManifestSummary buildManifest(
            Path sampleDir,
            int maxArchives,
            double valRatio,
            long randomSeed
    ) throws IOException {
        List<Path> tarFiles;
        try (var stream = Files.list(sampleDir)) {
            tarFiles = stream.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".tar"))
                    .sorted()
                    .toList();
        }

        if (maxArchives > 0 && tarFiles.size() > maxArchives) {
            tarFiles = tarFiles.subList(0, maxArchives);
        }

        Map<String, List<RawFrame>> sequenceToFrames = new LinkedHashMap<>();
        Map<String, Integer> exclusionReasons = new LinkedHashMap<>();
        int totalFrames = 0;
        int totalMatched = 0;
        int excludedAnnotations = 0;

        for (Path tarPath : tarFiles) {
            String archiveName = tarPath.getFileName().toString();
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
                        IOUtils.skip(tis, entry.getSize());
                        continue;
                    }

                    String stem = extractStem(name);
                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                        jpgEntries.put(stem, name);
                        IOUtils.skip(tis, entry.getSize());
                    } else if (lower.endsWith(".json")) {
                        byte[] bytes = tis.readAllBytes();
                        jsonBytesMap.put(stem, bytes);
                    } else {
                        IOUtils.skip(tis, entry.getSize());
                    }
                }
            }

            Set<String> allStems = new TreeSet<>(jpgEntries.keySet());
            for (String stem : allStems) {
                totalFrames++;
                if (!jsonBytesMap.containsKey(stem)) {
                    exclusionReasons.put("MISSING_JSON_ANNOTATION", exclusionReasons.getOrDefault("MISSING_JSON_ANNOTATION", 0) + 1);
                    continue;
                }
                totalMatched++;

                byte[] jsonBytes = jsonBytesMap.get(stem);
                List<BoundingBox> signBoxes = new ArrayList<>();

                try {
                    JsonNode root = objectMapper.readTree(new ByteArrayInputStream(jsonBytes));
                    JsonNode labelsNode = root.get("labels");
                    if (labelsNode != null && labelsNode.isArray()) {
                        for (JsonNode labelNode : labelsNode) {
                            String category = labelNode.hasNonNull("category") ? labelNode.get("category").asText() : "";
                            if (!"traffic sign".equalsIgnoreCase(category.trim())) {
                                continue;
                            }

                            JsonNode boxNode = labelNode.get("box2d");
                            if (boxNode == null || !boxNode.isObject()) {
                                exclusionReasons.put("MISSING_BOX2D", exclusionReasons.getOrDefault("MISSING_BOX2D", 0) + 1);
                                excludedAnnotations++;
                                continue;
                            }

                            double x1 = boxNode.hasNonNull("x1") ? boxNode.get("x1").asDouble() : -1.0;
                            double y1 = boxNode.hasNonNull("y1") ? boxNode.get("y1").asDouble() : -1.0;
                            double x2 = boxNode.hasNonNull("x2") ? boxNode.get("x2").asDouble() : -1.0;
                            double y2 = boxNode.hasNonNull("y2") ? boxNode.get("y2").asDouble() : -1.0;

                            if (x1 < 0.0 || y1 < 0.0 || x2 <= x1 || y2 <= y1) {
                                exclusionReasons.put("DEGENERATE_OR_NEGATIVE_BOX", exclusionReasons.getOrDefault("DEGENERATE_OR_NEGATIVE_BOX", 0) + 1);
                                excludedAnnotations++;
                                continue;
                            }

                            double clampedX1 = Math.min(x1, RoadDrivingTarReader.DEFAULT_FRAME_WIDTH - 1);
                            double clampedY1 = Math.min(y1, RoadDrivingTarReader.DEFAULT_FRAME_HEIGHT - 1);
                            double clampedX2 = Math.min(x2, RoadDrivingTarReader.DEFAULT_FRAME_WIDTH);
                            double clampedY2 = Math.min(y2, RoadDrivingTarReader.DEFAULT_FRAME_HEIGHT);

                            if (clampedX2 <= clampedX1 || clampedY2 <= clampedY1) {
                                exclusionReasons.put("OUT_OF_IMAGE_BOUNDS", exclusionReasons.getOrDefault("OUT_OF_IMAGE_BOUNDS", 0) + 1);
                                excludedAnnotations++;
                                continue;
                            }

                            signBoxes.add(BoundingBox.ofPixels(clampedX1, clampedY1, clampedX2, clampedY2, 0, "traffic sign"));
                        }
                    }
                } catch (Exception ex) {
                    exclusionReasons.put("JSON_PARSE_ERROR", exclusionReasons.getOrDefault("JSON_PARSE_ERROR", 0) + 1);
                }

                String seqId = parseSequenceId(stem);
                int frameIdx = parseFrameIndex(stem);

                sequenceToFrames.computeIfAbsent(seqId, k -> new ArrayList<>())
                        .add(new RawFrame(stem, archiveName, seqId, frameIdx, signBoxes));
            }
        }

        List<String> sequences = new ArrayList<>(sequenceToFrames.keySet());
        Collections.sort(sequences);
        Random rng = new Random(randomSeed);
        Collections.shuffle(sequences, rng);

        int totalSeqs = sequences.size();
        int valSeqCount = (int) Math.round(totalSeqs * valRatio);
        if (totalSeqs > 1 && valSeqCount == 0) valSeqCount = 1;
        if (totalSeqs > 1 && valSeqCount >= totalSeqs) valSeqCount = totalSeqs - 1;

        Set<String> trainSeqSet = new HashSet<>();
        Set<String> valSeqSet = new HashSet<>();

        for (int i = 0; i < totalSeqs; i++) {
            String sid = sequences.get(i);
            if (i < totalSeqs - valSeqCount) {
                trainSeqSet.add(sid);
            } else {
                valSeqSet.add(sid);
            }
        }

        List<DetectionSampleRecord> allRecords = new ArrayList<>();
        int trainFrames = 0;
        int valFrames = 0;
        int trainSignBoxes = 0;
        int valSignBoxes = 0;
        int trainWithSign = 0;
        int valWithSign = 0;
        int trainEmpty = 0;
        int valEmpty = 0;
        Map<String, Integer> sequenceDist = new TreeMap<>();

        for (Map.Entry<String, List<RawFrame>> entry : sequenceToFrames.entrySet()) {
            String seqId = entry.getKey();
            SplitPartition part = valSeqSet.contains(seqId) ? SplitPartition.VALIDATION : SplitPartition.TRAIN;
            sequenceDist.put(seqId, entry.getValue().size());

            for (RawFrame rf : entry.getValue()) {
                boolean hasSign = !rf.boxes.isEmpty();
                if (part == SplitPartition.TRAIN) {
                    trainFrames++;
                    trainSignBoxes += rf.boxes.size();
                    if (hasSign) trainWithSign++;
                    else trainEmpty++;
                } else {
                    valFrames++;
                    valSignBoxes += rf.boxes.size();
                    if (hasSign) valWithSign++;
                    else valEmpty++;
                }

                allRecords.add(new DetectionSampleRecord(
                        rf.stem,
                        rf.archiveName,
                        rf.sequenceId,
                        rf.frameIndex,
                        RoadDrivingTarReader.DEFAULT_FRAME_WIDTH,
                        RoadDrivingTarReader.DEFAULT_FRAME_HEIGHT,
                        Collections.unmodifiableList(rf.boxes),
                        hasSign,
                        part
                ));
            }
        }

        return new ManifestSummary(
                tarFiles.size(),
                totalFrames,
                totalMatched,
                totalSeqs,
                trainSeqSet.size(),
                valSeqSet.size(),
                trainFrames,
                valFrames,
                trainSignBoxes,
                valSignBoxes,
                trainWithSign,
                valWithSign,
                trainEmpty,
                valEmpty,
                excludedAnnotations,
                Collections.unmodifiableMap(exclusionReasons),
                Collections.unmodifiableMap(sequenceDist),
                Collections.unmodifiableList(allRecords)
        );
    }

    public static byte[] extractImageBytes(Path sampleDir, String archiveName, String frameStem) throws IOException {
        Path tarPath = sampleDir.resolve(archiveName);
        if (!Files.exists(tarPath)) {
            Path addPath = sampleDir.resolve("additional-shards").resolve(archiveName);
            if (Files.exists(addPath)) {
                tarPath = addPath;
            } else {
                throw new IllegalArgumentException("Archive does not exist: " + tarPath);
            }
        }

        try (InputStream fis = Files.newInputStream(tarPath);
             BufferedInputStream bis = new BufferedInputStream(fis);
             TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {

            TarArchiveEntry entry;
            while ((entry = tis.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String name = entry.getName();
                String stem = extractStem(name);
                String lower = name.toLowerCase();
                if (stem.equals(frameStem) && (lower.endsWith(".jpg") || lower.endsWith(".jpeg"))) {
                    return tis.readAllBytes();
                } else {
                    IOUtils.skip(tis, entry.getSize());
                }
            }
        }
        throw new IOException("Image frame " + frameStem + " not found in " + archiveName);
    }

    public static String parseSequenceId(String stem) {
        int idx = stem.lastIndexOf('_');
        return idx > 0 ? stem.substring(0, idx) : stem;
    }

    public static int parseFrameIndex(String stem) {
        int idx = stem.lastIndexOf('_');
        if (idx > 0 && idx < stem.length() - 1) {
            try {
                return Integer.parseInt(stem.substring(idx + 1));
            } catch (NumberFormatException ignored) {
            }
        }
        return 0;
    }

    private static String extractStem(String path) {
        String filename = path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path;
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private record RawFrame(
            String stem,
            String archiveName,
            String sequenceId,
            int frameIndex,
            List<BoundingBox> boxes
    ) {
    }
}
