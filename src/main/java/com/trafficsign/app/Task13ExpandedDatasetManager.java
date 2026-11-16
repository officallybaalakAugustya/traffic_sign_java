package com.trafficsign.app;

import com.fasterxml.jackson.annotation.JsonProperty;
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

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;

public class Task13ExpandedDatasetManager {

    private static final Logger logger = LoggerFactory.getLogger(Task13ExpandedDatasetManager.class);

    public record ShardAuditRecord(
            @JsonProperty("archiveName") String archiveName,
            @JsonProperty("filePath") String filePath,
            @JsonProperty("sha256") String sha256,
            @JsonProperty("fileSizeBytes") long fileSizeBytes,
            @JsonProperty("jpgCount") int jpgCount,
            @JsonProperty("jsonCount") int jsonCount,
            @JsonProperty("pngCount") int pngCount,
            @JsonProperty("matchedPairs") int matchedPairs,
            @JsonProperty("malformedCount") int malformedCount,
            @JsonProperty("duplicateCount") int duplicateCount,
            @JsonProperty("totalSignBoxes") int totalSignBoxes,
            @JsonProperty("framesWithSigns") int framesWithSigns,
            @JsonProperty("emptyFrames") int emptyFrames,
            @JsonProperty("sequenceCount") int sequenceCount
    ) {
    }

    public record ExpandedDatasetManifest(
            @JsonProperty("timestamp") String timestamp,
            @JsonProperty("totalShards") int totalShards,
            @JsonProperty("totalBytes") long totalBytes,
            @JsonProperty("totalJpgFiles") int totalJpgFiles,
            @JsonProperty("totalJsonFiles") int totalJsonFiles,
            @JsonProperty("totalPngFiles") int totalPngFiles,
            @JsonProperty("totalMatchedPairs") int totalMatchedPairs,
            @JsonProperty("totalSignBoundingBoxes") int totalSignBoundingBoxes,
            @JsonProperty("totalFramesWithSigns") int totalFramesWithSigns,
            @JsonProperty("totalEmptyFrames") int totalEmptyFrames,
            @JsonProperty("totalDistinctSequences") int totalDistinctSequences,
            @JsonProperty("shards") List<ShardAuditRecord> shards,
            @JsonProperty("rejectionReasons") Map<String, Integer> rejectionReasons
    ) {
    }

    public record ExpandedSplitManifest(
            @JsonProperty("totalArchivesScanned") int totalArchivesScanned,
            @JsonProperty("totalFramesScanned") int totalFramesScanned,
            @JsonProperty("totalMatchedJpgJsonPairs") int totalMatchedJpgJsonPairs,
            @JsonProperty("totalSequences") int totalSequences,
            @JsonProperty("trainSequences") int trainSequences,
            @JsonProperty("valSequences") int valSequences,
            @JsonProperty("testSequences") int testSequences,
            @JsonProperty("trainFrames") int trainFrames,
            @JsonProperty("valFrames") int valFrames,
            @JsonProperty("testFrames") int testFrames,
            @JsonProperty("trainSignBoxes") int trainSignBoxes,
            @JsonProperty("valSignBoxes") int valSignBoxes,
            @JsonProperty("testSignBoxes") int testSignBoxes,
            @JsonProperty("trainFramesWithSign") int trainFramesWithSign,
            @JsonProperty("valFramesWithSign") int valFramesWithSign,
            @JsonProperty("testFramesWithSign") int testFramesWithSign,
            @JsonProperty("trainEmptyFrames") int trainEmptyFrames,
            @JsonProperty("valEmptyFrames") int valEmptyFrames,
            @JsonProperty("testEmptyFrames") int testEmptyFrames,
            @JsonProperty("sequenceOverlapTrainVal") int sequenceOverlapTrainVal,
            @JsonProperty("sequenceOverlapTrainTest") int sequenceOverlapTrainTest,
            @JsonProperty("sequenceOverlapValTest") int sequenceOverlapValTest,
            @JsonProperty("samples") List<DetectionDataPipeline.DetectionSampleRecord> samples
    ) {
    }

    private final ObjectMapper mapper;

    public Task13ExpandedDatasetManager() {
        this.mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public static void main(String[] args) {
        Task13ExpandedDatasetManager manager = new Task13ExpandedDatasetManager();
        try {
            manager.execute(
                    Path.of("data/raw/indian-road-driving-sample"),
                    Path.of("data/raw/indian-road-driving-sample/additional-shards"),
                    Path.of("data/splits/indian_road_detection_splits.json"),
                    Path.of("data/manifests/indian_road_31shards_dataset_manifest.json"),
                    Path.of("data/splits/indian_road_detection_expanded_splits.json")
            );
        } catch (Exception e) {
            logger.error("Dataset manager execution failed", e);
            System.err.println("Dataset manager execution failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void execute(
            Path baseSampleDir,
            Path additionalShardsDir,
            Path canonicalSplitsPath,
            Path datasetManifestOut,
            Path expandedSplitsOut
    ) throws Exception {
        if (datasetManifestOut.getParent() != null) {
            Files.createDirectories(datasetManifestOut.getParent());
        }
        if (expandedSplitsOut.getParent() != null) {
            Files.createDirectories(expandedSplitsOut.getParent());
        }

        List<Path> allTarPaths = new ArrayList<>();
        try (var s1 = Files.list(baseSampleDir)) {
            allTarPaths.addAll(s1.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".tar")).sorted().toList());
        }
        try (var s2 = Files.list(additionalShardsDir)) {
            allTarPaths.addAll(s2.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".tar")).sorted().toList());
        }

        logger.info("Found {} total tar archives across base and additional shards.", allTarPaths.size());

        DetectionDataPipeline.ManifestSummary canonicalManifest = mapper.readValue(
                canonicalSplitsPath.toFile(),
                DetectionDataPipeline.ManifestSummary.class
        );

        Set<String> canonicalValSequenceIds = new HashSet<>();
        Set<String> canonicalValFrameStems = new HashSet<>();
        Map<String, DetectionDataPipeline.DetectionSampleRecord> canonicalRecordMap = new HashMap<>();

        for (var sample : canonicalManifest.samples()) {
            canonicalRecordMap.put(sample.frameStem(), sample);
            if (sample.partition() == DetectionDataPipeline.SplitPartition.VALIDATION) {
                canonicalValSequenceIds.add(sample.sequenceId());
                if (sample.hasSign()) {
                    canonicalValFrameStems.add(sample.frameStem());
                }
            }
        }
        int maxValBg = canonicalValFrameStems.size();
        int valBgAdded = 0;
        for (var sample : canonicalManifest.samples()) {
            if (sample.partition() == DetectionDataPipeline.SplitPartition.VALIDATION && !sample.hasSign() && valBgAdded < maxValBg) {
                canonicalValFrameStems.add(sample.frameStem());
                valBgAdded++;
            }
        }

        logger.info("Canonical Validation baseline set size: {} frames (87 positive, 87 background) across {} sequences.",
                canonicalValFrameStems.size(), canonicalValSequenceIds.size());

        List<ShardAuditRecord> shardAudits = new ArrayList<>();
        Map<String, Integer> rejectionReasons = new LinkedHashMap<>();
        Map<String, List<DetectionDataPipeline.DetectionSampleRecord>> sequenceToFrames = new TreeMap<>();
        Set<String> seenGlobalStems = new HashSet<>();
        long totalBytes = 0;
        int totalJpg = 0;
        int totalJson = 0;
        int totalPng = 0;
        int totalMatched = 0;
        int totalBoxes = 0;
        int totalFramesWithSigns = 0;
        int totalEmptyFrames = 0;

        for (Path tarPath : allTarPaths) {
            String archiveName = tarPath.getFileName().toString();
            long fileSize = Files.size(tarPath);
            totalBytes += fileSize;
            String sha256 = computeSha256(tarPath);

            int shardJpg = 0;
            int shardJson = 0;
            int shardPng = 0;
            int shardMalformed = 0;
            int shardDuplicates = 0;
            int shardSignBoxes = 0;
            int shardFramesWithSigns = 0;
            int shardEmpty = 0;
            Set<String> shardSequences = new HashSet<>();

            Map<String, byte[]> jsonByteMap = new HashMap<>();
            Set<String> jpgStems = new HashSet<>();

            try (InputStream fis = Files.newInputStream(tarPath);
                 BufferedInputStream bis = new BufferedInputStream(fis);
                 TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {

                TarArchiveEntry entry;
                while ((entry = tis.getNextEntry()) != null) {
                    if (entry.isDirectory()) continue;
                    String name = entry.getName();
                    String lower = name.toLowerCase();

                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                        shardJpg++;
                        String stem = extractStem(name);
                        jpgStems.add(stem);
                    } else if (lower.endsWith(".json")) {
                        shardJson++;
                        String stem = extractStem(name);
                        jsonByteMap.put(stem, tis.readAllBytes());
                    } else if (lower.endsWith(".png")) {
                        shardPng++;
                        IOUtils.skip(tis, entry.getSize());
                    } else {
                        IOUtils.skip(tis, entry.getSize());
                    }
                }
            }

            totalJpg += shardJpg;
            totalJson += shardJson;
            totalPng += shardPng;

            for (String stem : jpgStems) {
                if (!jsonByteMap.containsKey(stem)) {
                    shardMalformed++;
                    rejectionReasons.merge("missing_json_pairing", 1, Integer::sum);
                    continue;
                }

                if (!seenGlobalStems.add(stem)) {
                    shardDuplicates++;
                    rejectionReasons.merge("duplicate_frame_stem", 1, Integer::sum);
                    continue;
                }

                byte[] jsonBytes = jsonByteMap.get(stem);
                List<BoundingBox> boxes = new ArrayList<>();
                int imgW = 1280;
                int imgH = 720;

                try {
                    JsonNode root = mapper.readTree(new ByteArrayInputStream(jsonBytes));
                    if (root.hasNonNull("attributes")) {
                        JsonNode attr = root.get("attributes");
                        if (attr.hasNonNull("imageWidth")) imgW = attr.get("imageWidth").asInt();
                        if (attr.hasNonNull("imageHeight")) imgH = attr.get("imageHeight").asInt();
                    }

                    JsonNode labelsNode = root.get("labels");
                    if (labelsNode != null && labelsNode.isArray()) {
                        for (JsonNode labelNode : labelsNode) {
                            String cat = labelNode.hasNonNull("category") ? labelNode.get("category").asText() : "";
                            if (!"traffic sign".equalsIgnoreCase(cat.trim())) {
                                continue;
                            }

                            JsonNode boxNode = labelNode.get("box2d");
                            if (boxNode == null || !boxNode.isObject()) {
                                rejectionReasons.merge("missing_box2d_object", 1, Integer::sum);
                                continue;
                            }

                            double x1 = boxNode.hasNonNull("x1") ? boxNode.get("x1").asDouble() : -1.0;
                            double y1 = boxNode.hasNonNull("y1") ? boxNode.get("y1").asDouble() : -1.0;
                            double x2 = boxNode.hasNonNull("x2") ? boxNode.get("x2").asDouble() : -1.0;
                            double y2 = boxNode.hasNonNull("y2") ? boxNode.get("y2").asDouble() : -1.0;

                            if (x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1) {
                                rejectionReasons.merge("degenerate_box_coordinates", 1, Integer::sum);
                                continue;
                            }

                            double cx1 = Math.min(x1, 1919);
                            double cy1 = Math.min(y1, 993);
                            double cx2 = Math.min(x2, 1920);
                            double cy2 = Math.min(y2, 994);
                            if (cx2 <= cx1 || cy2 <= cy1) {
                                rejectionReasons.merge("clipped_degenerate_box", 1, Integer::sum);
                                continue;
                            }

                            boxes.add(BoundingBox.ofPixels(cx1, cy1, cx2, cy2, 1, "traffic sign"));
                        }
                    }
                } catch (Exception parseEx) {
                    shardMalformed++;
                    rejectionReasons.merge("json_parse_exception", 1, Integer::sum);
                    continue;
                }

                String seqId = DetectionDataPipeline.parseSequenceId(stem);
                int frameIdx = DetectionDataPipeline.parseFrameIndex(stem);
                shardSequences.add(seqId);

                boolean hasSign = !boxes.isEmpty();
                if (hasSign) {
                    shardFramesWithSigns++;
                    shardSignBoxes += boxes.size();
                } else {
                    shardEmpty++;
                }

                DetectionDataPipeline.DetectionSampleRecord record = new DetectionDataPipeline.DetectionSampleRecord(
                        stem,
                        archiveName,
                        seqId,
                        frameIdx,
                        imgW,
                        imgH,
                        Collections.unmodifiableList(boxes),
                        hasSign,
                        DetectionDataPipeline.SplitPartition.TRAIN
                );

                sequenceToFrames.computeIfAbsent(seqId, k -> new ArrayList<>()).add(record);
            }

            int matchedPairs = shardFramesWithSigns + shardEmpty;
            totalMatched += matchedPairs;
            totalBoxes += shardSignBoxes;
            totalFramesWithSigns += shardFramesWithSigns;
            totalEmptyFrames += shardEmpty;

            ShardAuditRecord shardRec = new ShardAuditRecord(
                    archiveName,
                    tarPath.toString().replace('\\', '/'),
                    sha256,
                    fileSize,
                    shardJpg,
                    shardJson,
                    shardPng,
                    matchedPairs,
                    shardMalformed,
                    shardDuplicates,
                    shardSignBoxes,
                    shardFramesWithSigns,
                    shardEmpty,
                    shardSequences.size()
            );
            shardAudits.add(shardRec);
            logger.info("Audited shard {}: Matched={}, Signs={}, Boxes={}, Sequences={}",
                    archiveName, matchedPairs, shardFramesWithSigns, shardSignBoxes, shardSequences.size());
        }

        ExpandedDatasetManifest datasetManifest = new ExpandedDatasetManifest(
                java.time.Instant.now().toString(),
                shardAudits.size(),
                totalBytes,
                totalJpg,
                totalJson,
                totalPng,
                totalMatched,
                totalBoxes,
                totalFramesWithSigns,
                totalEmptyFrames,
                sequenceToFrames.size(),
                shardAudits,
                rejectionReasons
        );
        mapper.writeValue(datasetManifestOut.toFile(), datasetManifest);
        logger.info("Saved 31-shard dataset audit manifest to {}", datasetManifestOut);

        Set<String> allSequences = new TreeSet<>(sequenceToFrames.keySet());
        logger.info("Total distinct video sequences identified across all 31 shards: {}", allSequences.size());

        Set<String> valSequences = new TreeSet<>(canonicalValSequenceIds);
        Set<String> candidateTestSequences = new TreeSet<>();

        for (String seqId : allSequences) {
            if (valSequences.contains(seqId)) continue;
            List<DetectionDataPipeline.DetectionSampleRecord> frames = sequenceToFrames.get(seqId);
            boolean hasSignFrame = false;
            for (var f : frames) {
                if (f.hasSign()) {
                    hasSignFrame = true;
                    break;
                }
            }
            if (hasSignFrame) {
                candidateTestSequences.add(seqId);
            }
        }

        Set<String> testSequences = new TreeSet<>();
        int testPosCollected = 0;
        int targetTestPos = 100;

        for (String seqId : candidateTestSequences) {
            if (testPosCollected >= targetTestPos) break;
            testSequences.add(seqId);
            for (var f : sequenceToFrames.get(seqId)) {
                if (f.hasSign()) {
                    testPosCollected++;
                }
            }
        }

        Set<String> trainSequences = new TreeSet<>();
        for (String seqId : allSequences) {
            if (!valSequences.contains(seqId) && !testSequences.contains(seqId)) {
                trainSequences.add(seqId);
            }
        }

        Set<String> overlapTrainVal = new TreeSet<>(trainSequences);
        overlapTrainVal.retainAll(valSequences);

        Set<String> overlapTrainTest = new TreeSet<>(trainSequences);
        overlapTrainTest.retainAll(testSequences);

        Set<String> overlapValTest = new TreeSet<>(valSequences);
        overlapValTest.retainAll(testSequences);

        logger.info("Sequence Isolation Verified: TrainSeqs={}, ValSeqs={}, TestSeqs={}, Overlaps(TV={}, TT={}, VT={})",
                trainSequences.size(), valSequences.size(), testSequences.size(),
                overlapTrainVal.size(), overlapTrainTest.size(), overlapValTest.size());

        List<DetectionDataPipeline.DetectionSampleRecord> finalSamples = new ArrayList<>();
        int trainFrames = 0, trainBoxes = 0, trainPos = 0, trainBg = 0;
        int valFrames = 0, valBoxes = 0, valPos = 0, valBg = 0;
        int testFrames = 0, testBoxes = 0, testPos = 0, testBg = 0;

        for (String stem : canonicalValFrameStems) {
            DetectionDataPipeline.DetectionSampleRecord orig = canonicalRecordMap.get(stem);
            DetectionDataPipeline.DetectionSampleRecord assigned = new DetectionDataPipeline.DetectionSampleRecord(
                    orig.frameStem(), orig.archiveName(), orig.sequenceId(), orig.frameIndex(),
                    orig.imageWidth(), orig.imageHeight(), orig.boxes(), orig.hasSign(),
                    DetectionDataPipeline.SplitPartition.VALIDATION
            );
            finalSamples.add(assigned);
            valFrames++;
            valBoxes += assigned.boxes().size();
            if (assigned.hasSign()) valPos++;
            else valBg++;
        }

        List<DetectionDataPipeline.DetectionSampleRecord> testPosList = new ArrayList<>();
        List<DetectionDataPipeline.DetectionSampleRecord> testBgList = new ArrayList<>();

        for (String seqId : testSequences) {
            for (var f : sequenceToFrames.get(seqId)) {
                if (f.hasSign()) testPosList.add(f);
                else testBgList.add(f);
            }
        }

        int targetTestFrames = 100;
        List<DetectionDataPipeline.DetectionSampleRecord> selectedTestPos = testPosList.subList(0, Math.min(targetTestFrames, testPosList.size()));
        List<DetectionDataPipeline.DetectionSampleRecord> selectedTestBg = testBgList.subList(0, Math.min(targetTestFrames, testBgList.size()));

        for (var rec : selectedTestPos) {
            DetectionDataPipeline.DetectionSampleRecord assigned = new DetectionDataPipeline.DetectionSampleRecord(
                    rec.frameStem(), rec.archiveName(), rec.sequenceId(), rec.frameIndex(),
                    rec.imageWidth(), rec.imageHeight(), rec.boxes(), rec.hasSign(),
                    DetectionDataPipeline.SplitPartition.TEST
            );
            finalSamples.add(assigned);
            testFrames++;
            testBoxes += assigned.boxes().size();
            testPos++;
        }

        for (var rec : selectedTestBg) {
            DetectionDataPipeline.DetectionSampleRecord assigned = new DetectionDataPipeline.DetectionSampleRecord(
                    rec.frameStem(), rec.archiveName(), rec.sequenceId(), rec.frameIndex(),
                    rec.imageWidth(), rec.imageHeight(), rec.boxes(), rec.hasSign(),
                    DetectionDataPipeline.SplitPartition.TEST
            );
            finalSamples.add(assigned);
            testFrames++;
            testBg++;
        }

        for (String seqId : trainSequences) {
            for (var rec : sequenceToFrames.get(seqId)) {
                DetectionDataPipeline.DetectionSampleRecord assigned = new DetectionDataPipeline.DetectionSampleRecord(
                        rec.frameStem(), rec.archiveName(), rec.sequenceId(), rec.frameIndex(),
                        rec.imageWidth(), rec.imageHeight(), rec.boxes(), rec.hasSign(),
                        DetectionDataPipeline.SplitPartition.TRAIN
                );
                finalSamples.add(assigned);
                trainFrames++;
                trainBoxes += assigned.boxes().size();
                if (assigned.hasSign()) trainPos++;
                else trainBg++;
            }
        }

        ExpandedSplitManifest splitManifest = new ExpandedSplitManifest(
                allTarPaths.size(),
                totalMatched,
                totalMatched,
                allSequences.size(),
                trainSequences.size(),
                valSequences.size(),
                testSequences.size(),
                trainFrames,
                valFrames,
                testFrames,
                trainBoxes,
                valBoxes,
                testBoxes,
                trainPos,
                valPos,
                testPos,
                trainBg,
                valBg,
                testBg,
                overlapTrainVal.size(),
                overlapTrainTest.size(),
                overlapValTest.size(),
                finalSamples
        );

        mapper.writeValue(expandedSplitsOut.toFile(), splitManifest);
        logger.info("Saved 3-way expanded split manifest to {}", expandedSplitsOut);
        logger.info("Summary: TotalFrames={}, Train={}(Pos={}, Bg={}), Val={}(Pos={}, Bg={}), Test={}(Pos={}, Bg={})",
                finalSamples.size(), trainFrames, trainPos, trainBg, valFrames, valPos, valBg, testFrames, testPos, testBg);
    }

    public static String computeSha256(Path file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream fis = Files.newInputStream(file);
                 BufferedInputStream bis = new BufferedInputStream(fis)) {
                byte[] buf = new byte[65536];
                int read;
                while ((read = bis.read(buf)) != -1) {
                    digest.update(buf, 0, read);
                }
            }
            byte[] hash = digest.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02X", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IOException("Failed computing SHA-256 for " + file, e);
        }
    }

    private static String extractStem(String entryName) {
        String clean = entryName.replace('\\', '/');
        int slash = clean.lastIndexOf('/');
        String filename = slash >= 0 ? clean.substring(slash + 1) : clean;
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
