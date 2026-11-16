package com.trafficsign.inference;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.data.ImagePreprocessor;
import com.trafficsign.data.IndianTrafficSignTaxonomy;
import com.trafficsign.data.detection.DetectionUtils;
import com.trafficsign.neural.djl.CompactSignDetectorBlock;
import com.trafficsign.neural.djl.HighResSignDetectorBlock;
import com.trafficsign.neural.djl.IndianTrafficSignCnnBlock;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class RoadSceneInferencePipeline implements AutoCloseable {

    public enum UncertaintyState {
        CONFIDENT,
        LOW_DETECTION_CONFIDENCE,
        LOW_CLASSIFICATION_CONFIDENCE,
        UNKNOWN_CLASS
    }

    public record RoadSceneDetection(
            BoundingBox absoluteBox,
            BoundingBox normalizedBox,
            float detectionConfidence,
            int predictedClassId,
            String predictedClassName,
            String category,
            String safetyImplication,
            float classificationConfidence,
            UncertaintyState uncertaintyState,
            String displayLabel
    ) {
        public RoadSceneDetection(
                BoundingBox absoluteBox,
                BoundingBox normalizedBox,
                float detectionConfidence,
                int predictedClassId,
                String predictedClassName,
                String category,
                String safetyImplication,
                float classificationConfidence,
                UncertaintyState uncertaintyState
        ) {
            this(absoluteBox, normalizedBox, detectionConfidence, predictedClassId, predictedClassName,
                    category, safetyImplication, classificationConfidence, uncertaintyState, predictedClassName);
        }
    }

    public record PipelineResult(
            int imageWidth,
            int imageHeight,
            long detectionLatencyMs,
            long classificationLatencyMs,
            List<RoadSceneDetection> detections,
            long cropLatencyMs,
            long classifierForwardLatencyMs,
            int rawCandidateCount
    ) {
        public PipelineResult(
                int imageWidth,
                int imageHeight,
                long detectionLatencyMs,
                long classificationLatencyMs,
                List<RoadSceneDetection> detections
        ) {
            this(imageWidth, imageHeight, detectionLatencyMs, classificationLatencyMs, detections, 0L, classificationLatencyMs, detections.size());
        }

        public long totalLatencyMs() {
            return detectionLatencyMs + classificationLatencyMs;
        }

        public double throughputFps() {
            long tot = totalLatencyMs();
            return tot > 0 ? (1000.0 / tot) : 0.0;
        }
    }

    private final Model detectorModel;
    private final Model classifierModel;
    private final Device device;
    private final IndianTrafficSignTaxonomy taxonomy;
    private final ImagePreprocessor classifierPreprocessor;
    private final ImagePreprocessor detectorPreprocessor;
    private final int classifierResolution;

    private float detectionConfidenceThreshold = 0.50f;
    private float classificationConfidenceThreshold = 0.25f;
    private double nmsIouThreshold = 0.40;
    private int maxProposalsPerFrame = 25;
    private boolean batchClassifierInference = true;

    public RoadSceneInferencePipeline(
            Path detectorCheckpointDir,
            Path classifierCheckpointDir,
            Device device
    ) throws IOException, MalformedModelException {
        this(
                detectorCheckpointDir,
                detectModelName(detectorCheckpointDir, "sign_detector"),
                detectDetectorBlock(detectorCheckpointDir),
                classifierCheckpointDir,
                detectModelName(classifierCheckpointDir, "exp_b_48x48"),
                detectResolution(classifierCheckpointDir, 48),
                device
        );
    }

    public RoadSceneInferencePipeline(
            Path detectorCheckpointDir,
            String detectorModelName,
            Path classifierCheckpointDir,
            String classifierModelName,
            int classifierResolution,
            Device device
    ) throws IOException, MalformedModelException {
        this(
                detectorCheckpointDir,
                detectorModelName,
                detectDetectorBlock(detectorCheckpointDir),
                classifierCheckpointDir,
                classifierModelName,
                classifierResolution,
                device
        );
    }

    public RoadSceneInferencePipeline(
            Path detectorCheckpointDir,
            String detectorModelName,
            Block detectorBlock,
            Path classifierCheckpointDir,
            String classifierModelName,
            int classifierResolution,
            Device device
    ) throws IOException, MalformedModelException {
        this.device = Objects.requireNonNull(device, "device must not be null");
        this.taxonomy = new IndianTrafficSignTaxonomy();
        this.classifierResolution = classifierResolution;
        this.classifierPreprocessor = new ImagePreprocessor(classifierResolution, classifierResolution, false);
        this.detectorPreprocessor = new ImagePreprocessor(128, 128, false);

        Block blockToUse = detectorBlock != null ? detectorBlock : detectDetectorBlock(detectorCheckpointDir);
        this.detectorModel = Model.newInstance("sign_detector_pipeline", device);
        this.detectorModel.setBlock(blockToUse);
        this.detectorModel.load(detectorCheckpointDir, detectorModelName);

        this.classifierModel = Model.newInstance("sign_classifier_pipeline", device);
        this.classifierModel.setBlock(IndianTrafficSignCnnBlock.build(85));
        this.classifierModel.load(classifierCheckpointDir, classifierModelName);
    }

    public static String detectModelName(Path dir, String fallback) {
        if (dir == null || !Files.exists(dir)) return fallback;
        try (var stream = Files.list(dir)) {
            var params = stream.filter(p -> p.getFileName().toString().endsWith(".params")).toList();
            if (!params.isEmpty()) {
                String name = params.get(0).getFileName().toString();
                int idx = name.lastIndexOf('-');
                if (idx > 0) return name.substring(0, idx);
            }
        } catch (Exception ignored) {
        }
        return fallback;
    }

    public static Block detectDetectorBlock(Path dir) {
        if (dir == null || !Files.exists(dir)) return CompactSignDetectorBlock.build();
        String pathStr = dir.toString().toLowerCase();
        if (pathStr.contains("16x16") || pathStr.contains("highres") || pathStr.contains("exp_b") || pathStr.contains("exp_c")
                || pathStr.contains("task11") || pathStr.contains("task13") || pathStr.contains("task14")
                || pathStr.contains("task15") || pathStr.contains("task16") || pathStr.contains("box16")) {
            return HighResSignDetectorBlock.build();
        }
        try (var stream = Files.list(dir)) {
            var params = stream.filter(p -> p.getFileName().toString().endsWith(".params")).toList();
            if (!params.isEmpty()) {
                String name = params.get(0).getFileName().toString().toLowerCase();
                if (name.contains("16x16") || name.contains("highres") || name.contains("task11")
                        || name.contains("task13") || name.contains("task14") || name.contains("task15")
                        || name.contains("task16") || name.contains("box16")) {
                    return HighResSignDetectorBlock.build();
                }
            }
        } catch (Exception ignored) {
        }
        return CompactSignDetectorBlock.build();
    }

    public static int detectResolution(Path dir, int fallback) {
        if (dir == null || !Files.exists(dir)) return fallback;
        try (var stream = Files.list(dir)) {
            var params = stream.filter(p -> p.getFileName().toString().endsWith(".params")).toList();
            if (!params.isEmpty()) {
                String name = params.get(0).getFileName().toString().toLowerCase();
                if (name.contains("48")) return 48;
                if (name.contains("64")) return 64;
                if (name.contains("32")) return 32;
                if (name.contains("indian_cnn")) return 32;
            }
        } catch (Exception ignored) {
        }
        return fallback;
    }

    public void setDetectionConfidenceThreshold(float threshold) {
        this.detectionConfidenceThreshold = threshold;
    }

    public void setClassificationConfidenceThreshold(float threshold) {
        this.classificationConfidenceThreshold = threshold;
    }

    public void setNmsIouThreshold(double threshold) {
        this.nmsIouThreshold = threshold;
    }

    public void setMaxProposalsPerFrame(int maxProposals) {
        this.maxProposalsPerFrame = Math.max(1, maxProposals);
    }

    public void setBatchClassifierInference(boolean batchClassifierInference) {
        this.batchClassifierInference = batchClassifierInference;
    }

    public float getDetectionConfidenceThreshold() {
        return detectionConfidenceThreshold;
    }

    public float getClassificationConfidenceThreshold() {
        return classificationConfidenceThreshold;
    }

    public double getNmsIouThreshold() {
        return nmsIouThreshold;
    }

    public int getMaxProposalsPerFrame() {
        return maxProposalsPerFrame;
    }

    public boolean isBatchClassifierInference() {
        return batchClassifierInference;
    }

    public int getClassifierResolution() {
        return classifierResolution;
    }

    public PipelineResult processImage(BufferedImage fullFrame) {
        Objects.requireNonNull(fullFrame, "fullFrame must not be null");
        int width = fullFrame.getWidth();
        int height = fullFrame.getHeight();

        long detStart = System.currentTimeMillis();
        float[][][] detTensor = detectorPreprocessor.toTensorCHW(fullFrame);
        float[] flatDetPixels = flatten(detTensor);

        List<DetectionUtils.DetectionPrediction> detections;
        try (NDManager manager = NDManager.newBaseManager(device)) {
            ParameterStore ps = new ParameterStore(manager, false);
            NDArray x = manager.create(flatDetPixels, new Shape(1, 3, 128, 128));
            NDArray out = detectorModel.getBlock().forward(ps, new NDList(x), false).head();

            Shape outShape = out.getShape();
            int gridRows = (int) outShape.get(outShape.dimension() - 2);
            int gridCols = (int) outShape.get(outShape.dimension() - 1);

            float[] flatOut = out.toFloatArray();
            float[][][] gridTensor = new float[5][gridRows][gridCols];
            int idx = 0;
            for (int ch = 0; ch < 5; ch++) {
                for (int r = 0; r < gridRows; r++) {
                    for (int c = 0; c < gridCols; c++) {
                        gridTensor[ch][r][c] = flatOut[idx++];
                    }
                }
            }

            List<DetectionUtils.DetectionPrediction> raw = DetectionUtils.decodeGridPredictions(
                    gridTensor, detectionConfidenceThreshold, gridCols, gridRows
            );
            detections = DetectionUtils.nonMaximumSuppression(raw, nmsIouThreshold);
        }
        long detDuration = System.currentTimeMillis() - detStart;

        int rawCandidatesCount = detections.size();
        if (detections.size() > maxProposalsPerFrame) {
            List<DetectionUtils.DetectionPrediction> mutable = new ArrayList<>(detections);
            mutable.sort(Comparator.comparingDouble((DetectionUtils.DetectionPrediction d) -> d.confidence()).reversed());
            detections = mutable.subList(0, maxProposalsPerFrame);
        }

        long clsStart = System.currentTimeMillis();
        long cropDuration = 0L;
        long fwdDuration = 0L;
        List<RoadSceneDetection> resultDetections = new ArrayList<>();

        if (!detections.isEmpty()) {
            if (batchClassifierInference) {
                record CropCandidate(
                        BoundingBox absBox,
                        BoundingBox normBox,
                        float detConfidence,
                        float[] pixels
                ) {
                }

                long cropT0 = System.currentTimeMillis();
                List<CropCandidate> validCandidates = new ArrayList<>();

                for (var det : detections) {
                    BoundingBox normBox = det.box();
                    BoundingBox absBox = normBox.toAbsolute(width, height);

                    int cropX1 = Math.max(0, Math.min((int) Math.round(absBox.x1()), width - 1));
                    int cropY1 = Math.max(0, Math.min((int) Math.round(absBox.y1()), height - 1));
                    int cropX2 = Math.max(cropX1 + 1, Math.min((int) Math.round(absBox.x2()), width));
                    int cropY2 = Math.max(cropY1 + 1, Math.min((int) Math.round(absBox.y2()), height));

                    int cropW = cropX2 - cropX1;
                    int cropH = cropY2 - cropY1;
                    if (cropW <= 2 || cropH <= 2) {
                        continue;
                    }

                    BufferedImage cropped = fullFrame.getSubimage(cropX1, cropY1, cropW, cropH);
                    BufferedImage rgbCrop = new BufferedImage(cropW, cropH, BufferedImage.TYPE_INT_RGB);
                    Graphics2D gCrop = rgbCrop.createGraphics();
                    gCrop.drawImage(cropped, 0, 0, null);
                    gCrop.dispose();

                    float[][][] cropTensor = classifierPreprocessor.toTensorCHW(rgbCrop);
                    float[] flatCropPixels = flatten(cropTensor);
                    validCandidates.add(new CropCandidate(absBox, normBox, det.confidence(), flatCropPixels));
                }
                cropDuration = System.currentTimeMillis() - cropT0;

                int k = validCandidates.size();
                if (k > 0) {
                    long fwdT0 = System.currentTimeMillis();
                    int pixelsPerCrop = 3 * classifierResolution * classifierResolution;
                    float[] batchPixels = new float[k * pixelsPerCrop];
                    for (int i = 0; i < k; i++) {
                        System.arraycopy(validCandidates.get(i).pixels(), 0, batchPixels, i * pixelsPerCrop, pixelsPerCrop);
                    }

                    try (NDManager manager = NDManager.newBaseManager(device)) {
                        ParameterStore ps = new ParameterStore(manager, false);
                        NDArray cx = manager.create(batchPixels, new Shape(k, 3, classifierResolution, classifierResolution));
                        NDArray logits = classifierModel.getBlock().forward(ps, new NDList(cx), false).head();
                        NDArray probs = logits.softmax(1);
                        float[] allProbs = probs.toFloatArray();

                        int numClasses = 85;
                        for (int i = 0; i < k; i++) {
                            CropCandidate cand = validCandidates.get(i);
                            int offset = i * numClasses;

                            int maxClassId = 0;
                            float maxProb = allProbs[offset];
                            for (int c = 1; c < numClasses; c++) {
                                float prob = allProbs[offset + c];
                                if (prob > maxProb) {
                                    maxProb = prob;
                                    maxClassId = c;
                                }
                            }

                            UncertaintyState state;
                            if (cand.detConfidence() < detectionConfidenceThreshold) {
                                state = UncertaintyState.LOW_DETECTION_CONFIDENCE;
                            } else if (maxProb < classificationConfidenceThreshold) {
                                state = UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE;
                            } else {
                                state = UncertaintyState.CONFIDENT;
                            }

                            IndianTrafficSignTaxonomy.IndianSignClassInfo signInfo = taxonomy.getClassInfo(maxClassId).orElse(null);
                            String className = signInfo != null ? signInfo.canonicalName() : "UNKNOWN";
                            String category = signInfo != null ? signInfo.category() : "UNCERTAIN";
                            String safety = signInfo != null ? signInfo.safetyImplication() : "";

                            String displayLabel;
                            if (state == UncertaintyState.CONFIDENT) {
                                displayLabel = className;
                            } else if (state == UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE) {
                                displayLabel = "UNCERTAIN (" + className + ")";
                            } else {
                                displayLabel = "LOW_CONFIDENCE";
                            }

                            resultDetections.add(new RoadSceneDetection(
                                    cand.absBox(),
                                    cand.normBox(),
                                    cand.detConfidence(),
                                    maxClassId,
                                    className,
                                    category,
                                    safety,
                                    maxProb,
                                    state,
                                    displayLabel
                            ));
                        }
                    }
                    fwdDuration = System.currentTimeMillis() - fwdT0;
                }
            } else {
                long cropT0 = System.currentTimeMillis();
                try (NDManager manager = NDManager.newBaseManager(device)) {
                    ParameterStore ps = new ParameterStore(manager, false);

                    for (var det : detections) {
                        BoundingBox normBox = det.box();
                        BoundingBox absBox = normBox.toAbsolute(width, height);

                        int cropX1 = Math.max(0, Math.min((int) Math.round(absBox.x1()), width - 1));
                        int cropY1 = Math.max(0, Math.min((int) Math.round(absBox.y1()), height - 1));
                        int cropX2 = Math.max(cropX1 + 1, Math.min((int) Math.round(absBox.x2()), width));
                        int cropY2 = Math.max(cropY1 + 1, Math.min((int) Math.round(absBox.y2()), height));

                        int cropW = cropX2 - cropX1;
                        int cropH = cropY2 - cropY1;
                        if (cropW <= 2 || cropH <= 2) {
                            continue;
                        }

                        BufferedImage cropped = fullFrame.getSubimage(cropX1, cropY1, cropW, cropH);
                        BufferedImage rgbCrop = new BufferedImage(cropW, cropH, BufferedImage.TYPE_INT_RGB);
                        Graphics2D gCrop = rgbCrop.createGraphics();
                        gCrop.drawImage(cropped, 0, 0, null);
                        gCrop.dispose();

                        float[][][] cropTensor = classifierPreprocessor.toTensorCHW(rgbCrop);
                        float[] flatCropPixels = flatten(cropTensor);

                        long f0 = System.currentTimeMillis();
                        NDArray cx = manager.create(flatCropPixels, new Shape(1, 3, classifierResolution, classifierResolution));
                        NDArray logits = classifierModel.getBlock().forward(ps, new NDList(cx), false).head();
                        NDArray probs = logits.softmax(1);

                        float[] probArray = probs.toFloatArray();
                        fwdDuration += (System.currentTimeMillis() - f0);

                        int maxClassId = 0;
                        float maxProb = probArray[0];
                        for (int c = 1; c < probArray.length; c++) {
                            if (probArray[c] > maxProb) {
                                maxProb = probArray[c];
                                maxClassId = c;
                            }
                        }

                        UncertaintyState state;
                        if (det.confidence() < detectionConfidenceThreshold) {
                            state = UncertaintyState.LOW_DETECTION_CONFIDENCE;
                        } else if (maxProb < classificationConfidenceThreshold) {
                            state = UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE;
                        } else {
                            state = UncertaintyState.CONFIDENT;
                        }

                        IndianTrafficSignTaxonomy.IndianSignClassInfo signInfo = taxonomy.getClassInfo(maxClassId).orElse(null);
                        String className = signInfo != null ? signInfo.canonicalName() : "UNKNOWN";
                        String category = signInfo != null ? signInfo.category() : "UNCERTAIN";
                        String safety = signInfo != null ? signInfo.safetyImplication() : "";

                        String displayLabel;
                        if (state == UncertaintyState.CONFIDENT) {
                            displayLabel = className;
                        } else if (state == UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE) {
                            displayLabel = "UNCERTAIN (" + className + ")";
                        } else {
                            displayLabel = "LOW_CONFIDENCE";
                        }

                        resultDetections.add(new RoadSceneDetection(
                                absBox,
                                normBox,
                                det.confidence(),
                                maxClassId,
                                className,
                                category,
                                safety,
                                maxProb,
                                state,
                                displayLabel
                        ));
                    }
                }
                cropDuration = System.currentTimeMillis() - cropT0 - fwdDuration;
            }
        }
        long clsDuration = System.currentTimeMillis() - clsStart;

        return new PipelineResult(width, height, detDuration, clsDuration, resultDetections, cropDuration, fwdDuration, rawCandidatesCount);
    }

    public BufferedImage drawDetections(BufferedImage inputFrame, PipelineResult result) {
        Objects.requireNonNull(inputFrame, "inputFrame must not be null");
        Objects.requireNonNull(result, "result must not be null");

        int w = inputFrame.getWidth();
        int h = inputFrame.getHeight();
        BufferedImage annotated = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = annotated.createGraphics();
        try {
            g2d.drawImage(inputFrame, 0, 0, null);
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            for (RoadSceneDetection det : result.detections()) {
                BoundingBox box = det.absoluteBox();
                int bx = (int) Math.round(box.x1());
                int by = (int) Math.round(box.y1());
                int bw = (int) Math.round(box.x2() - box.x1());
                int bh = (int) Math.round(box.y2() - box.y1());

                Color boxColor;
                switch (det.uncertaintyState()) {
                    case CONFIDENT -> boxColor = new Color(0, 204, 68);
                    case LOW_CLASSIFICATION_CONFIDENCE -> boxColor = new Color(255, 153, 0);
                    case LOW_DETECTION_CONFIDENCE -> boxColor = new Color(51, 153, 255);
                    case UNKNOWN_CLASS -> boxColor = new Color(230, 40, 40);
                    default -> boxColor = Color.YELLOW;
                }

                g2d.setColor(boxColor);
                g2d.setStroke(new BasicStroke(3.0f));
                g2d.drawRect(bx, by, bw, bh);

                String text = String.format("%s (cls: %.0f%%, det: %.0f%%)",
                        det.displayLabel(),
                        det.classificationConfidence() * 100.0f,
                        det.detectionConfidence() * 100.0f);

                g2d.setFont(new Font("SansSerif", Font.BOLD, 14));
                FontMetrics fm = g2d.getFontMetrics();
                int textW = fm.stringWidth(text);
                int textH = fm.getHeight();

                int pillX = Math.max(0, Math.min(bx, w - textW - 14));
                int pillY = Math.max(textH + 6, by - 6);

                g2d.setColor(new Color(0, 0, 0, 200));
                g2d.fillRoundRect(pillX, pillY - textH + 2, textW + 12, textH + 4, 6, 6);

                g2d.setColor(boxColor);
                g2d.setStroke(new BasicStroke(1.5f));
                g2d.drawRoundRect(pillX, pillY - textH + 2, textW + 12, textH + 4, 6, 6);

                g2d.setColor(Color.WHITE);
                g2d.drawString(text, pillX + 6, pillY - 2);
            }
        } finally {
            g2d.dispose();
        }
        return annotated;
    }

    private static float[] flatten(float[][][] tensor) {
        int c = tensor.length;
        int h = tensor[0].length;
        int w = tensor[0][0].length;
        float[] flat = new float[c * h * w];
        int idx = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int r = 0; r < h; r++) {
                for (int col = 0; col < w; col++) {
                    flat[idx++] = tensor[ch][r][col];
                }
            }
        }
        return flat;
    }

    @Override
    public void close() {
        if (detectorModel != null) detectorModel.close();
        if (classifierModel != null) classifierModel.close();
    }
}
