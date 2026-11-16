package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class DetectionUtils {

    public record DetectionPrediction(
            BoundingBox box,
            float confidence,
            int gridRow,
            int gridCol
    ) {
    }

    public static float[] boxToTarget(BoundingBox box, int gridCols, int gridRows) {
        Objects.requireNonNull(box, "box must not be null");
        double bx = (box.x1() + box.x2()) / 2.0;
        double by = (box.y1() + box.y2()) / 2.0;
        double bw = box.x2() - box.x1();
        double bh = box.y2() - box.y1();

        int col = (int) Math.floor(bx * gridCols);
        int row = (int) Math.floor(by * gridRows);

        col = Math.max(0, Math.min(col, gridCols - 1));
        row = Math.max(0, Math.min(row, gridRows - 1));

        double cellX = bx * gridCols - col;
        double cellY = by * gridRows - row;

        return new float[]{
                (float) cellX,
                (float) cellY,
                (float) bw,
                (float) bh,
                (float) row,
                (float) col
        };
    }

    public static BoundingBox targetToBox(
            float cellX,
            float cellY,
            float bw,
            float bh,
            int row,
            int col,
            int gridCols,
            int gridRows
    ) {
        double bx = (col + cellX) / (double) gridCols;
        double by = (row + cellY) / (double) gridRows;

        double x1 = Math.max(0.0, bx - bw / 2.0);
        double y1 = Math.max(0.0, by - bh / 2.0);
        double x2 = Math.min(1.0, bx + bw / 2.0);
        double y2 = Math.min(1.0, by + bh / 2.0);

        if (x2 <= x1) x2 = Math.min(1.0, x1 + 0.01);
        if (y2 <= y1) y2 = Math.min(1.0, y1 + 0.01);

        return BoundingBox.ofNormalized(x1, y1, x2, y2, 0, "traffic sign");
    }

    public static List<DetectionPrediction> decodeGridPredictions(
            float[][][] gridTensor,
            float confidenceThreshold,
            int gridCols,
            int gridRows
    ) {
        return decodeGridPredictions(gridTensor, confidenceThreshold, gridCols, gridRows, false, false);
    }

    public static List<DetectionPrediction> decodeGridPredictions(
            float[][][] gridTensor,
            float confidenceThreshold,
            int gridCols,
            int gridRows,
            boolean clampCoordinatesDirectly,
            boolean useLinearConfidence
    ) {
        List<DetectionPrediction> rawDetections = new ArrayList<>();

        for (int r = 0; r < gridRows; r++) {
            for (int c = 0; c < gridCols; c++) {
                float confLogit = gridTensor[0][r][c];
                float conf = useLinearConfidence
                        ? Math.max(0.0f, Math.min(1.0f, confLogit))
                        : sigmoid(confLogit);

                if (conf >= confidenceThreshold) {
                    float rawCellX = gridTensor[1][r][c];
                    float rawCellY = gridTensor[2][r][c];

                    float cellX = clampCoordinatesDirectly
                            ? Math.max(0.0f, Math.min(1.0f, rawCellX))
                            : sigmoid(rawCellX);
                    float cellY = clampCoordinatesDirectly
                            ? Math.max(0.0f, Math.min(1.0f, rawCellY))
                            : sigmoid(rawCellY);

                    float bw = (float) Math.exp(gridTensor[3][r][c]) / (float) gridCols;
                    float bh = (float) Math.exp(gridTensor[4][r][c]) / (float) gridRows;

                    BoundingBox box = targetToBox(cellX, cellY, bw, bh, r, c, gridCols, gridRows);
                    rawDetections.add(new DetectionPrediction(box, conf, r, c));
                }
            }
        }

        return rawDetections;
    }

    public static List<DetectionPrediction> nonMaximumSuppression(
            List<DetectionPrediction> detections,
            double iouThreshold
    ) {
        if (detections == null || detections.isEmpty()) {
            return List.of();
        }

        List<DetectionPrediction> sorted = new ArrayList<>(detections);
        sorted.sort(Comparator.comparingDouble((DetectionPrediction d) -> d.confidence()).reversed());

        List<DetectionPrediction> selected = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            DetectionPrediction current = sorted.get(i);
            selected.add(current);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                DetectionPrediction candidate = sorted.get(j);
                double iou = current.box().computeIoU(candidate.box());
                if (iou >= iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return selected;
    }

    public static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
}
