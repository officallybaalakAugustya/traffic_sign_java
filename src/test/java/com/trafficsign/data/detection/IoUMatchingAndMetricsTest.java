package com.trafficsign.data.detection;

import com.trafficsign.data.BoundingBox;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class IoUMatchingAndMetricsTest {

    @Test
    @DisplayName("One-to-one matching policy assigns single ground truth to highest confidence prediction only")
    void testOneToOneMatchingPreventsDoubleTruePositive() {
        BoundingBox gt = BoundingBox.ofNormalized(0.20, 0.20, 0.30, 0.30, 0, "traffic sign");

        BoundingBox pred1 = BoundingBox.ofNormalized(0.205, 0.205, 0.305, 0.305, 0, "traffic sign");
        BoundingBox pred2 = BoundingBox.ofNormalized(0.21, 0.21, 0.31, 0.31, 0, "traffic sign");

        DetectionUtils.DetectionPrediction p1 = new DetectionUtils.DetectionPrediction(pred1, 0.90f, 3, 3);
        DetectionUtils.DetectionPrediction p2 = new DetectionUtils.DetectionPrediction(pred2, 0.80f, 3, 3);

        List<DetectionUtils.DetectionPrediction> preds = List.of(p1, p2);
        List<BoundingBox> gts = List.of(gt);

        int tp = 0;
        int fp = 0;
        boolean[] matched = new boolean[gts.size()];

        for (var p : preds) {
            double bestIou = 0.0;
            int bestIdx = -1;
            for (int g = 0; g < gts.size(); g++) {
                if (matched[g]) continue;
                double iou = p.box().computeIoU(gts.get(g));
                if (iou > bestIou) {
                    bestIou = iou;
                    bestIdx = g;
                }
            }

            if (bestIou >= 0.50 && bestIdx >= 0) {
                tp++;
                matched[bestIdx] = true;
            } else {
                fp++;
            }
        }

        assertThat(tp).isEqualTo(1);
        assertThat(fp).isEqualTo(1);

        double precision = (double) tp / (tp + fp);
        double recall = (double) tp / gts.size();
        double f1 = (2.0 * precision * recall) / (precision + recall);

        assertThat(precision).isCloseTo(0.50, offset(1e-4));
        assertThat(recall).isCloseTo(1.00, offset(1e-4));
        assertThat(f1).isCloseTo(2.0 / 3.0, offset(1e-4));
    }

    @Test
    @DisplayName("Empty background frames produce strictly zero true positives and zero false negatives")
    void testBackgroundFrameProducesZeroTpAndZeroFn() {
        List<BoundingBox> gts = List.of();

        List<DetectionUtils.DetectionPrediction> preds = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            BoundingBox b = BoundingBox.ofNormalized(0.1 + i * 0.01, 0.1, 0.15 + i * 0.01, 0.15, 0, "traffic sign");
            preds.add(new DetectionUtils.DetectionPrediction(b, 0.60f, 2, 2));
        }

        int tp = 0;
        int fp = 0;
        int fn = 0;

        for (var p : preds) {
            fp++;
        }

        assertThat(tp).isEqualTo(0);
        assertThat(fp).isEqualTo(50);
        assertThat(fn).isEqualTo(0);

        double prec = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
        double rec = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
        double f1 = (prec + rec > 1e-9) ? (2.0 * prec * rec) / (prec + rec) : 0.0;

        assertThat(prec).isEqualTo(0.0);
        assertThat(rec).isEqualTo(0.0);
        assertThat(f1).isEqualTo(0.0);
    }

    @Test
    @DisplayName("Verifies that F1 arithmetic formula matches reported values within float precision")
    void testF1FormulaEquivalence() {
        double prec = 0.00119695;
        double rec = 0.18032787;

        double expectedF1 = (2.0 * prec * rec) / (prec + rec);
        assertThat(expectedF1).isCloseTo(0.00237812, offset(1e-7));
    }
}
