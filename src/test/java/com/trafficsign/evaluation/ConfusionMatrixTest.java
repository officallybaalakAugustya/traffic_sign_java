package com.trafficsign.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ConfusionMatrix Evaluation Metrics Tests")
class ConfusionMatrixTest {

    @Test
    @DisplayName("Should correctly calculate accuracy, precision, recall and F1 on known 3-class distribution")
    void testStandardConfusionMatrix() {
        // 3 classes (0, 1, 2)
        ConfusionMatrix cm = new ConfusionMatrix(3);

        // Ground truth 0: 5 samples -> 4 predicted as 0, 1 predicted as 1
        cm.record(0, 0); cm.record(0, 0); cm.record(0, 0); cm.record(0, 0);
        cm.record(0, 1);

        // Ground truth 1: 5 samples -> 1 predicted as 0, 3 predicted as 1, 1 predicted as 2
        cm.record(1, 0);
        cm.record(1, 1); cm.record(1, 1); cm.record(1, 1);
        cm.record(1, 2);

        // Ground truth 2: 5 samples -> 0 predicted as 0, 1 predicted as 1, 4 predicted as 2
        cm.record(2, 1);
        cm.record(2, 2); cm.record(2, 2); cm.record(2, 2); cm.record(2, 2);

        assertEquals(15, cm.getTotalSamples());
        // Diagonal: 4 + 3 + 4 = 11 correct out of 15
        assertEquals(11.0 / 15.0, cm.getAccuracy(), 1e-5);

        // Class 0:
        // TP = 4, FP = 1 (from class 1), FN = 1 (predicted 1)
        // Precision = 4 / (4 + 1) = 0.8
        // Recall = 4 / (4 + 1) = 0.8
        // F1 = 0.8
        assertEquals(0.8, cm.getPrecision(0), 1e-5);
        assertEquals(0.8, cm.getRecall(0), 1e-5);
        assertEquals(0.8, cm.getF1(0), 1e-5);

        // Class 1:
        // TP = 3, FP = 1 (class 0) + 1 (class 2) = 2
        // Precision = 3 / (3 + 2) = 0.6
        // Recall = 3 / 5 = 0.6
        // F1 = 0.6
        assertEquals(0.6, cm.getPrecision(1), 1e-5);
        assertEquals(0.6, cm.getRecall(1), 1e-5);
        assertEquals(0.6, cm.getF1(1), 1e-5);

        // Class 2:
        // TP = 4, FP = 1 (class 1)
        // Precision = 4 / (4 + 1) = 0.8
        // Recall = 4 / 5 = 0.8
        // F1 = 0.8
        assertEquals(0.8, cm.getPrecision(2), 1e-5);
        assertEquals(0.8, cm.getRecall(2), 1e-5);
        assertEquals(0.8, cm.getF1(2), 1e-5);

        // Macro averages:
        // Macro Precision = (0.8 + 0.6 + 0.8) / 3 = 2.2 / 3 ≈ 0.733333
        // Macro Recall = (0.8 + 0.6 + 0.8) / 3 ≈ 0.733333
        // Macro F1 = (0.8 + 0.6 + 0.8) / 3 ≈ 0.733333
        assertEquals(2.2 / 3.0, cm.getMacroPrecision(), 1e-5);
        assertEquals(2.2 / 3.0, cm.getMacroRecall(), 1e-5);
        assertEquals(2.2 / 3.0, cm.getMacroF1(), 1e-5);

        // String representation should contain class labels
        String summary = cm.formatSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("Overall Accuracy"));
        assertTrue(summary.contains("Macro F1"));
    }

    @Test
    @DisplayName("Should safely handle degenerate zero-count classes without NaN or division by zero")
    void testDegenerateZeroCounts() {
        ConfusionMatrix cm = new ConfusionMatrix(3);
        // Only record class 0
        cm.record(0, 0);

        assertEquals(1.0, cm.getAccuracy());
        assertEquals(1.0, cm.getPrecision(0));
        assertEquals(1.0, cm.getRecall(0));
        assertEquals(1.0, cm.getF1(0));

        // Class 1 and 2 have 0 TP, 0 FP, 0 FN
        assertEquals(0.0, cm.getPrecision(1));
        assertEquals(0.0, cm.getRecall(1));
        assertEquals(0.0, cm.getF1(1));

        assertFalse(Double.isNaN(cm.getMacroPrecision()));
        assertFalse(Double.isNaN(cm.getMacroRecall()));
        assertFalse(Double.isNaN(cm.getMacroF1()));
    }

    @Test
    @DisplayName("Should validate constructor bounds and out-of-range labels")
    void testBoundaryValidation() {
        assertThrows(IllegalArgumentException.class, () -> new ConfusionMatrix(0));
        assertThrows(IllegalArgumentException.class, () -> new ConfusionMatrix(-1));

        ConfusionMatrix cm = new ConfusionMatrix(5);
        assertThrows(IllegalArgumentException.class, () -> cm.record(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> cm.record(0, 5));
        assertThrows(IllegalArgumentException.class, () -> cm.getPrecision(5));
        assertThrows(IllegalArgumentException.class, () -> cm.getRecall(-1));
    }
}
