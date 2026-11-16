package com.trafficsign.data.detection;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trafficsign.app.Task13ExpandedDatasetManager;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class ExpandedDatasetSplitIsolationTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("Expanded dataset 3-way split manifest guarantees strict sequence isolation and canonical val preservation")
    void testThreeWaySequenceIsolationAndIntegrity() throws Exception {
        Path manifestPath = Path.of("data/splits/indian_road_detection_expanded_splits.json");
        if (!Files.exists(manifestPath)) {
            manifestPath = Path.of("S:/TrafficSignRecognitionRoadSafetyMonitor/data/splits/indian_road_detection_expanded_splits.json");
        }
        assertThat(Files.exists(manifestPath)).isTrue();

        Task13ExpandedDatasetManager.ExpandedSplitManifest manifest = mapper.readValue(
                manifestPath.toFile(),
                Task13ExpandedDatasetManager.ExpandedSplitManifest.class
        );

        assertThat(manifest.totalArchivesScanned()).isEqualTo(31);
        assertThat(manifest.totalMatchedJpgJsonPairs()).isEqualTo(31000);
        assertThat(manifest.valSequences()).isEqualTo(27);
        assertThat(manifest.valFrames()).isEqualTo(174);
        assertThat(manifest.valFramesWithSign()).isEqualTo(87);
        assertThat(manifest.valEmptyFrames()).isEqualTo(87);
        assertThat(manifest.valSignBoxes()).isEqualTo(122);

        assertThat(manifest.sequenceOverlapTrainVal()).isEqualTo(0);
        assertThat(manifest.sequenceOverlapTrainTest()).isEqualTo(0);
        assertThat(manifest.sequenceOverlapValTest()).isEqualTo(0);

        assertThat(manifest.testFrames()).isEqualTo(200);
        assertThat(manifest.testFramesWithSign()).isEqualTo(100);
        assertThat(manifest.testEmptyFrames()).isEqualTo(100);
        assertThat(manifest.testSignBoxes()).isGreaterThan(0);

        Set<String> trainSeqs = new HashSet<>();
        Set<String> valSeqs = new HashSet<>();
        Set<String> testSeqs = new HashSet<>();

        int actualTrain = 0, actualVal = 0, actualTest = 0;
        int actualValSigns = 0, actualTestSigns = 0;

        for (var sample : manifest.samples()) {
            if (sample.partition() == DetectionDataPipeline.SplitPartition.TRAIN) {
                trainSeqs.add(sample.sequenceId());
                actualTrain++;
            } else if (sample.partition() == DetectionDataPipeline.SplitPartition.VALIDATION) {
                valSeqs.add(sample.sequenceId());
                actualVal++;
                if (sample.hasSign()) {
                    actualValSigns += sample.boxes().size();
                    assertThat(sample.boxes()).isNotEmpty();
                } else {
                    assertThat(sample.boxes()).isEmpty();
                }
            } else if (sample.partition() == DetectionDataPipeline.SplitPartition.TEST) {
                testSeqs.add(sample.sequenceId());
                actualTest++;
                if (sample.hasSign()) {
                    actualTestSigns += sample.boxes().size();
                    assertThat(sample.boxes()).isNotEmpty();
                    for (var box : sample.boxes()) {
                        assertThat(box.x1()).isLessThan(box.x2());
                        assertThat(box.y1()).isLessThan(box.y2());
                    }
                } else {
                    assertThat(sample.boxes()).isEmpty();
                }
            }
        }

        assertThat(actualVal).isEqualTo(174);
        assertThat(actualValSigns).isEqualTo(122);
        assertThat(actualTest).isEqualTo(200);
        assertThat(actualTrain).isEqualTo(manifest.trainFrames());

        Set<String> trainValIntersect = new HashSet<>(trainSeqs);
        trainValIntersect.retainAll(valSeqs);
        assertThat(trainValIntersect).isEmpty();

        Set<String> trainTestIntersect = new HashSet<>(trainSeqs);
        trainTestIntersect.retainAll(testSeqs);
        assertThat(trainTestIntersect).isEmpty();

        Set<String> valTestIntersect = new HashSet<>(valSeqs);
        valTestIntersect.retainAll(testSeqs);
        assertThat(valTestIntersect).isEmpty();
    }
}
