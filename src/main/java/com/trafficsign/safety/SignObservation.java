package com.trafficsign.safety;

import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;

public record SignObservation(
        int frameIndex,
        long timestampMs,
        BoundingBox absoluteBox,
        BoundingBox normalizedBox,
        float detectionConfidence,
        int predictedClassId,
        String predictedClassName,
        float classificationConfidence,
        SignRuleCatalogue.SignRule rule,
        RoadSceneInferencePipeline.UncertaintyState uncertaintyState
) {
    public boolean isConfident() {
        return uncertaintyState == RoadSceneInferencePipeline.UncertaintyState.CONFIDENT;
    }
}
