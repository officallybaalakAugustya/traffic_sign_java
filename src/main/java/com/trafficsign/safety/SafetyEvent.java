package com.trafficsign.safety;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;

import java.util.UUID;

public record SafetyEvent(
        @JsonProperty("eventId") String eventId,
        @JsonProperty("frameIndex") int frameIndex,
        @JsonProperty("timestampMs") long timestampMs,
        @JsonProperty("trackId") int trackId,
        @JsonProperty("eventType") SignRuleCatalogue.EventType eventType,
        @JsonProperty("severity") SignRuleCatalogue.SeverityLevel severity,
        @JsonProperty("classId") int classId,
        @JsonProperty("canonicalLabel") String canonicalLabel,
        @JsonProperty("displayName") String displayName,
        @JsonProperty("category") SignRuleCatalogue.RuleCategory category,
        @JsonProperty("detectionConfidence") float detectionConfidence,
        @JsonProperty("classificationConfidence") float classificationConfidence,
        @JsonProperty("absoluteBox") BoundingBox absoluteBox,
        @JsonProperty("normalizedBox") BoundingBox normalizedBox,
        @JsonProperty("driverAdvisory") String driverAdvisory,
        @JsonProperty("numericalValue") Double numericalValue,
        @JsonProperty("unit") String unit,
        @JsonProperty("evidenceImagePath") String evidenceImagePath,
        @JsonProperty("operatorStatus") OperatorStatus operatorStatus,
        @JsonProperty("operatorNote") String operatorNote
) {

    public enum OperatorStatus {
        UNREVIEWED,
        CONFIRMED,
        FLAGGED,
        REJECTED
    }

    public static SafetyEvent create(
            int frameIndex,
            long timestampMs,
            int trackId,
            SignObservation obs,
            String evidenceImagePath
    ) {
        SignRuleCatalogue.SignRule rule = obs.rule();
        String id = "EVT-" + frameIndex + "-" + trackId + "-" + UUID.randomUUID().toString().substring(0, 8);
        boolean isConfident = obs.uncertaintyState() == RoadSceneInferencePipeline.UncertaintyState.CONFIDENT;

        SignRuleCatalogue.EventType eventType;
        SignRuleCatalogue.SeverityLevel severity;
        String driverAdvisory;
        Double numericalValue;
        String unit;
        String displayName;
        SignRuleCatalogue.RuleCategory category;

        if (isConfident && rule != null && rule.eventType() != SignRuleCatalogue.EventType.UNCERTAIN_DETECTION) {
            eventType = rule.eventType();
            severity = rule.severity();
            driverAdvisory = rule.driverAdvisory();
            numericalValue = rule.numericalValue();
            unit = rule.unit() != null ? rule.unit() : "";
            displayName = rule.displayName();
            category = rule.category();
        } else {
            eventType = SignRuleCatalogue.EventType.UNCERTAIN_DETECTION;
            severity = SignRuleCatalogue.SeverityLevel.INFO;
            driverAdvisory = "Uncertain sign observation (" + (obs.predictedClassName() != null ? obs.predictedClassName() : "Unknown")
                    + ", conf " + Math.round(obs.classificationConfidence() * 100.0f) + "%); visual verification required.";
            numericalValue = null;
            unit = "";
            displayName = "Uncertain: " + (obs.predictedClassName() != null ? obs.predictedClassName() : "Unknown");
            category = rule != null ? rule.category() : SignRuleCatalogue.RuleCategory.UNKNOWN;
        }

        return new SafetyEvent(
                id,
                frameIndex,
                timestampMs,
                trackId,
                eventType,
                severity,
                rule != null ? rule.classId() : obs.predictedClassId(),
                rule != null ? rule.canonicalLabel() : "UNKNOWN",
                displayName,
                category,
                obs.detectionConfidence(),
                obs.classificationConfidence(),
                obs.absoluteBox(),
                obs.normalizedBox(),
                driverAdvisory,
                numericalValue,
                unit,
                evidenceImagePath,
                OperatorStatus.UNREVIEWED,
                ""
        );
    }

    public SafetyEvent withOperatorReview(OperatorStatus newStatus, String newNote) {
        return new SafetyEvent(
                this.eventId,
                this.frameIndex,
                this.timestampMs,
                this.trackId,
                this.eventType,
                this.severity,
                this.classId,
                this.canonicalLabel,
                this.displayName,
                this.category,
                this.detectionConfidence,
                this.classificationConfidence,
                this.absoluteBox,
                this.normalizedBox,
                this.driverAdvisory,
                this.numericalValue,
                this.unit,
                this.evidenceImagePath,
                newStatus,
                newNote != null ? newNote : ""
        );
    }
}
