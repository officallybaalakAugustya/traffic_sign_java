package com.trafficsign.safety;

import com.trafficsign.data.BoundingBox;

public class SignTrack {

    private final int trackId;
    private final int firstSeenFrame;
    private int lastSeenFrame;
    private long lastSeenTimestampMs;
    private int observationCount;
    private BoundingBox latestNormalizedBox;
    private BoundingBox latestAbsoluteBox;
    private int bestClassId;
    private String bestClassName;
    private float bestClassificationConfidence;
    private float bestDetectionConfidence;
    private SignRuleCatalogue.SignRule bestRule;
    private boolean eventEmitted;
    private boolean confidentEventEmitted;
    private boolean active;

    public SignTrack(int trackId, SignObservation initialObservation) {
        this.trackId = trackId;
        this.firstSeenFrame = initialObservation.frameIndex();
        this.lastSeenFrame = initialObservation.frameIndex();
        this.lastSeenTimestampMs = initialObservation.timestampMs();
        this.observationCount = 1;
        this.latestNormalizedBox = initialObservation.normalizedBox();
        this.latestAbsoluteBox = initialObservation.absoluteBox();
        this.bestClassId = initialObservation.predictedClassId();
        this.bestClassName = initialObservation.predictedClassName();
        this.bestClassificationConfidence = initialObservation.classificationConfidence();
        this.bestDetectionConfidence = initialObservation.detectionConfidence();
        this.bestRule = initialObservation.rule();
        this.eventEmitted = false;
        this.confidentEventEmitted = false;
        this.active = true;
    }

    public void update(SignObservation obs) {
        this.lastSeenFrame = obs.frameIndex();
        this.lastSeenTimestampMs = obs.timestampMs();
        this.observationCount++;
        this.latestNormalizedBox = obs.normalizedBox();
        this.latestAbsoluteBox = obs.absoluteBox();

        if (obs.classificationConfidence() > this.bestClassificationConfidence) {
            this.bestClassId = obs.predictedClassId();
            this.bestClassName = obs.predictedClassName();
            this.bestClassificationConfidence = obs.classificationConfidence();
            this.bestRule = obs.rule();
        }
        if (obs.detectionConfidence() > this.bestDetectionConfidence) {
            this.bestDetectionConfidence = obs.detectionConfidence();
        }
    }

    public boolean matches(BoundingBox candidateNormalizedBox, double minIou) {
        if (candidateNormalizedBox == null || latestNormalizedBox == null) {
            return false;
        }
        double iou = latestNormalizedBox.computeIoU(candidateNormalizedBox);
        if (iou >= minIou) {
            return true;
        }
        double cx1 = (latestNormalizedBox.x1() + latestNormalizedBox.x2()) / 2.0;
        double cy1 = (latestNormalizedBox.y1() + latestNormalizedBox.y2()) / 2.0;
        double cx2 = (candidateNormalizedBox.x1() + candidateNormalizedBox.x2()) / 2.0;
        double cy2 = (candidateNormalizedBox.y1() + candidateNormalizedBox.y2()) / 2.0;
        double dx = Math.abs(cx1 - cx2);
        double dy = Math.abs(cy1 - cy2);
        return (dx <= 0.08 && dy <= 0.08 && iou >= 0.10);
    }

    public int getTrackId() {
        return trackId;
    }

    public int getFirstSeenFrame() {
        return firstSeenFrame;
    }

    public int getLastSeenFrame() {
        return lastSeenFrame;
    }

    public long getLastSeenTimestampMs() {
        return lastSeenTimestampMs;
    }

    public int getObservationCount() {
        return observationCount;
    }

    public BoundingBox getLatestNormalizedBox() {
        return latestNormalizedBox;
    }

    public BoundingBox getLatestAbsoluteBox() {
        return latestAbsoluteBox;
    }

    public int getBestClassId() {
        return bestClassId;
    }

    public String getBestClassName() {
        return bestClassName;
    }

    public float getBestClassificationConfidence() {
        return bestClassificationConfidence;
    }

    public float getBestDetectionConfidence() {
        return bestDetectionConfidence;
    }

    public SignRuleCatalogue.SignRule getBestRule() {
        return bestRule;
    }

    public boolean isEventEmitted() {
        return eventEmitted;
    }

    public void setEventEmitted(boolean eventEmitted) {
        this.eventEmitted = eventEmitted;
    }

    public boolean isConfidentEventEmitted() {
        return confidentEventEmitted;
    }

    public void setConfidentEventEmitted(boolean confidentEventEmitted) {
        this.confidentEventEmitted = confidentEventEmitted;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }
}
