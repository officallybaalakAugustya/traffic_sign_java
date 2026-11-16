package com.trafficsign.safety;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SafetyState {

    private Integer currentPostedSpeedLimit;
    private final Set<String> activeRestrictions = new HashSet<>();
    private final Map<String, Integer> hazardExpiryMap = new LinkedHashMap<>();
    private SafetyEvent lastAlert;
    private int lastUpdatedFrame;

    public synchronized void updateWithEvent(SafetyEvent event) {
        if (event == null) {
            return;
        }
        this.lastAlert = event;
        this.lastUpdatedFrame = event.frameIndex();

        if (event.eventType() == SignRuleCatalogue.EventType.UNCERTAIN_DETECTION) {
            return;
        }

        switch (event.eventType()) {
            case SPEED_LIMIT_POSTED -> {
                if (event.numericalValue() != null) {
                    this.currentPostedSpeedLimit = (int) Math.round(event.numericalValue());
                }
            }
            case RESTRICTION_ENDS -> {
                this.activeRestrictions.clear();
            }
            case PROHIBITION_ALERT, RESTRICTION_ALERT -> {
                this.activeRestrictions.add(event.displayName());
            }
            case HAZARD_WARNING, LEVEL_CROSSING_ALERT, TRAFFIC_SIGNAL_AHEAD, STOP_MANDATE, GIVE_WAY_MANDATE -> {
                this.hazardExpiryMap.put(event.displayName(), event.frameIndex());
            }
            default -> {
            }
        }
    }

    public synchronized void prune(int currentFrame, int maxTtlFrames) {
        hazardExpiryMap.entrySet().removeIf(e -> (currentFrame - e.getValue()) > maxTtlFrames);
    }

    public synchronized void reset() {
        this.currentPostedSpeedLimit = null;
        this.activeRestrictions.clear();
        this.hazardExpiryMap.clear();
        this.lastAlert = null;
        this.lastUpdatedFrame = 0;
    }

    public synchronized Integer getCurrentPostedSpeedLimit() {
        return currentPostedSpeedLimit;
    }

    public synchronized Set<String> getActiveRestrictions() {
        return Collections.unmodifiableSet(new HashSet<>(activeRestrictions));
    }

    public synchronized List<String> getActiveHazards() {
        return Collections.unmodifiableList(new ArrayList<>(hazardExpiryMap.keySet()));
    }

    public synchronized SafetyEvent getLastAlert() {
        return lastAlert;
    }

    public synchronized int getLastUpdatedFrame() {
        return lastUpdatedFrame;
    }
}
