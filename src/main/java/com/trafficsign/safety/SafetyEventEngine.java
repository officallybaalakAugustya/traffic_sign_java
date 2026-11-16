package com.trafficsign.safety;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class SafetyEventEngine {

    public interface SafetyEventListener {
        void onSafetyEvent(SafetyEvent event);
        void onSafetyStateUpdated(SafetyState state);
    }

    private final SignRuleCatalogue catalogue;
    private final SafetyState safetyState;
    private final List<SignTrack> activeTracks = new ArrayList<>();
    private final List<SafetyEvent> eventLog = new ArrayList<>();
    private final List<SafetyEventListener> listeners = new CopyOnWriteArrayList<>();
    private final AtomicInteger nextTrackId = new AtomicInteger(1001);
    private final ObjectMapper mapper;

    private Path evidenceDirectory = Path.of("reports/evidence");
    private boolean saveEvidenceSnapshots = true;
    private double trackIouThreshold = 0.25;
    private int maxMissingFrames = 15;
    private float minDetectionConfidence = 0.20f;
    private float minClassificationConfidence = 0.20f;
    private int minHitsToEmit = 1;

    public SafetyEventEngine() {
        this(new SignRuleCatalogue(), new SafetyState());
    }

    public SafetyEventEngine(SignRuleCatalogue catalogue, SafetyState safetyState) {
        this.catalogue = Objects.requireNonNull(catalogue, "catalogue must not be null");
        this.safetyState = Objects.requireNonNull(safetyState, "safetyState must not be null");
        this.mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void addListener(SafetyEventListener listener) {
        if (listener != null) {
            listeners.add(listener);
        }
    }

    public void removeListener(SafetyEventListener listener) {
        listeners.remove(listener);
    }

    public synchronized List<SafetyEvent> processFrame(
            int frameIndex,
            long timestampMs,
            BufferedImage fullFrame,
            RoadSceneInferencePipeline.PipelineResult pipelineResult
    ) {
        List<SafetyEvent> newEvents = new ArrayList<>();
        safetyState.prune(frameIndex, 60);

        List<SignObservation> observations = new ArrayList<>();
        if (pipelineResult != null && pipelineResult.detections() != null) {
            for (var det : pipelineResult.detections()) {
                if (det.detectionConfidence() < minDetectionConfidence) {
                    continue;
                }
                SignRuleCatalogue.SignRule rule = catalogue.getRule(det.predictedClassId());
                RoadSceneInferencePipeline.UncertaintyState state = det.uncertaintyState();
                if (det.classificationConfidence() < minClassificationConfidence && state == RoadSceneInferencePipeline.UncertaintyState.CONFIDENT) {
                    state = RoadSceneInferencePipeline.UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE;
                }
                SignObservation obs = new SignObservation(
                        frameIndex,
                        timestampMs,
                        det.absoluteBox(),
                        det.normalizedBox(),
                        det.detectionConfidence(),
                        det.predictedClassId(),
                        det.predictedClassName(),
                        det.classificationConfidence(),
                        rule,
                        state
                );
                observations.add(obs);
            }
        }

        boolean[] matchedObservations = new boolean[observations.size()];
        for (SignTrack track : activeTracks) {
            if (!track.isActive()) {
                continue;
            }
            int bestMatchIdx = -1;
            double bestMatchScore = -1.0;

            for (int i = 0; i < observations.size(); i++) {
                if (matchedObservations[i]) {
                    continue;
                }
                SignObservation obs = observations.get(i);
                if (track.matches(obs.normalizedBox(), trackIouThreshold)) {
                    double iou = track.getLatestNormalizedBox().computeIoU(obs.normalizedBox());
                    if (iou > bestMatchScore) {
                        bestMatchScore = iou;
                        bestMatchIdx = i;
                    }
                }
            }

            if (bestMatchIdx >= 0) {
                SignObservation matchedObs = observations.get(bestMatchIdx);
                matchedObservations[bestMatchIdx] = true;
                int prevClassId = track.getBestClassId();
                track.update(matchedObs);

                boolean isConfident = matchedObs.uncertaintyState() == RoadSceneInferencePipeline.UncertaintyState.CONFIDENT
                        && matchedObs.classificationConfidence() >= minClassificationConfidence;

                if (!track.isEventEmitted() && track.getObservationCount() >= minHitsToEmit) {
                    SafetyEvent event = emitEventForTrack(frameIndex, timestampMs, fullFrame, track, matchedObs);
                    if (event != null) {
                        newEvents.add(event);
                    }
                } else if (track.isEventEmitted() && !track.isConfidentEventEmitted() && isConfident) {
                    SafetyEvent event = emitEventForTrack(frameIndex, timestampMs, fullFrame, track, matchedObs);
                    if (event != null) {
                        newEvents.add(event);
                    }
                } else if (track.isEventEmitted() && prevClassId != track.getBestClassId() && matchedObs.classificationConfidence() >= 0.50f) {
                    SafetyEvent event = emitEventForTrack(frameIndex, timestampMs, fullFrame, track, matchedObs);
                    if (event != null) {
                        newEvents.add(event);
                    }
                }
            }
        }

        for (int i = 0; i < observations.size(); i++) {
            if (!matchedObservations[i]) {
                SignObservation obs = observations.get(i);
                int trackId = nextTrackId.getAndIncrement();
                SignTrack newTrack = new SignTrack(trackId, obs);
                activeTracks.add(newTrack);

                if (minHitsToEmit <= 1) {
                    SafetyEvent event = emitEventForTrack(frameIndex, timestampMs, fullFrame, newTrack, obs);
                    if (event != null) {
                        newEvents.add(event);
                    }
                }
            }
        }

        Iterator<SignTrack> trackIterator = activeTracks.iterator();
        while (trackIterator.hasNext()) {
            SignTrack track = trackIterator.next();
            if ((frameIndex - track.getLastSeenFrame()) > maxMissingFrames) {
                track.setActive(false);
                trackIterator.remove();
            }
        }

        if (!newEvents.isEmpty()) {
            notifyStateUpdated();
        }

        return newEvents;
    }

    private SafetyEvent emitEventForTrack(
            int frameIndex,
            long timestampMs,
            BufferedImage fullFrame,
            SignTrack track,
            SignObservation obs
    ) {
        String evidencePath = "";
        if (saveEvidenceSnapshots && fullFrame != null) {
            evidencePath = saveEvidence(frameIndex, track.getTrackId(), obs, fullFrame);
        }

        SafetyEvent event = SafetyEvent.create(frameIndex, timestampMs, track.getTrackId(), obs, evidencePath);
        track.setEventEmitted(true);
        if (event.eventType() != SignRuleCatalogue.EventType.UNCERTAIN_DETECTION
                && event.severity() != SignRuleCatalogue.SeverityLevel.INFO) {
            track.setConfidentEventEmitted(true);
        }
        eventLog.add(event);
        safetyState.updateWithEvent(event);

        for (SafetyEventListener listener : listeners) {
            try {
                listener.onSafetyEvent(event);
            } catch (Exception ignored) {
            }
        }
        return event;
    }

    private String saveEvidence(
            int frameIndex,
            int trackId,
            SignObservation obs,
            BufferedImage fullFrame
    ) {
        try {
            Files.createDirectories(evidenceDirectory);
            BoundingBox abs = obs.absoluteBox();
            int x1 = Math.max(0, Math.min((int) Math.round(abs.x1()), fullFrame.getWidth() - 1));
            int y1 = Math.max(0, Math.min((int) Math.round(abs.y1()), fullFrame.getHeight() - 1));
            int x2 = Math.max(x1 + 1, Math.min((int) Math.round(abs.x2()), fullFrame.getWidth()));
            int y2 = Math.max(y1 + 1, Math.min((int) Math.round(abs.y2()), fullFrame.getHeight()));
            int w = Math.max(1, x2 - x1);
            int h = Math.max(1, y2 - y1);

            BufferedImage crop = fullFrame.getSubimage(x1, y1, w, h);
            String fileName = String.format("evidence_f%05d_t%d_%s.png",
                    frameIndex, trackId, obs.rule().canonicalLabel().toLowerCase());
            Path targetFile = evidenceDirectory.resolve(fileName);
            ImageIO.write(crop, "png", targetFile.toFile());
            return targetFile.toString();
        } catch (Exception e) {
            return "";
        }
    }

    private void notifyStateUpdated() {
        for (SafetyEventListener listener : listeners) {
            try {
                listener.onSafetyStateUpdated(safetyState);
            } catch (Exception ignored) {
            }
        }
    }

    public synchronized boolean updateOperatorReview(
            String eventId,
            SafetyEvent.OperatorStatus status,
            String note
    ) {
        for (int i = 0; i < eventLog.size(); i++) {
            SafetyEvent evt = eventLog.get(i);
            if (evt.eventId().equals(eventId)) {
                SafetyEvent updated = evt.withOperatorReview(status, note);
                eventLog.set(i, updated);
                return true;
            }
        }
        return false;
    }

    public synchronized void exportEventsToCsv(Path csvPath) throws IOException {
        if (csvPath.getParent() != null) {
            Files.createDirectories(csvPath.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(csvPath)) {
            writer.write("eventId,frameIndex,timestampMs,trackId,eventType,severity,classId,canonicalLabel,displayName,category,detectionConfidence,classificationConfidence,box_x1,box_y1,box_x2,box_y2,driverAdvisory,numericalValue,unit,evidenceImagePath,operatorStatus,operatorNote");
            writer.newLine();

            for (SafetyEvent evt : eventLog) {
                BoundingBox b = evt.normalizedBox() != null ? evt.normalizedBox() : BoundingBox.ofNormalized(0, 0, 0, 0, -1, "NONE");
                String line = String.format(
                        "\"%s\",%d,%d,%d,\"%s\",\"%s\",%d,\"%s\",\"%s\",\"%s\",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\"%s\",%s,\"%s\",\"%s\",\"%s\",\"%s\"",
                        escapeCsv(evt.eventId()),
                        evt.frameIndex(),
                        evt.timestampMs(),
                        evt.trackId(),
                        evt.eventType() != null ? evt.eventType().name() : "",
                        evt.severity() != null ? evt.severity().name() : "",
                        evt.classId(),
                        escapeCsv(evt.canonicalLabel()),
                        escapeCsv(evt.displayName()),
                        evt.category() != null ? evt.category().name() : "",
                        evt.detectionConfidence(),
                        evt.classificationConfidence(),
                        b.x1(), b.y1(), b.x2(), b.y2(),
                        escapeCsv(evt.driverAdvisory()),
                        evt.numericalValue() != null ? evt.numericalValue().toString() : "",
                        escapeCsv(evt.unit() != null ? evt.unit() : ""),
                        escapeCsv(evt.evidenceImagePath() != null ? evt.evidenceImagePath() : ""),
                        evt.operatorStatus() != null ? evt.operatorStatus().name() : "",
                        escapeCsv(evt.operatorNote() != null ? evt.operatorNote() : "")
                );
                writer.write(line);
                writer.newLine();
            }
        }
    }

    public synchronized void exportEventsToJson(Path jsonPath) throws IOException {
        if (jsonPath.getParent() != null) {
            Files.createDirectories(jsonPath.getParent());
        }
        mapper.writeValue(jsonPath.toFile(), eventLog);
    }

    private String escapeCsv(String str) {
        if (str == null) return "";
        return str.replace("\"", "\"\"");
    }

    public synchronized List<SafetyEvent> getEventLog() {
        return Collections.unmodifiableList(new ArrayList<>(eventLog));
    }

    public synchronized List<SignTrack> getActiveTracks() {
        return Collections.unmodifiableList(new ArrayList<>(activeTracks));
    }

    public SafetyState getSafetyState() {
        return safetyState;
    }

    public SignRuleCatalogue getCatalogue() {
        return catalogue;
    }

    public Path getEvidenceDirectory() {
        return evidenceDirectory;
    }

    public void setEvidenceDirectory(Path evidenceDirectory) {
        this.evidenceDirectory = evidenceDirectory;
    }

    public boolean isSaveEvidenceSnapshots() {
        return saveEvidenceSnapshots;
    }

    public void setSaveEvidenceSnapshots(boolean saveEvidenceSnapshots) {
        this.saveEvidenceSnapshots = saveEvidenceSnapshots;
    }

    public double getTrackIouThreshold() {
        return trackIouThreshold;
    }

    public void setTrackIouThreshold(double trackIouThreshold) {
        this.trackIouThreshold = trackIouThreshold;
    }

    public int getMaxMissingFrames() {
        return maxMissingFrames;
    }

    public void setMaxMissingFrames(int maxMissingFrames) {
        this.maxMissingFrames = maxMissingFrames;
    }

    public float getMinDetectionConfidence() {
        return minDetectionConfidence;
    }

    public void setMinDetectionConfidence(float minDetectionConfidence) {
        this.minDetectionConfidence = minDetectionConfidence;
    }

    public float getMinClassificationConfidence() {
        return minClassificationConfidence;
    }

    public void setMinClassificationConfidence(float minClassificationConfidence) {
        this.minClassificationConfidence = minClassificationConfidence;
    }

    public int getMinHitsToEmit() {
        return minHitsToEmit;
    }

    public void setMinHitsToEmit(int minHitsToEmit) {
        this.minHitsToEmit = minHitsToEmit;
    }

    public synchronized void clear() {
        this.activeTracks.clear();
        this.eventLog.clear();
        this.safetyState.reset();
        this.nextTrackId.set(1001);
    }
}
