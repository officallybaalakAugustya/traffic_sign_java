"""
Safety Event Model and Spatio-Temporal Event Engine
mirroring com.trafficsign.safety.SafetyEvent, OperatorStatus, and SafetyEventEngine.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Union
import uuid

import numpy as np
from PIL import Image

from trafficsign.data.bounding_box import BoundingBox
from trafficsign.safety.catalogue import EventType, RuleCategory, SeverityLevel, SignRule, SignRuleCatalogue
from trafficsign.safety.state import SafetyState
from trafficsign.safety.tracker import SignObservation, SignTrack


class OperatorStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    CONFIRMED = "CONFIRMED"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


@dataclass
class SafetyEvent:
    event_id: str
    frame_index: int
    timestamp_ms: int
    track_id: int
    event_type: EventType
    severity: SeverityLevel
    class_id: int
    canonical_label: str
    display_name: str
    category: RuleCategory
    detection_confidence: float
    classification_confidence: float
    absolute_box: BoundingBox
    normalized_box: BoundingBox
    driver_advisory: str
    numerical_value: Optional[float] = None
    unit: str = ""
    evidence_image_path: str = ""
    operator_status: OperatorStatus = OperatorStatus.UNREVIEWED
    operator_note: str = ""

    @classmethod
    def create(
        cls,
        frame_index: int,
        timestamp_ms: int,
        track_id: int,
        obs: SignObservation,
        evidence_image_path: str = ""
    ) -> SafetyEvent:
        rule = obs.rule
        evt_id = f"EVT-{frame_index}-{track_id}-{uuid.uuid4().hex[:8]}"
        state_val = getattr(obs.uncertainty_state, "name", str(obs.uncertainty_state))
        if "." in state_val:
            state_val = state_val.split(".")[-1]
        is_confident = (state_val == "CONFIDENT")

        if is_confident and rule is not None and rule.event_type != EventType.UNCERTAIN_DETECTION:
            event_type = rule.event_type
            severity = rule.severity
            driver_advisory = rule.driver_advisory
            numerical_value = rule.numerical_value
            unit = rule.unit or ""
            display_name = rule.display_name
            category = rule.category
        else:
            event_type = EventType.UNCERTAIN_DETECTION
            severity = SeverityLevel.INFO
            driver_advisory = (
                f"Uncertain sign observation ({obs.predicted_class_name or 'Unknown'}, "
                f"conf {round(obs.classification_confidence * 100)}%); visual verification required."
            )
            numerical_value = None
            unit = ""
            display_name = f"Uncertain: {obs.predicted_class_name or 'Unknown'}"
            category = rule.category if rule is not None else RuleCategory.UNKNOWN

        return cls(
            event_id=evt_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            track_id=track_id,
            event_type=event_type,
            severity=severity,
            class_id=rule.class_id if rule is not None else obs.predicted_class_id,
            canonical_label=rule.canonical_label if rule is not None else "UNKNOWN",
            display_name=display_name,
            category=category,
            detection_confidence=obs.detection_confidence,
            classification_confidence=obs.classification_confidence,
            absolute_box=obs.absolute_box,
            normalized_box=obs.normalized_box,
            driver_advisory=driver_advisory,
            numerical_value=numerical_value,
            unit=unit,
            evidence_image_path=evidence_image_path,
            operator_status=OperatorStatus.UNREVIEWED,
            operator_note=""
        )


class SafetyEventEngine:
    """
    Spatio-temporal tracking and safety event generation engine.
    """

    def __init__(
        self,
        catalogue: Optional[SignRuleCatalogue] = None,
        safety_state: Optional[SafetyState] = None
    ):
        self.catalogue = catalogue or SignRuleCatalogue()
        self.safety_state = safety_state or SafetyState()
        self.active_tracks: List[SignTrack] = []
        self.event_log: List[SafetyEvent] = []
        self.listeners: List[Callable[[SafetyEvent], None]] = []
        self.next_track_id = 1001

        self.evidence_directory = Path("reports/evidence")
        self.save_evidence_snapshots = True
        self.track_iou_threshold = 0.25
        self.max_missing_frames = 15
        self.min_detection_confidence = 0.20
        self.min_classification_confidence = 0.20
        self.min_hits_to_emit = 1

    def add_listener(self, listener: Callable[[SafetyEvent], None]) -> None:
        self.listeners.append(listener)

    def process_frame(
        self,
        frame_index: int,
        timestamp_ms: int,
        full_frame: Optional[Union[Image.Image, np.ndarray]],
        detections: Sequence[object]
    ) -> List[SafetyEvent]:
        new_events: List[SafetyEvent] = []
        self.safety_state.prune(frame_index, 60)

        observations: List[SignObservation] = []
        for det in (detections or []):
            det_conf = getattr(det, "detection_confidence", 0.0)
            if det_conf < self.min_detection_confidence:
                continue

            class_id = getattr(det, "predicted_class_id", -1)
            rule = self.catalogue.get_rule(class_id)
            state_obj = getattr(det, "uncertainty_state", "CONFIDENT")
            state_str = getattr(state_obj, "name", str(state_obj))
            if "." in state_str:
                state_str = state_str.split(".")[-1]
            cls_conf = getattr(det, "classification_confidence", 0.0)

            if cls_conf < self.min_classification_confidence and state_str == "CONFIDENT":
                state_str = "LOW_CLASSIFICATION_CONFIDENCE"

            obs = SignObservation(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                absolute_box=getattr(det, "absolute_box"),
                normalized_box=getattr(det, "normalized_box"),
                detection_confidence=det_conf,
                predicted_class_id=class_id,
                predicted_class_name=getattr(det, "predicted_class_name", ""),
                classification_confidence=cls_conf,
                rule=rule,
                uncertainty_state=state_str
            )
            observations.append(obs)

        # Spatio-temporal association
        matched_obs = [False] * len(observations)

        for track in self.active_tracks:
            if not track.is_active:
                continue

            best_match_idx = -1
            best_match_score = -1.0

            for i, obs in enumerate(observations):
                if matched_obs[i]:
                    continue
                if track.matches(obs.normalized_box, self.track_iou_threshold):
                    iou = track.latest_normalized_box.compute_iou(obs.normalized_box)
                    if iou > best_match_score:
                        best_match_score = iou
                        best_match_idx = i

            if best_match_idx >= 0:
                matched = observations[best_match_idx]
                matched_obs[best_match_idx] = True
                prev_cid = track.best_class_id
                track.update(matched)

                m_state = getattr(matched.uncertainty_state, "name", str(matched.uncertainty_state))
                if "." in m_state:
                    m_state = m_state.split(".")[-1]
                is_confident = (m_state == "CONFIDENT" and matched.classification_confidence >= self.min_classification_confidence)

                if not track.is_event_emitted and track.observation_count >= self.min_hits_to_emit:
                    event = self._emit_event(frame_index, timestamp_ms, full_frame, track, matched)
                    if event:
                        new_events.append(event)
                elif track.is_event_emitted and not track.is_confident_event_emitted and is_confident:
                    event = self._emit_event(frame_index, timestamp_ms, full_frame, track, matched)
                    if event:
                        new_events.append(event)
                elif track.is_event_emitted and prev_cid != track.best_class_id and matched.classification_confidence >= 0.50:
                    event = self._emit_event(frame_index, timestamp_ms, full_frame, track, matched)
                    if event:
                        new_events.append(event)

        # Unmatched observations initiate new tracks
        for i, obs in enumerate(observations):
            if not matched_obs[i]:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = SignTrack(tid, obs)
                self.active_tracks.append(new_track)

                if self.min_hits_to_emit <= 1:
                    event = self._emit_event(frame_index, timestamp_ms, full_frame, new_track, obs)
                    if event:
                        new_events.append(event)

        # Prune dead tracks
        self.active_tracks = [t for t in self.active_tracks if (frame_index - t.last_seen_frame) <= self.max_missing_frames]

        return new_events

    def _emit_event(
        self,
        frame_index: int,
        timestamp_ms: int,
        full_frame: Optional[Union[Image.Image, np.ndarray]],
        track: SignTrack,
        obs: SignObservation
    ) -> Optional[SafetyEvent]:
        evidence_path = ""
        if self.save_evidence_snapshots and full_frame is not None:
            evidence_path = self._save_evidence(frame_index, track.track_id, obs, full_frame)

        event = SafetyEvent.create(frame_index, timestamp_ms, track.track_id, obs, evidence_path)
        track.is_event_emitted = True
        if event.event_type != EventType.UNCERTAIN_DETECTION and event.severity != SeverityLevel.INFO:
            track.is_confident_event_emitted = True

        self.event_log.append(event)
        self.safety_state.update_with_event(event)

        for listener in self.listeners:
            try:
                listener(event)
            except Exception:
                pass

        return event

    def _save_evidence(
        self,
        frame_index: int,
        track_id: int,
        obs: SignObservation,
        full_frame: Union[Image.Image, np.ndarray]
    ) -> str:
        try:
            self.evidence_directory.mkdir(parents=True, exist_ok=True)
            fname = f"evidence_frame{frame_index:04d}_track{track_id}_{obs.rule.canonical_label.lower()}.png"
            out_path = self.evidence_directory / fname

            if isinstance(full_frame, np.ndarray):
                pil_img = Image.fromarray(full_frame)
            else:
                pil_img = full_frame

            box = obs.absolute_box
            w, h = pil_img.size
            crop_x1 = max(0, min(int(round(box.x1)), w - 1))
            crop_y1 = max(0, min(int(round(box.y1)), h - 1))
            crop_x2 = max(crop_x1 + 1, min(int(round(box.x2)), w))
            crop_y2 = max(crop_y1 + 1, min(int(round(box.y2)), h))

            cropped = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped.save(out_path)
            return str(out_path)
        except Exception:
            return ""

    def export_to_csv(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "eventId", "frameIndex", "timestampMs", "trackId", "eventType",
                "severity", "classId", "displayName", "detectionConfidence",
                "classificationConfidence", "driverAdvisory", "operatorStatus"
            ])
            for e in self.event_log:
                writer.writerow([
                    e.event_id, e.frame_index, e.timestamp_ms, e.track_id,
                    str(e.event_type), str(e.severity), e.class_id, e.display_name,
                    f"{e.detection_confidence:.4f}", f"{e.classification_confidence:.4f}",
                    e.driver_advisory, str(e.operator_status)
                ])

    def export_to_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for e in self.event_log:
            d = asdict(e)
            d["event_type"] = str(e.event_type)
            d["severity"] = str(e.severity)
            d["category"] = str(e.category)
            d["operator_status"] = str(e.operator_status)
            serializable.append(d)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def reset(self) -> None:
        self.active_tracks.clear()
        self.event_log.clear()
        self.safety_state.reset()
        self.next_track_id = 1001
