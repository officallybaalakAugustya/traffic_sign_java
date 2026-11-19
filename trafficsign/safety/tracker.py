"""
Spatio-temporal Sign Observation and Tracking
mirroring com.trafficsign.safety.SignObservation and SignTrack.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from trafficsign.data.bounding_box import BoundingBox
from trafficsign.safety.catalogue import SignRule


@dataclass
class SignObservation:
    frame_index: int
    timestamp_ms: int
    absolute_box: BoundingBox
    normalized_box: BoundingBox
    detection_confidence: float
    predicted_class_id: int
    predicted_class_name: str
    classification_confidence: float
    rule: SignRule
    uncertainty_state: str  # CONFIDENT, LOW_DETECTION_CONFIDENCE, LOW_CLASSIFICATION_CONFIDENCE, UNKNOWN_CLASS


class SignTrack:
    """
    Temporal tracker tracking a traffic sign instance across consecutive video frames.
    """

    def __init__(self, track_id: int, initial_observation: SignObservation):
        self.track_id = track_id
        self.observations: List[SignObservation] = [initial_observation]
        self.latest_normalized_box: BoundingBox = initial_observation.normalized_box
        self.latest_absolute_box: BoundingBox = initial_observation.absolute_box
        self.last_seen_frame: int = initial_observation.frame_index
        self.is_active: bool = True
        self.is_event_emitted: bool = False
        self.is_confident_event_emitted: bool = False

        self._class_confidence_sum: Dict[int, float] = {initial_observation.predicted_class_id: initial_observation.classification_confidence}
        self._class_counts: Dict[int, int] = {initial_observation.predicted_class_id: 1}
        self.best_class_id: int = initial_observation.predicted_class_id
        self.best_rule: SignRule = initial_observation.rule

    def matches(self, candidate_box: BoundingBox, iou_threshold: float = 0.25) -> bool:
        iou = self.latest_normalized_box.compute_iou(candidate_box)
        return iou >= iou_threshold

    def update(self, observation: SignObservation) -> None:
        self.observations.append(observation)
        self.latest_normalized_box = observation.normalized_box
        self.latest_absolute_box = observation.absolute_box
        self.last_seen_frame = observation.frame_index

        cid = observation.predicted_class_id
        conf = observation.classification_confidence
        self._class_confidence_sum[cid] = self._class_confidence_sum.get(cid, 0.0) + conf
        self._class_counts[cid] = self._class_counts.get(cid, 0) + 1

        # Select class with highest accumulated confidence * count
        best_score = -1.0
        best_cid = self.best_class_id
        for c, count in self._class_counts.items():
            avg_conf = self._class_confidence_sum[c] / count
            score = count * avg_conf
            if score > best_score:
                best_score = score
                best_cid = c

        self.best_class_id = best_cid
        self.best_rule = observation.rule if observation.predicted_class_id == best_cid else self.best_rule

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def latest_observation(self) -> SignObservation:
        return self.observations[-1]
