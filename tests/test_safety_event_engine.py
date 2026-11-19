from trafficsign.data.bounding_box import BoundingBox
from trafficsign.inference.pipeline import RoadSceneDetection, UncertaintyState
from trafficsign.safety.catalogue import EventType, SeverityLevel, SignRuleCatalogue
from trafficsign.safety.engine import SafetyEventEngine
from trafficsign.safety.state import SafetyState


def test_safety_event_engine_tracking_and_emission():
    catalogue = SignRuleCatalogue()
    state = SafetyState()
    engine = SafetyEventEngine(catalogue, state)
    engine.save_evidence_snapshots = False

    det1 = RoadSceneDetection(
        absolute_box=BoundingBox(10, 10, 50, 50),
        normalized_box=BoundingBox(0.1, 0.1, 0.2, 0.2),
        detection_confidence=0.85,
        predicted_class_id=51,  # Speed Limit 40
        predicted_class_name="Speed Limit 40",
        category="MANDATORY",
        safety_implication="Speed limit 40 km/h",
        classification_confidence=0.92,
        uncertainty_state=UncertaintyState.CONFIDENT,
        display_label="Speed Limit 40"
    )

    # Frame 1: First detection triggers event
    events_f1 = engine.process_frame(1, 1000, None, [det1])
    assert len(events_f1) == 1
    assert events_f1[0].event_type == EventType.SPEED_LIMIT_POSTED
    assert state.get_current_posted_speed_limit() == 40

    # Frame 2: Same sign instance (overlapping box) should NOT emit duplicate event
    det2 = RoadSceneDetection(
        absolute_box=BoundingBox(12, 12, 52, 52),
        normalized_box=BoundingBox(0.11, 0.11, 0.21, 0.21),
        detection_confidence=0.86,
        predicted_class_id=51,
        predicted_class_name="Speed Limit 40",
        category="MANDATORY",
        safety_implication="Speed limit 40 km/h",
        classification_confidence=0.94,
        uncertainty_state=UncertaintyState.CONFIDENT,
        display_label="Speed Limit 40"
    )
    events_f2 = engine.process_frame(2, 1040, None, [det2])
    assert len(events_f2) == 0  # Deduplicated!
    assert state.get_current_posted_speed_limit() == 40
