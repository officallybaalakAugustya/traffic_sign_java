from trafficsign.data.bounding_box import BoundingBox
from trafficsign.inference.pipeline import RoadSceneDetection, UncertaintyState
from trafficsign.safety.catalogue import EventType, SignRuleCatalogue
from trafficsign.safety.engine import SafetyEventEngine
from trafficsign.safety.state import SafetyState


def test_task17_empty_background_produces_no_false_alerts():
    catalogue = SignRuleCatalogue()
    state = SafetyState()
    engine = SafetyEventEngine(catalogue, state)
    engine.save_evidence_snapshots = False

    # Simulate 10 frames of an empty highway (zero detections)
    for frame in range(1, 11):
        events = engine.process_frame(frame, frame * 40, None, [])
        assert len(events) == 0

    assert state.get_current_posted_speed_limit() is None
    assert len(state.get_active_restrictions()) == 0
    assert len(state.get_active_hazards()) == 0


def test_task17_speed_limit_state_and_restriction_clearing():
    catalogue = SignRuleCatalogue()
    state = SafetyState()
    engine = SafetyEventEngine(catalogue, state)
    engine.save_evidence_snapshots = False

    # 1. Post speed limit 40
    det_speed = RoadSceneDetection(
        absolute_box=BoundingBox(10, 10, 50, 50),
        normalized_box=BoundingBox(0.1, 0.1, 0.2, 0.2),
        detection_confidence=0.88,
        predicted_class_id=51,
        predicted_class_name="Speed Limit 40",
        category="MANDATORY",
        safety_implication="",
        classification_confidence=0.90,
        uncertainty_state=UncertaintyState.CONFIDENT,
        display_label="Speed Limit 40"
    )
    engine.process_frame(1, 1000, None, [det_speed])
    assert state.get_current_posted_speed_limit() == 40

    # 2. Add prohibition (No Overtaking, class 34)
    det_no_overtake = RoadSceneDetection(
        absolute_box=BoundingBox(100, 10, 150, 50),
        normalized_box=BoundingBox(0.5, 0.1, 0.6, 0.2),
        detection_confidence=0.85,
        predicted_class_id=34,
        predicted_class_name="Overtaking Prohibited",
        category="MANDATORY",
        safety_implication="",
        classification_confidence=0.91,
        uncertainty_state=UncertaintyState.CONFIDENT,
        display_label="Overtaking Prohibited"
    )
    engine.process_frame(2, 1040, None, [det_no_overtake])
    assert len(state.get_active_restrictions()) == 1

    # 3. Restriction ends (class 40)
    det_ends = RoadSceneDetection(
        absolute_box=BoundingBox(200, 10, 250, 50),
        normalized_box=BoundingBox(0.7, 0.1, 0.8, 0.2),
        detection_confidence=0.89,
        predicted_class_id=40,
        predicted_class_name="Restriction Ends",
        category="MANDATORY",
        safety_implication="",
        classification_confidence=0.93,
        uncertainty_state=UncertaintyState.CONFIDENT,
        display_label="All Restrictions End"
    )
    engine.process_frame(3, 1080, None, [det_ends])
    assert len(state.get_active_restrictions()) == 0
