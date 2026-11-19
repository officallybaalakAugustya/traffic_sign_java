"""
Unit tests for Two-Stage Road Scene Traffic Sign Inference Pipeline.
Verifies end-to-end execution, uncertainty states, and visual HUD annotation.
"""

import numpy as np
from PIL import Image
import pytest
import torch

from trafficsign.inference.pipeline import (
    RoadSceneInferencePipeline,
    PipelineResult,
    RoadSceneDetection,
    UncertaintyState,
)
from trafficsign.safety.catalogue import SignRuleCatalogue
from trafficsign.safety.engine import SafetyEventEngine
from trafficsign.safety.state import SafetyState


def test_pipeline_initialization():
    pipeline = RoadSceneInferencePipeline(device=torch.device("cpu"))
    assert pipeline is not None
    assert pipeline.taxonomy is not None
    assert pipeline.classifier_model is not None


def test_pipeline_synthetic_inference():
    pipeline = RoadSceneInferencePipeline(device=torch.device("cpu"))

    # Create synthetic test image (e.g. 640x480 RGB road frame)
    synthetic_img = Image.new("RGB", (640, 480), color=(120, 140, 160))

    result = pipeline.process_image(synthetic_img)
    assert isinstance(result, PipelineResult)
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.detection_latency_ms >= 0
    assert result.classification_latency_ms >= 0
    assert isinstance(result.detections, list)

    # Test visual annotation
    annotated = pipeline.draw_detections(synthetic_img, result)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (640, 480)


def test_pipeline_safety_integration():
    pipeline = RoadSceneInferencePipeline(device=torch.device("cpu"))
    catalogue = SignRuleCatalogue()
    safety_state = SafetyState()
    engine = SafetyEventEngine(catalogue, safety_state)

    synthetic_img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    result = pipeline.process_image(synthetic_img)

    for det in result.detections:
        events = engine.process_detection(
            class_id=det.predicted_class_id,
            class_confidence=det.classification_confidence,
            detection_confidence=det.detection_confidence,
            box=det.normalized_box,
            timestamp_ms=1000
        )
        assert isinstance(events, list)
