"""
Unit tests for Tkinter GUI Application (RoadSafetyMonitorApp).
Verifies widget construction, variable binding, HUD rendering, and event review.
"""

from PIL import Image
import tkinter as tk
import pytest
from unittest.mock import MagicMock

from trafficsign.inference.pipeline import PipelineResult
from trafficsign.safety.engine import OperatorStatus
from trafficsign.ui.app import RoadSafetyMonitorApp


def test_app_initialization_and_teardown():
    try:
        root = tk.Tk()
    except Exception as e:
        pytest.skip(f"Tkinter display unavailable in this environment: {e}")

    mock_pipeline = MagicMock()
    mock_pipeline.draw_detections.return_value = Image.new("RGB", (320, 240), color="blue")
    mock_pipeline.process_image.return_value = PipelineResult(
        image_width=320,
        image_height=240,
        detection_latency_ms=10,
        classification_latency_ms=10,
        detections=[]
    )
    app = RoadSafetyMonitorApp(root=root, pipeline=mock_pipeline)

    assert app.root is not None
    assert app.canvas is not None
    assert app.hud_speed_canvas is not None
    assert app.advisory_var is not None
    assert app.tree is not None

    # Test HUD drawing
    app._draw_speed_limit_circle(50)
    app._update_hud(50, ["NO_OVERTAKING"], ["PEDESTRIAN_CROSSING"])

    # Clean up
    root.destroy()
