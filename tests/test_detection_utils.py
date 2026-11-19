import numpy as np
from trafficsign.data.bounding_box import BoundingBox
from trafficsign.data.detection import DetectionPrediction, DetectionUtils


def test_box_target_roundtrip():
    box = BoundingBox(x1=0.2, y1=0.3, x2=0.4, y2=0.5)
    target = DetectionUtils.box_to_target(box, grid_cols=16, grid_rows=16)

    cell_x, cell_y, bw, bh, row, col = target
    recovered = DetectionUtils.target_to_box(cell_x, cell_y, bw, bh, row, col, 16, 16)

    assert abs(recovered.x1 - box.x1) < 1e-4
    assert abs(recovered.y1 - box.y1) < 1e-4
    assert abs(recovered.x2 - box.x2) < 1e-4
    assert abs(recovered.y2 - box.y2) < 1e-4


def test_non_maximum_suppression():
    box1 = BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    box2 = BoundingBox(x1=0.11, y1=0.11, x2=0.31, y2=0.31)  # High overlap
    box3 = BoundingBox(x1=0.7, y1=0.7, x2=0.9, y2=0.9)      # Separate box

    dets = [
        DetectionPrediction(box=box1, confidence=0.85),
        DetectionPrediction(box=box2, confidence=0.90),  # higher confidence
        DetectionPrediction(box=box3, confidence=0.75),
    ]

    selected = DetectionUtils.non_maximum_suppression(dets, iou_threshold=0.3)
    assert len(selected) == 2
    # box2 (0.90) and box3 (0.75) should be kept; box1 suppressed
    assert selected[0].confidence == 0.90
    assert selected[1].confidence == 0.75
