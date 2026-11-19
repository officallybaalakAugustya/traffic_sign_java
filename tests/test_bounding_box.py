import pytest
from trafficsign.data.bounding_box import BoundingBox


def test_bounding_box_geometry_and_iou():
    box1 = BoundingBox(x1=10.0, y1=10.0, x2=30.0, y2=30.0)
    assert box1.width == 20.0
    assert box1.height == 20.0
    assert box1.center_x == 20.0
    assert box1.center_y == 20.0
    assert box1.area == 400.0

    # Identical box IoU = 1.0
    assert box1.compute_iou(box1) == 1.0

    # Non-overlapping box IoU = 0.0
    box2 = BoundingBox(x1=50.0, y1=50.0, x2=70.0, y2=70.0)
    assert box1.compute_iou(box2) == 0.0

    # 50% overlapping box
    box3 = BoundingBox(x1=20.0, y1=10.0, x2=40.0, y2=30.0)
    # intersection: 10 * 20 = 200, union: 400 + 400 - 200 = 600 -> iou = 200 / 600 = 1/3
    assert abs(box1.compute_iou(box3) - (1.0 / 3.0)) < 1e-6

    # Coordinate conversions
    norm = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    abs_box = norm.to_absolute(1000, 500)
    assert abs_box.x1 == 100.0
    assert abs_box.y1 == 100.0
    assert abs_box.x2 == 500.0
    assert abs_box.y2 == 300.0
