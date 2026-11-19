"""
BoundingBox implementation mirroring com.trafficsign.data.BoundingBox.
Supports normalized/absolute coordinates, IoU computation, clamping,
format conversions (xyxy, xywh, center_wh), and validations.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable representation of an axis-aligned bounding box.
    Coordinates are defined as [x1, y1, x2, y2] where (x1, y1) is top-left
    and (x2, y2) is bottom-right.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if math.isnan(self.x1) or math.isnan(self.y1) or math.isnan(self.x2) or math.isnan(self.y2):
            raise ValueError(f"BoundingBox coordinates cannot be NaN: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_center_wh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        half_w = w / 2.0
        half_h = h / 2.0
        return cls(x1=cx - half_w, y1=cy - half_h, x2=cx + half_w, y2=cy + half_h)

    def compute_iou(self, other: BoundingBox) -> float:
        if other is None:
            return 0.0

        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h

        if intersection_area <= 0.0:
            return 0.0

        union_area = self.area + other.area - intersection_area
        return (intersection_area / union_area) if union_area > 0.0 else 0.0

    def to_absolute(self, frame_width: int, frame_height: int) -> BoundingBox:
        return BoundingBox(
            x1=self.x1 * frame_width,
            y1=self.y1 * frame_height,
            x2=self.x2 * frame_width,
            y2=self.y2 * frame_height
        )

    def to_normalized(self, frame_width: int, frame_height: int) -> BoundingBox:
        fw = float(frame_width)
        fh = float(frame_height)
        return BoundingBox(
            x1=self.x1 / fw,
            y1=self.y1 / fh,
            x2=self.x2 / fw,
            y2=self.y2 / fh
        )

    def clamp(self, min_x: float = 0.0, min_y: float = 0.0, max_x: float = 1.0, max_y: float = 1.0) -> BoundingBox:
        cx1 = max(min_x, min(self.x1, max_x))
        cy1 = max(min_y, min(self.y1, max_y))
        cx2 = max(min_x, min(self.x2, max_x))
        cy2 = max(min_y, min(self.y2, max_y))
        return BoundingBox(
            x1=min(cx1, cx2),
            y1=min(cy1, cy2),
            x2=max(cx1, cx2),
            y2=max(cy1, cy2)
        )

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
