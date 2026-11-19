"""
Detection data structures and utilities mirroring com.trafficsign.data.detection.
Includes DetectionAnnotation, DetectionSample, DetectionPrediction, and DetectionUtils.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from trafficsign.data.bounding_box import BoundingBox


@dataclass
class DetectionAnnotation:
    box: BoundingBox
    class_id: int
    class_name: str


@dataclass
class DetectionSample:
    image_path: Path
    annotations: List[DetectionAnnotation] = field(default_factory=list)
    width: int = 0
    height: int = 0


@dataclass
class DetectionPrediction:
    box: BoundingBox
    confidence: float
    grid_row: int = 0
    grid_col: int = 0


class DetectionUtils:
    """
    Grid decoding, target encoding, and Non-Maximum Suppression (NMS) utilities.
    """

    @staticmethod
    def sigmoid(x: float) -> float:
        if x < -45.0:
            return 0.0
        if x > 45.0:
            return 1.0
        return float(1.0 / (1.0 + math.exp(-x)))

    @classmethod
    def box_to_target(cls, box: BoundingBox, grid_cols: int, grid_rows: int) -> Tuple[float, float, float, float, int, int]:
        bx = (box.x1 + box.x2) / 2.0
        by = (box.y1 + box.y2) / 2.0
        bw = box.x2 - box.x1
        bh = box.y2 - box.y1

        col = int(math.floor(bx * grid_cols))
        row = int(math.floor(by * grid_rows))

        col = max(0, min(col, grid_cols - 1))
        row = max(0, min(row, grid_rows - 1))

        cell_x = bx * grid_cols - col
        cell_y = by * grid_rows - row

        return (float(cell_x), float(cell_y), float(bw), float(bh), row, col)

    @classmethod
    def target_to_box(
        cls,
        cell_x: float,
        cell_y: float,
        bw: float,
        bh: float,
        row: int,
        col: int,
        grid_cols: int,
        grid_rows: int
    ) -> BoundingBox:
        bx = (col + cell_x) / float(grid_cols)
        by = (row + cell_y) / float(grid_rows)

        x1 = max(0.0, bx - bw / 2.0)
        y1 = max(0.0, by - bh / 2.0)
        x2 = min(1.0, bx + bw / 2.0)
        y2 = min(1.0, by + bh / 2.0)

        if x2 <= x1:
            x2 = min(1.0, x1 + 0.01)
        if y2 <= y1:
            y2 = min(1.0, y1 + 0.01)

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def decode_grid_predictions(
        cls,
        grid_tensor: Union[np.ndarray, Sequence[Sequence[Sequence[float]]]],
        confidence_threshold: float,
        grid_cols: int,
        grid_rows: int,
        clamp_coordinates_directly: bool = False,
        use_linear_confidence: bool = False
    ) -> List[DetectionPrediction]:
        arr = np.array(grid_tensor, dtype=np.float32)
        raw_detections: List[DetectionPrediction] = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                conf_logit = float(arr[0, r, c])
                conf = max(0.0, min(1.0, conf_logit)) if use_linear_confidence else cls.sigmoid(conf_logit)

                if conf >= confidence_threshold:
                    raw_cell_x = float(arr[1, r, c])
                    raw_cell_y = float(arr[2, r, c])

                    cell_x = max(0.0, min(1.0, raw_cell_x)) if clamp_coordinates_directly else cls.sigmoid(raw_cell_x)
                    cell_y = max(0.0, min(1.0, raw_cell_y)) if clamp_coordinates_directly else cls.sigmoid(raw_cell_y)

                    bw = float(math.exp(arr[3, r, c])) / float(grid_cols)
                    bh = float(math.exp(arr[4, r, c])) / float(grid_rows)

                    box = cls.target_to_box(cell_x, cell_y, bw, bh, r, c, grid_cols, grid_rows)
                    raw_detections.append(DetectionPrediction(box=box, confidence=conf, grid_row=r, grid_col=c))

        return raw_detections

    @classmethod
    def non_maximum_suppression(
        cls,
        detections: Sequence[DetectionPrediction],
        iou_threshold: float
    ) -> List[DetectionPrediction]:
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        selected: List[DetectionPrediction] = []
        suppressed = [False] * len(sorted_dets)

        for i, current in enumerate(sorted_dets):
            if suppressed[i]:
                continue
            selected.append(current)

            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue
                candidate = sorted_dets[j]
                iou = current.box.compute_iou(candidate.box)
                if iou >= iou_threshold:
                    suppressed[j] = True

        return selected
