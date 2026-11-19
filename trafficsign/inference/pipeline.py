"""
Two-Stage Road Scene Traffic Sign Inference Pipeline
mirroring com.trafficsign.inference.RoadSceneInferencePipeline.
Stage 1: Sign proposal detector (128x128 input -> 16x16 grid decoding + NMS).
Stage 2: 85-class IRC:67 classifier on cropped ROIs.
Visual overlay annotation with uncertainty-color-coded bounding boxes and HUD labels.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from trafficsign.data.bounding_box import BoundingBox
from trafficsign.data.detection import DetectionPrediction, DetectionUtils
from trafficsign.data.preprocessor import ImagePreprocessor
from trafficsign.data.taxonomy import IndianTrafficSignTaxonomy
from trafficsign.neural.models import CompactSignDetector, HighResSignDetector, IndianTrafficSignCNN


class UncertaintyState(str, Enum):
    CONFIDENT = "CONFIDENT"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
    LOW_CLASSIFICATION_CONFIDENCE = "LOW_CLASSIFICATION_CONFIDENCE"
    UNKNOWN_CLASS = "UNKNOWN_CLASS"


@dataclass(frozen=True)
class RoadSceneDetection:
    absolute_box: BoundingBox
    normalized_box: BoundingBox
    detection_confidence: float
    predicted_class_id: int
    predicted_class_name: str
    category: str
    safety_implication: str
    classification_confidence: float
    uncertainty_state: UncertaintyState
    display_label: str


@dataclass
class PipelineResult:
    image_width: int
    image_height: int
    detection_latency_ms: int
    classification_latency_ms: int
    detections: List[RoadSceneDetection]
    crop_latency_ms: int = 0
    classifier_forward_latency_ms: int = 0
    raw_candidate_count: int = 0

    @property
    def total_latency_ms(self) -> int:
        return self.detection_latency_ms + self.classification_latency_ms

    @property
    def throughput_fps(self) -> float:
        tot = self.total_latency_ms
        return (1000.0 / tot) if tot > 0 else 0.0


class RoadSceneInferencePipeline:
    """
    Two-stage Road Scene Traffic Sign Inference Pipeline.
    """

    def __init__(
        self,
        detector_checkpoint: Optional[Union[str, Path]] = None,
        classifier_checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        classifier_resolution: int = 48
    ):
        self.device = device or torch.device("cpu")
        self.taxonomy = IndianTrafficSignTaxonomy()
        self.classifier_resolution = classifier_resolution
        self.detector_preprocessor = ImagePreprocessor(128, 128, False)
        self.classifier_preprocessor = ImagePreprocessor(classifier_resolution, classifier_resolution, False)

        self.detection_confidence_threshold: float = 0.50
        self.classification_confidence_threshold: float = 0.25
        self.nms_iou_threshold: float = 0.40
        self.max_proposals_per_frame: int = 25
        self.batch_classifier_inference: bool = True

        # Resolve detector checkpoint
        det_path = self._resolve_detector_path(detector_checkpoint)
        is_high_res = "16x16" in str(det_path).lower() or "highres" in str(det_path).lower() or "task" in str(det_path).lower() or "box16" in str(det_path).lower()

        if is_high_res:
            self.detector_model = HighResSignDetector().to(self.device)
        else:
            self.detector_model = CompactSignDetector().to(self.device)

        if det_path and det_path.exists():
            self.detector_model.load_djl_checkpoint(det_path)
        self.detector_model.eval()

        # Resolve classifier checkpoint
        cls_path = self._resolve_classifier_path(classifier_checkpoint)
        self.classifier_model = IndianTrafficSignCNN(85, self.classifier_resolution).to(self.device)
        if cls_path and cls_path.exists():
            self.classifier_model.load_djl_checkpoint(cls_path)
        self.classifier_model.eval()

    def _resolve_detector_path(self, checkpoint: Optional[Union[str, Path]]) -> Optional[Path]:
        if checkpoint is not None:
            p = Path(checkpoint)
            if p.is_file():
                return p
            if p.is_dir():
                for f in p.glob("*.params"):
                    return f

        candidates = [
            Path("models/checkpoints/detector_task11_champion/sign_detector-0005.params"),
            Path("S:/TrafficSignRecognitionRoadSafetyMonitor/models/checkpoints/detector_task11_champion/sign_detector-0005.params"),
            Path("models/checkpoints/detector_task15_champion/sign_detector-0004.params"),
            Path("models/checkpoints/detector_task14_champion/sign_detector-0002.params"),
            Path("models/checkpoints/detector_task13_calibrated_norm/sign_detector-0004.params"),
            Path("models/checkpoints/detector/detector_best/sign_detector-0004.params"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _resolve_classifier_path(self, checkpoint: Optional[Union[str, Path]]) -> Optional[Path]:
        if checkpoint is not None:
            p = Path(checkpoint)
            if p.is_file():
                return p
            if p.is_dir():
                for f in p.glob("*.params"):
                    return f

        candidates = [
            Path("models/checkpoints/task7_champion/exp_b_48x48-0008.params"),
            Path("S:/TrafficSignRecognitionRoadSafetyMonitor/models/checkpoints/task7_champion/exp_b_48x48-0008.params"),
            Path("models/checkpoints/classifier_exp_b_48x48/exp_b_48x48-0020.params"),
            Path("models/checkpoints/weighted/indian_cnn_best/indian_cnn-0008.params"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def process_image(self, full_frame: Union[Image.Image, np.ndarray, str, Path]) -> PipelineResult:
        if isinstance(full_frame, (str, Path)):
            full_frame = Image.open(full_frame).convert("RGB")
        elif isinstance(full_frame, np.ndarray):
            full_frame = Image.fromarray(full_frame).convert("RGB")
        elif isinstance(full_frame, Image.Image):
            if full_frame.mode != "RGB":
                full_frame = full_frame.convert("RGB")

        w, h = full_frame.size

        # Stage 1: Detection
        det_t0 = time.time()
        det_chw = self.detector_preprocessor.to_tensor_chw(full_frame)
        det_tensor = torch.from_numpy(det_chw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.detector_model(det_tensor)  # [1, 5, grid_rows, grid_cols]
            out_arr = out.squeeze(0).cpu().numpy()

        grid_rows = out_arr.shape[1]
        grid_cols = out_arr.shape[2]

        raw_dets = DetectionUtils.decode_grid_predictions(
            out_arr, self.detection_confidence_threshold, grid_cols, grid_rows
        )
        filtered_dets = DetectionUtils.non_maximum_suppression(raw_dets, self.nms_iou_threshold)
        det_duration = int((time.time() - det_t0) * 1000)

        raw_candidate_count = len(filtered_dets)
        if len(filtered_dets) > self.max_proposals_per_frame:
            filtered_dets = sorted(filtered_dets, key=lambda d: d.confidence, reverse=True)[:self.max_proposals_per_frame]

        # Stage 2: Classification on crops
        cls_t0 = time.time()
        crop_duration = 0
        fwd_duration = 0
        results: List[RoadSceneDetection] = []

        if filtered_dets:
            crop_t0 = time.time()
            valid_candidates = []

            for det in filtered_dets:
                norm_box = det.box
                abs_box = norm_box.to_absolute(w, h)

                crop_x1 = max(0, min(int(round(abs_box.x1)), w - 1))
                crop_y1 = max(0, min(int(round(abs_box.y1)), h - 1))
                crop_x2 = max(crop_x1 + 1, min(int(round(abs_box.x2)), w))
                crop_y2 = max(crop_y1 + 1, min(int(round(abs_box.y2)), h))

                if (crop_x2 - crop_x1) <= 2 or (crop_y2 - crop_y1) <= 2:
                    continue

                crop_img = full_frame.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_chw = self.classifier_preprocessor.to_tensor_chw(crop_img)
                valid_candidates.append((abs_box, norm_box, det.confidence, crop_chw))

            crop_duration = int((time.time() - crop_t0) * 1000)

            k = len(valid_candidates)
            if k > 0:
                fwd_t0 = time.time()
                batch_crops = np.stack([c[3] for c in valid_candidates], axis=0)
                batch_tensor = torch.from_numpy(batch_crops).to(self.device)

                with torch.no_grad():
                    logits = self.classifier_model(batch_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                fwd_duration = int((time.time() - fwd_t0) * 1000)

                for i in range(k):
                    abs_box, norm_box, det_conf, _ = valid_candidates[i]
                    p_row = probs[i]
                    max_class_id = int(np.argmax(p_row))
                    max_prob = float(p_row[max_class_id])

                    if det_conf < self.detection_confidence_threshold:
                        state = UncertaintyState.LOW_DETECTION_CONFIDENCE
                    elif max_prob < self.classification_confidence_threshold:
                        state = UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE
                    else:
                        state = UncertaintyState.CONFIDENT

                    info = self.taxonomy.get_class_info(max_class_id)
                    class_name = info.canonical_name if info else "UNKNOWN"
                    category = info.category if info else "UNCERTAIN"
                    safety = info.safety_implication if info else ""

                    if state == UncertaintyState.CONFIDENT:
                        display_label = class_name
                    elif state == UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE:
                        display_label = f"UNCERTAIN ({class_name})"
                    else:
                        display_label = "LOW_CONFIDENCE"

                    results.append(RoadSceneDetection(
                        absolute_box=abs_box,
                        normalized_box=norm_box,
                        detection_confidence=det_conf,
                        predicted_class_id=max_class_id,
                        predicted_class_name=class_name,
                        category=category,
                        safety_implication=safety,
                        classification_confidence=max_prob,
                        uncertainty_state=state,
                        display_label=display_label
                    ))

        cls_duration = int((time.time() - cls_t0) * 1000)

        return PipelineResult(
            image_width=w,
            image_height=h,
            detection_latency_ms=det_duration,
            classification_latency_ms=cls_duration,
            detections=results,
            crop_latency_ms=crop_duration,
            classifier_forward_latency_ms=fwd_duration,
            raw_candidate_count=raw_candidate_count
        )

    def draw_detections(
        self,
        input_frame: Union[Image.Image, np.ndarray],
        result: PipelineResult
    ) -> Image.Image:
        if isinstance(input_frame, np.ndarray):
            annotated = Image.fromarray(input_frame).convert("RGB")
        else:
            annotated = input_frame.copy().convert("RGB")

        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        # Color palette
        color_map = {
            UncertaintyState.CONFIDENT: (0, 204, 68),                # Green
            UncertaintyState.LOW_CLASSIFICATION_CONFIDENCE: (255, 153, 0),  # Orange
            UncertaintyState.LOW_DETECTION_CONFIDENCE: (51, 153, 255),      # Blue
            UncertaintyState.UNKNOWN_CLASS: (230, 40, 40),          # Red
        }

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for det in result.detections:
            box = det.absolute_box
            bx = int(round(box.x1))
            by = int(round(box.y1))
            bw = int(round(box.x2 - box.x1))
            bh = int(round(box.y2 - box.y1))

            box_color = color_map.get(det.uncertainty_state, (255, 255, 0))

            # Bounding box (3px stroke)
            for offset in range(3):
                draw.rectangle(
                    [bx - offset, by - offset, bx + bw + offset, by + bh + offset],
                    outline=box_color
                )

            label_text = f"{det.display_label} (cls: {det.classification_confidence * 100:.0f}%, det: {det.detection_confidence * 100:.0f}%)"

            # Pill position
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]

            pill_x = max(0, min(bx, w - text_w - 14))
            pill_y = max(text_h + 6, by - 6)

            # Draw dark semi-transparent rounded pill
            pill_box = [pill_x, pill_y - text_h - 4, pill_x + text_w + 12, pill_y + 4]
            draw.rounded_rectangle(pill_box, radius=4, fill=(0, 0, 0, 200), outline=box_color, width=1)
            draw.text((pill_x + 6, pill_y - text_h - 2), label_text, fill=(255, 255, 255), font=font)

        return annotated
