"""
Task 17 False-Positive and Safety-State Integrity Auditor
mirroring com.trafficsign.app.Task17AuditRunner.
Audits:
- A clear 40 km/h speed-limit sign
- A road image with no obvious traffic signs (empty negative background)
- A road image with several genuine signs
Saves comprehensive audit payload to reports/task17_audit_report.json.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from trafficsign.inference.pipeline import PipelineResult, RoadSceneInferencePipeline
from trafficsign.safety.catalogue import SignRuleCatalogue
from trafficsign.safety.engine import SafetyEvent, SafetyEventEngine
from trafficsign.safety.state import SafetyState


def compute_file_sha256(path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class Task17AuditRunner:
    """
    Task 17 False-Positive and Safety-State Integrity Audit Runner.
    """

    def __init__(self):
        self.pipeline = RoadSceneInferencePipeline()
        self.catalogue = SignRuleCatalogue()
        self.state = SafetyState()
        self.engine = SafetyEventEngine(self.catalogue, self.state)

    def execute(self) -> Dict[str, object]:
        print("=" * 80)
        print("TASK 17 - FALSE-POSITIVE AND SAFETY-STATE INTEGRITY AUDIT")
        print("=" * 80)

        # 1. Audit Checkpoints
        checkpoints_info = []
        det_path = Path("models/checkpoints/detector_task11_champion/sign_detector-0005.params")
        if not det_path.exists():
            det_path = Path("models/checkpoints/detector_task15_champion/sign_detector-0004.params")
        cls_path = Path("models/checkpoints/task7_champion/exp_b_48x48-0008.params")

        if det_path.exists():
            checkpoints_info.append({
                "modelType": "HighResSignDetector",
                "checkpointPath": str(det_path),
                "sha256Hex": compute_file_sha256(det_path),
                "fileSizeBytes": det_path.stat().st_size
            })

        if cls_path.exists():
            checkpoints_info.append({
                "modelType": "IndianTrafficSignCNN",
                "checkpointPath": str(cls_path),
                "sha256Hex": compute_file_sha256(cls_path),
                "fileSizeBytes": cls_path.stat().st_size
            })

        # 2. Test Cases
        cases = [
            ("SPEED_LIMIT_40", self._find_or_create_speed40_image()),
            ("BACKGROUND_EMPTY", self._find_or_create_empty_image()),
            ("GENUINE_MULTI_SIGN", self._find_or_create_multi_sign_image()),
        ]

        configs = [
            ("Default Operating Config", 0.50, 0.25, 0.40, 25),
            ("Relaxed Diagnostic Config", 0.20, 0.20, 0.25, 50),
        ]

        eval_cases = []

        for case_name, img_path in cases:
            print(f"\nEvaluating Case: {case_name} ({img_path})")
            raw_img = Image.open(img_path).convert("RGB")
            w, h = raw_img.size

            case_data = {
                "caseName": case_name,
                "inputPath": str(img_path),
                "imageWidth": w,
                "imageHeight": h,
                "configResults": {}
            }

            for cfg_name, det_thresh, cls_thresh, nms_thresh, prop_cap in configs:
                self.pipeline.detection_confidence_threshold = det_thresh
                self.pipeline.classification_confidence_threshold = cls_thresh
                self.pipeline.nms_iou_threshold = nms_thresh
                self.pipeline.max_proposals_per_frame = prop_cap

                self.engine.reset()
                res: PipelineResult = self.pipeline.process_image(raw_img)
                events: List[SafetyEvent] = self.engine.process_frame(1, 1000, raw_img, res.detections)

                hud_state = {
                    "currentPostedSpeedLimit": self.state.get_current_posted_speed_limit(),
                    "activeRestrictions": list(self.state.get_active_restrictions()),
                    "activeHazards": list(self.state.get_active_hazards())
                }

                dets_summary = []
                for d in res.detections:
                    dets_summary.append({
                        "classId": d.predicted_class_id,
                        "className": d.predicted_class_name,
                        "detConfidence": float(d.detection_confidence),
                        "clsConfidence": float(d.classification_confidence),
                        "uncertaintyState": str(d.uncertainty_state),
                        "normalizedBox": [d.normalized_box.x1, d.normalized_box.y1, d.normalized_box.x2, d.normalized_box.y2]
                    })

                events_summary = []
                for e in events:
                    events_summary.append({
                        "eventId": e.event_id,
                        "eventType": str(e.event_type),
                        "severity": str(e.severity),
                        "displayName": e.display_name,
                        "numericalValue": e.numerical_value,
                        "driverAdvisory": e.driver_advisory
                    })

                cfg_result = {
                    "detConfidenceThreshold": det_thresh,
                    "clsConfidenceThreshold": cls_thresh,
                    "nmsIouThreshold": nms_thresh,
                    "maxProposalsPerFrame": prop_cap,
                    "rawProposalsCount": res.raw_candidate_count,
                    "confidentDetectionsCount": len([d for d in res.detections if str(d.uncertainty_state) == "CONFIDENT"]),
                    "uncertainDetectionsCount": len([d for d in res.detections if str(d.uncertainty_state) != "CONFIDENT"]),
                    "detections": dets_summary,
                    "emittedEvents": events_summary,
                    "hudState": hud_state
                }
                case_data["configResults"][cfg_name] = cfg_result

                print(f"  [{cfg_name}] Proposals: {res.raw_candidate_count}, Detections: {len(res.detections)}, "
                      f"Events: {len(events)}, Posted Speed: {hud_state['currentPostedSpeedLimit']}")

            eval_cases.append(case_data)

        report = {
            "title": "Task 17 False-Positive and Safety-State Integrity Audit Report",
            "timestamp": datetime.datetime.now().isoformat(),
            "checkpoints": checkpoints_info,
            "evaluationCases": eval_cases
        }

        out_path = Path("reports/task17_audit_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\nAudit complete! Report saved to {out_path}")
        return report

    def _find_or_create_speed40_image(self) -> Path:
        p = Path("reports/evidence/speed_limit_40_test.png")
        if p.exists():
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        # Create standard speed limit 40 image
        img = Image.new("RGB", (640, 480), (120, 130, 140))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw road & sky
        draw.rectangle([0, 0, 640, 240], fill=(135, 180, 220))
        draw.rectangle([0, 240, 640, 480], fill=(70, 70, 75))
        # Draw speed limit sign
        cx, cy, r = 320, 240, 60
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(220, 20, 20), width=10)
        draw.text((cx - 20, cy - 20), "40", fill=(0, 0, 0))
        img.save(p)
        return p

    def _find_or_create_empty_image(self) -> Path:
        p = Path("reports/evidence/empty_background_test.png")
        if p.exists():
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 480), (100, 110, 120))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 640, 240], fill=(140, 170, 210))
        draw.rectangle([0, 240, 640, 480], fill=(60, 60, 65))
        img.save(p)
        return p

    def _find_or_create_multi_sign_image(self) -> Path:
        p = Path("reports/evidence/multi_sign_test.png")
        if p.exists():
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 480), (100, 110, 120))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 640, 240], fill=(140, 170, 210))
        draw.rectangle([0, 240, 640, 480], fill=(60, 60, 65))
        # Sign 1: Speed limit
        draw.ellipse([150, 150, 230, 230], fill=(255, 255, 255), outline=(220, 20, 20), width=8)
        draw.text((175, 175), "50", fill=(0, 0, 0))
        # Sign 2: Stop sign
        draw.polygon([(450, 150), (490, 150), (520, 180), (520, 220), (490, 250), (450, 250), (420, 220), (420, 180)], fill=(220, 20, 20))
        draw.text((455, 190), "STOP", fill=(255, 255, 255))
        img.save(p)
        return p


def main():
    auditor = Task17AuditRunner()
    auditor.execute()


if __name__ == "__main__":
    main()
