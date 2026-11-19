"""
Phase 8: Real-World Inference Refinement & Evaluation.
Evaluates the retrained YOLO detector(s) and end-to-end Two-Stage Pipeline on the 200 held-out test frames:
- 100 positive frames (verified signs)
- 100 empty background frames (challenging negatives)
Analyzes:
- Confidence calibration & threshold sweep
- Separation of FP on empty frames vs positive frames
- Performance on small signs (< 32x32) vs medium/large signs
- End-to-end pipeline latency, FPS, and visual HUD overlays
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trafficsign.data.bounding_box import BoundingBox
from trafficsign.inference.pipeline import RoadSceneInferencePipeline


def load_test_manifest() -> List[dict]:
    manifest_path = REPO_ROOT / "data" / "processed" / "test_200" / "test_meta.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for m in meta:
        m["cached_image_path"] = f"data/processed/test_200/{m['filename']}"
        m["is_background"] = not m.get("has_sign", False)
        m["image_file"] = m["filename"]
    return meta


def evaluate_detector_thresholds(
    model_path: Path,
    test_manifest: List[dict],
    thresholds: List[float] = [0.15, 0.25, 0.35, 0.50, 0.60],
    iou_thresh: float = 0.50
) -> Dict[str, dict]:
    det_label = model_path.parent.parent.name if model_path.parent.name == "weights" else model_path.name
    print(f"\nEvaluating detector: {det_label} on {len(test_manifest)} test frames...")
    model = YOLO(str(model_path))

    # Pre-extract all raw detections across test frames at lowest conf threshold
    min_conf = min(thresholds)
    frame_preds = []
    t0 = time.time()

    for item in test_manifest:
        img_path = REPO_ROOT / item["cached_image_path"]
        res = model(str(img_path), conf=min_conf, iou=0.40, verbose=False, device=0)[0]
        h, w = res.orig_shape
        preds = []
        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            norm_box = BoundingBox(
                x1=float(xyxy[0]) / w,
                y1=float(xyxy[1]) / h,
                x2=float(xyxy[2]) / w,
                y2=float(xyxy[3]) / h
            )
            preds.append((norm_box, conf))
        frame_preds.append((preds, item["boxes"], item["is_background"]))

    elapsed = time.time() - t0
    avg_latency_ms = (elapsed / len(test_manifest)) * 1000
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    print(f"Extracted predictions in {elapsed:.2f}s ({avg_latency_ms:.1f} ms/frame, {fps:.1f} FPS)")

    threshold_results = {}
    for conf_t in thresholds:
        tp = 0
        fp_pos = 0
        fp_bg = 0
        fn = 0
        total_proposals = 0
        bg_frame_count = 0
        pos_frame_count = 0

        # Small vs large sign breakdown
        tp_small = 0
        fn_small = 0
        tp_large = 0
        fn_large = 0

        for preds, gt_boxes, is_bg in frame_preds:
            active_preds = [p[0] for p in preds if p[1] >= conf_t]
            total_proposals += len(active_preds)

            if is_bg:
                bg_frame_count += 1
                fp_bg += len(active_preds)
            else:
                pos_frame_count += 1
                matched_gt = set()
                pred_matched = [False] * len(active_preds)

                gt_objs = []
                for b in gt_boxes:
                    bb = BoundingBox(b["x1"], b["y1"], b["x2"], b["y2"])
                    gt_objs.append((bb, b.get("is_small", False)))

                for p_idx, pred_box in enumerate(active_preds):
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, (gt_box, _) in enumerate(gt_objs):
                        if g_idx in matched_gt:
                            continue
                        iou = pred_box.compute_iou(gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                    if best_iou >= iou_thresh:
                        matched_gt.add(best_gt_idx)
                        pred_matched[p_idx] = True

                frame_tp = len(matched_gt)
                frame_fp = len(active_preds) - frame_tp
                frame_fn = len(gt_objs) - frame_tp

                tp += frame_tp
                fp_pos += frame_fp
                fn += frame_fn

                for g_idx, (_, is_small) in enumerate(gt_objs):
                    if g_idx in matched_gt:
                        if is_small:
                            tp_small += 1
                        else:
                            tp_large += 1
                    else:
                        if is_small:
                            fn_small += 1
                        else:
                            fn_large += 1

        total_fp = fp_pos + fp_bg
        precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        bg_fp_rate = fp_bg / bg_frame_count if bg_frame_count > 0 else 0.0
        props_per_frame = total_proposals / len(test_manifest)

        small_recall = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        large_recall = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0

        threshold_results[f"conf_{conf_t:.2f}"] = {
            "confidence_threshold": conf_t,
            "iou_threshold": iou_thresh,
            "tp": tp,
            "fp_total": total_fp,
            "fp_positive_frames": fp_pos,
            "fp_background_frames": fp_bg,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "proposals_per_frame": round(props_per_frame, 2),
            "background_fp_per_frame": round(bg_fp_rate, 3),
            "small_sign_recall": round(small_recall, 4),
            "large_sign_recall": round(large_recall, 4),
        }

    return {
        "model_name": model_path.name,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(fps, 1),
        "operating_points": threshold_results
    }


def run_pipeline_demonstration(detector_path: Path, test_manifest: List[dict]):
    print(f"\nRunning Full End-to-End Pipeline Demonstration with {detector_path.name}...")
    pipeline = RoadSceneInferencePipeline(
        detector_checkpoint=detector_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    out_dir = REPO_ROOT / "reports" / "visual_eval_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick 5 positive frames and 2 background frames
    demo_samples = [item for item in test_manifest if not item["is_background"]][:5]
    demo_samples += [item for item in test_manifest if item["is_background"]][:2]

    demo_results = []
    for item in demo_samples:
        img_path = REPO_ROOT / item["cached_image_path"]
        result = pipeline.process_image(img_path)

        # Save annotated image
        annotated_img = pipeline.draw_detections(img_path, result)
        save_path = out_dir / f"annotated_{Path(item['image_file']).stem}.jpg"
        annotated_img.save(save_path)

        demo_results.append({
            "image": item["image_file"],
            "is_background": item["is_background"],
            "detections_count": len(result.detections),
            "detection_latency_ms": result.detection_latency_ms,
            "classification_latency_ms": result.classification_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "detected_signs": [
                {
                    "class_name": d.predicted_class_name,
                    "category": d.category,
                    "det_conf": round(d.detection_confidence, 3),
                    "cls_conf": round(d.classification_confidence, 3),
                    "uncertainty": d.uncertainty_state.value
                }
                for d in result.detections
            ]
        })

    return demo_results


def main():
    manifest = load_test_manifest()
    results = {}

    yolo26_pt = REPO_ROOT / "models" / "checkpoints" / "retrained_yolo26n" / "weights" / "best.pt"
    if yolo26_pt.exists():
        results["yolo26n"] = evaluate_detector_thresholds(yolo26_pt, manifest)

    yolo11_pt = REPO_ROOT / "models" / "checkpoints" / "retrained_yolo11n" / "weights" / "best.pt"
    if yolo11_pt.exists():
        results["yolo11n"] = evaluate_detector_thresholds(yolo11_pt, manifest)

    # Use the best detector for end-to-end pipeline demo
    best_pt = yolo26_pt if yolo26_pt.exists() else yolo11_pt
    if best_pt.exists():
        demo_results = run_pipeline_demonstration(best_pt, manifest)
        results["pipeline_demonstration"] = demo_results

    out_path = REPO_ROOT / "reports" / "inference_refinement_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nInference Refinement Report saved to: {out_path}")


if __name__ == "__main__":
    main()
