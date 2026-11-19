"""
Phase 5: Baseline Model Evaluation Script.
Evaluates the existing Task 11 champion detector and Task 7 champion classifier on:
1. Canonical 174-frame validation partition (87 positive frames, 87 background frames, 122 GT signs).
2. Classification validation partition (900 validation images across 85 Indian traffic sign classes).
Calculates:
- Precision, Recall, F1, TP, FP, FN at IoU 0.50 and IoU 0.30
- mAP@50 and mAP@50-95
- Confusion matrix and PR curves
- Latency (ms) and Throughput (FPS)
- Model size (MB)
Saves results to reports/baseline_metrics.json.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trafficsign.neural.models import HighResSignDetector, IndianTrafficSignCNN
from trafficsign.data.tar_loader import IndianRoadTarLoader
from trafficsign.data.detection import DetectionUtils, DetectionPrediction
from trafficsign.data.bounding_box import BoundingBox
from trafficsign.data.preprocessor import ImagePreprocessor
from trafficsign.data.taxonomy import IndianTrafficSignTaxonomy

def calculate_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    return b1.compute_iou(b2)

def evaluate_detector(
    detector: HighResSignDetector,
    val_frames: List[Dict],
    conf_thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    iou_eval_thresholds: List[float] = [0.30, 0.50],
    device: str = "cpu"
) -> Dict:
    detector.eval()
    detector.to(device)
    
    preprocess = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    
    latencies = []
    
    # Store all frame predictions: list of (predictions, gt_boxes, is_positive)
    frame_results = []
    
    total_frames = len(val_frames)
    total_gt_boxes = sum(len(f["boxes"]) for f in val_frames)
    
    print(f"Evaluating detector on {total_frames} validation frames ({total_gt_boxes} GT boxes)...")
    
    for f in val_frames:
        img: Image.Image = f["image"]
        t0 = time.perf_counter()
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = detector(inp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        grid_tensor = out[0].cpu().numpy() # [5, 16, 16]
        
        # Raw decoded detections with minimal threshold (0.05) so we can filter at any threshold
        raw_dets = DetectionUtils.decode_grid_predictions(
            grid_tensor=grid_tensor,
            confidence_threshold=0.05,
            grid_cols=16,
            grid_rows=16
        )
        frame_results.append({
            "stem": f["stem"],
            "has_sign": f["has_sign"],
            "gt_boxes": f["boxes"],
            "raw_dets": raw_dets
        })
        
    avg_latency = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Evaluate across confidence threshold sweep
    sweep_results = []
    
    for conf_th in conf_thresholds:
        point_stats = {}
        for iou_th in iou_eval_thresholds:
            tp = 0
            fp = 0
            fn = 0
            total_proposals = 0
            pos_proposals = 0
            bg_proposals = 0
            fp_pos = 0
            fp_bg = 0
            
            for item in frame_results:
                # Filter by confidence and apply NMS at IoU 0.30
                filtered = [d for d in item["raw_dets"] if d.confidence >= conf_th]
                suppressed = DetectionUtils.non_maximum_suppression(filtered, iou_threshold=0.30)
                
                num_props = len(suppressed)
                total_proposals += num_props
                if item["has_sign"]:
                    pos_proposals += num_props
                else:
                    bg_proposals += num_props
                    fp_bg += num_props
                    
                gt_boxes = item["gt_boxes"]
                matched_gt = set()
                
                for det in suppressed:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, gt in enumerate(gt_boxes):
                        iou = calculate_iou(det.box, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                            
                    if best_iou >= iou_th and best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                        if item["has_sign"]:
                            fp_pos += 1
                            
                fn += (len(gt_boxes) - len(matched_gt))
                
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            key_iou = f"iou_{int(iou_th * 100)}"
            point_stats[key_iou] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "total_proposals": total_proposals,
                "pos_proposals": pos_proposals,
                "bg_proposals": bg_proposals,
                "fp_pos": fp_pos,
                "fp_bg": fp_bg,
                "proposals_per_frame": round(total_proposals / total_frames, 4),
                "bg_proposals_per_frame": round(bg_proposals / (total_frames // 2), 4)
            }
            
        sweep_results.append({
            "confidence_threshold": conf_th,
            "metrics": point_stats
        })

    # Compute mAP@50 and mAP@50-95 using PR curve across dense thresholds
    coco_iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    dense_conf_sweep = [float(x) for x in np.linspace(0.05, 0.95, 19)]
    
    # Precompute suppressed detections for each confidence threshold across all frames
    cached_suppressed_by_conf = {}
    for c_th in dense_conf_sweep:
        cached_suppressed_by_conf[c_th] = []
        for item in frame_results:
            filtered = [d for d in item["raw_dets"] if d.confidence >= c_th]
            supp = DetectionUtils.non_maximum_suppression(filtered, iou_threshold=0.30)
            cached_suppressed_by_conf[c_th].append(supp)

    ap_per_iou = []
    pr_curve_50 = []
    
    for iou_th in coco_iou_thresholds:
        precisions = []
        recalls = []
        for c_th in dense_conf_sweep:
            tp = 0
            fp = 0
            fn = 0
            for idx, item in enumerate(frame_results):
                suppressed = cached_suppressed_by_conf[c_th][idx]
                gt_boxes = item["gt_boxes"]
                matched_gt = set()
                for det in suppressed:
                    best_iou = max((calculate_iou(det.box, gt) for gt in gt_boxes), default=0.0)
                    best_idx = next((i for i, gt in enumerate(gt_boxes) if calculate_iou(det.box, gt) == best_iou), -1)
                    if best_iou >= iou_th and best_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_idx)
                    else:
                        fp += 1
                fn += (len(gt_boxes) - len(matched_gt))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            if abs(iou_th - 0.50) < 1e-4:
                pr_curve_50.append({"conf": round(float(c_th), 3), "precision": round(p, 4), "recall": round(r, 4)})
                
        # 11-point interpolated AP
        interp_recalls = np.linspace(0.0, 1.0, 11)
        interp_precisions = []
        for r_target in interp_recalls:
            p_above = [p for p, r in zip(precisions, recalls) if r >= r_target]
            interp_precisions.append(max(p_above) if p_above else 0.0)
        ap = float(np.mean(interp_precisions))
        ap_per_iou.append(ap)
        
    map_50 = ap_per_iou[0]
    map_50_95 = float(np.mean(ap_per_iou))

    # Select standard operating point (conf = 0.50)
    op_point = next(item for item in sweep_results if abs(item["confidence_threshold"] - 0.50) < 1e-4)

    return {
        "model_name": "HighResSignDetector (Task 11 Baseline Champion)",
        "total_val_frames": total_frames,
        "total_gt_boxes": total_gt_boxes,
        "operating_point_conf_0_50": op_point,
        "map_50": round(map_50, 4),
        "map_50_95": round(map_50_95, 4),
        "ap_per_iou": {f"iou_{int(iou*100)}": round(ap, 4) for iou, ap in zip(coco_iou_thresholds, ap_per_iou)},
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
        "device": device,
        "pr_curve_iou_50": pr_curve_50,
        "threshold_sweep": sweep_results
    }

def evaluate_classifier(
    classifier: IndianTrafficSignCNN,
    splits_path: Path,
    dataset_root: Path,
    device: str = "cpu"
) -> Dict:
    classifier.eval()
    classifier.to(device)
    
    preprocess = T.Compose([
        T.Resize((48, 48)),
        T.ToTensor()
    ])
    
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    val_entries = [e for e in splits.get("entries", []) if e.get("partition") == "VALIDATION"]
    print(f"Evaluating classifier on {len(val_entries)} validation images across classes...")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    per_class_tp = Counter()
    per_class_total = Counter()
    per_class_pred = Counter()
    
    # 85x85 confusion matrix
    confusion = np.zeros((85, 85), dtype=np.int32)
    latencies = []
    
    for entry in val_entries:
        rel_path = entry.get("relativePath") or entry.get("relative_path")
        img_path = dataset_root / rel_path
        if not img_path.exists():
            continue
            
        gt_cls = int(entry.get("classId") or entry.get("class_id", 0))
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                inp = preprocess(img_rgb).unsqueeze(0).to(device)
                
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = classifier(inp)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_cls = int(np.argmax(probs))
            top5 = np.argsort(probs)[-5:][::-1]
            
            confusion[gt_cls, pred_cls] += 1
            per_class_total[gt_cls] += 1
            per_class_pred[pred_cls] += 1
            
            if pred_cls == gt_cls:
                correct_top1 += 1
                per_class_tp[gt_cls] += 1
            if gt_cls in top5:
                correct_top5 += 1
            total += 1
        except Exception:
            continue
            
    top1_acc = (correct_top1 / total) if total > 0 else 0.0
    top5_acc = (correct_top5 / total) if total > 0 else 0.0
    avg_lat = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    per_class_report = {}
    for c in range(85):
        tp_c = per_class_tp[c]
        tot_c = per_class_total[c]
        pred_c = per_class_pred[c]
        p_c = tp_c / pred_c if pred_c > 0 else 0.0
        r_c = tp_c / tot_c if tot_c > 0 else 0.0
        f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        per_class_report[str(c)] = {
            "gt_count": tot_c,
            "pred_count": pred_c,
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f1_c, 4)
        }

    return {
        "model_name": "IndianTrafficSignCNN (Task 7 Baseline Champion)",
        "total_evaluated": total,
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "latency_ms": round(avg_lat, 2),
        "fps": round(fps, 1),
        "per_class_summary": per_class_report,
        "confusion_matrix": confusion.tolist()
    }

def main():
    print("=" * 60)
    print("PHASE 5: BASELINE MODEL BENCHMARK EVALUATION")
    print("=" * 60)
    
    # Check GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation device: {device}")
    
    # 1. Load Detector
    detector_path = REPO_ROOT / "models" / "checkpoints" / "detector_task11_champion" / "sign_detector-0005.params"
    detector = HighResSignDetector()
    detector.load_djl_checkpoint(detector_path)
    det_size_mb = detector_path.stat().st_size / (1024 * 1024)
    
    # Load Validation Frames from cache
    loader = IndianRoadTarLoader()
    val_frames = loader.load_partition("val", cache_dir=REPO_ROOT / "data" / "processed" / "val_174")
    
    det_metrics = evaluate_detector(detector, val_frames, device=device)
    det_metrics["model_size_mb"] = round(det_size_mb, 2)
    
    # 2. Load Classifier
    classifier_path = REPO_ROOT / "models" / "checkpoints" / "task7_champion" / "exp_b_48x48-0008.params"
    classifier = IndianTrafficSignCNN(num_classes=85, input_resolution=48)
    classifier.load_djl_checkpoint(classifier_path)
    cls_size_mb = classifier_path.stat().st_size / (1024 * 1024)
    
    cls_splits_path = REPO_ROOT / "data" / "splits" / "indian_traffic_sign_splits.json"
    cls_root = REPO_ROOT / "data" / "raw" / "traffic_sign_classification_dataset"
    
    cls_metrics = evaluate_classifier(classifier, cls_splits_path, cls_root, device=device)
    cls_metrics["model_size_mb"] = round(cls_size_mb, 2)
    
    # Full baseline metrics report
    baseline_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "detector_baseline": det_metrics,
        "classifier_baseline": cls_metrics,
        "end_to_end_summary": {
            "detector_precision_iou50": det_metrics["operating_point_conf_0_50"]["metrics"]["iou_50"]["precision"],
            "detector_recall_iou50": det_metrics["operating_point_conf_0_50"]["metrics"]["iou_50"]["recall"],
            "detector_f1_iou50": det_metrics["operating_point_conf_0_50"]["metrics"]["iou_50"]["f1"],
            "detector_map50": det_metrics["map_50"],
            "detector_map50_95": det_metrics["map_50_95"],
            "classifier_top1_accuracy": cls_metrics["top1_accuracy"],
            "classifier_top5_accuracy": cls_metrics["top5_accuracy"],
            "detector_fps": det_metrics["fps"],
            "classifier_fps": cls_metrics["fps"],
            "detector_size_mb": det_size_mb,
            "classifier_size_mb": cls_size_mb
        }
    }
    
    out_path = REPO_ROOT / "reports" / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2)
        
    print(f"\nBaseline metrics successfully saved to {out_path}")
    print("\n--- BASELINE RESULTS SUMMARY ---")
    print(f"Detector (conf=0.50, IoU=0.50):")
    print(f"  Precision: {baseline_report['end_to_end_summary']['detector_precision_iou50']:.4f}")
    print(f"  Recall:    {baseline_report['end_to_end_summary']['detector_recall_iou50']:.4f}")
    print(f"  F1:        {baseline_report['end_to_end_summary']['detector_f1_iou50']:.4f}")
    print(f"  mAP@50:    {baseline_report['end_to_end_summary']['detector_map50']:.4f}")
    print(f"  mAP@50-95: {baseline_report['end_to_end_summary']['detector_map50_95']:.4f}")
    print(f"  Latency:   {det_metrics['latency_ms']} ms ({det_metrics['fps']} FPS)")
    print(f"Classifier (85 Classes):")
    print(f"  Top-1 Accuracy: {cls_metrics['top1_accuracy'] * 100.0:.2f}%")
    print(f"  Top-5 Accuracy: {cls_metrics['top5_accuracy'] * 100.0:.2f}%")
    print(f"  Latency:        {cls_metrics['latency_ms']} ms ({cls_metrics['fps']} FPS)")

if __name__ == "__main__":
    main()
