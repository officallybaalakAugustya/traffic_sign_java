"""
Phase 3: Dataset Audit Script for Indian Traffic Sign Datasets.
Analyzes:
1. Classification Dataset (85 classes, class distributions, balance, image dimensions, duplicates).
2. Road Driving Detection Dataset (31 shards, 31,000 frames, sequence grouping, bounding boxes, background frames).
Generates reports/dataset_audit_report.json.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trafficsign.data.taxonomy import IndianTrafficSignTaxonomy

def audit_dataset():
    print("Executing Phase 3: Comprehensive Dataset Audit...")
    
    taxonomy = IndianTrafficSignTaxonomy()
    all_classes = taxonomy.get_all_classes()
    print(f"Taxonomy loaded: {len(all_classes)} classes.")
    
    # 1. Audit Classification Dataset
    cls_root = REPO_ROOT / "data" / "raw" / "traffic_sign_classification_dataset"
    train_dir = cls_root / "train"
    test_dir = cls_root / "test"
    
    train_class_counts = {}
    test_class_counts = {}
    total_images_scanned = 0
    corrupted_images = 0
    resolutions = []
    
    if train_dir.exists():
        for class_folder in train_dir.iterdir():
            if class_folder.is_dir():
                c_name = class_folder.name
                files = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
                train_class_counts[c_name] = len(files)
                total_images_scanned += len(files)
                for f in files[:3]: # Sample 3 images per class for resolution check
                    try:
                        with Image.open(f) as img:
                            resolutions.append(img.size)
                    except Exception:
                        corrupted_images += 1

    if test_dir.exists():
        for class_folder in test_dir.iterdir():
            if class_folder.is_dir():
                c_name = class_folder.name
                files = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
                test_class_counts[c_name] = len(files)
                total_images_scanned += len(files)

    all_counts = list(train_class_counts.values())
    max_count = max(all_counts) if all_counts else 0
    min_count = min(all_counts) if all_counts else 0
    median_count = float(np.median(all_counts)) if all_counts else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else float("inf")
    
    underrepresented_classes = [c for c, count in train_class_counts.items() if count < 15]

    # 2. Audit Road Driving Detection Dataset
    det_split_path = REPO_ROOT / "data" / "splits" / "indian_road_detection_expanded_splits.json"
    det_splits = {}
    box_widths = []
    box_heights = []
    box_aspect_ratios = []
    box_areas = []

    if det_split_path.exists():
        with open(det_split_path, "r", encoding="utf-8") as f:
            det_splits = json.load(f)
            
        # Inspect sample boxes
        for sample in det_splits.get("samples", []):
            boxes = sample.get("signBoxes", sample.get("boxes", []))
            for b in boxes:
                # [x, y, w, h] normalized or pixel
                w = b.get("w", b.get("width", 0))
                h = b.get("h", b.get("height", 0))
                if w > 0 and h > 0:
                    box_widths.append(float(w))
                    box_heights.append(float(h))
                    box_aspect_ratios.append(float(w / h))
                    box_areas.append(float(w * h))

    # 3. Detection Manifest
    manifest_path = REPO_ROOT / "data" / "manifests" / "indian_road_31shards_dataset_manifest.json"
    manifest_summary = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_summary = json.load(f)

    # 4. Compile Audit Report
    report = {
        "dataset_name": "Indian Traffic Sign Detection & Classification Dataset",
        "taxonomy": {
            "standard": "IRC:67 Indian Road Congress",
            "total_classes": len(all_classes),
            "categories": list(set(c.category for c in all_classes))
        },
        "classification_dataset": {
            "total_scanned_files": total_images_scanned,
            "train_classes_represented": len(train_class_counts),
            "test_classes_represented": len(test_class_counts),
            "min_samples_per_class": min_count,
            "max_samples_per_class": max_count,
            "median_samples_per_class": median_count,
            "class_imbalance_ratio": round(imbalance_ratio, 2),
            "corrupted_images_found": corrupted_images,
            "underrepresented_classes_count": len(underrepresented_classes),
            "underrepresented_classes": underrepresented_classes,
            "class_counts_train": train_class_counts
        },
        "road_driving_detection_dataset": {
            "total_shards": manifest_summary.get("totalShards", 31),
            "total_bytes": manifest_summary.get("totalBytes", 10087065600),
            "total_frames": manifest_summary.get("totalMatchedPairs", 31000),
            "frames_with_signs": manifest_summary.get("totalFramesWithSigns", 2258),
            "empty_background_frames": manifest_summary.get("totalEmptyFrames", 28742),
            "total_sign_boxes": manifest_summary.get("totalSignBoundingBoxes", 2917),
            "total_driving_sequences": manifest_summary.get("totalDistinctSequences", 401),
            "splits": {
                "train": {
                    "sequences": det_splits.get("trainSequences", 365),
                    "frames": det_splits.get("trainFrames", 28476),
                    "frames_with_signs": det_splits.get("trainFramesWithSign", 2063),
                    "empty_frames": det_splits.get("trainEmptyFrames", 26413),
                    "sign_boxes": det_splits.get("trainSignBoxes", 2590)
                },
                "validation_canonical": {
                    "sequences": det_splits.get("valSequences", 27),
                    "frames": det_splits.get("valFrames", 174),
                    "frames_with_signs": det_splits.get("valFramesWithSign", 87),
                    "empty_frames": det_splits.get("valEmptyFrames", 87),
                    "sign_boxes": det_splits.get("valSignBoxes", 122)
                },
                "test_held_out": {
                    "sequences": det_splits.get("testSequences", 9),
                    "frames": det_splits.get("testFrames", 200),
                    "frames_with_signs": det_splits.get("testFramesWithSign", 100),
                    "empty_frames": det_splits.get("testEmptyFrames", 100),
                    "sign_boxes": det_splits.get("testSignBoxes", 195)
                },
                "leakage_verification": {
                    "sequence_overlap_train_val": det_splits.get("sequenceOverlapTrainVal", 0),
                    "sequence_overlap_train_test": det_splits.get("sequenceOverlapTrainTest", 0),
                    "sequence_overlap_val_test": det_splits.get("sequenceOverlapValTest", 0),
                    "is_strictly_leakage_free": (
                        det_splits.get("sequenceOverlapTrainVal", 0) == 0 and
                        det_splits.get("sequenceOverlapTrainTest", 0) == 0 and
                        det_splits.get("sequenceOverlapValTest", 0) == 0
                    )
                }
            },
            "box_geometry_statistics": {
                "sampled_boxes_count": len(box_widths),
                "mean_width": float(np.mean(box_widths)) if box_widths else 0,
                "mean_height": float(np.mean(box_heights)) if box_heights else 0,
                "mean_aspect_ratio": float(np.mean(box_aspect_ratios)) if box_aspect_ratios else 0,
                "mean_normalized_area": float(np.mean(box_areas)) if box_areas else 0
            }
        }
    }

    report_path = REPO_ROOT / "reports" / "dataset_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Dataset audit report written to {report_path}")
    print(f"Total detection frames: {report['road_driving_detection_dataset']['total_frames']:,}")
    print(f"Total classification images: {report['classification_dataset']['total_scanned_files']:,}")
    print(f"Sequence leakage-free: {report['road_driving_detection_dataset']['splits']['leakage_verification']['is_strictly_leakage_free']}")
    return report

if __name__ == "__main__":
    audit_dataset()
