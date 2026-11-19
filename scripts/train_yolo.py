"""
Phase 7: Model Retraining Script for YOLO26n and YOLO11n on the Indian Road Dataset.
Trains using GPU acceleration on NVIDIA RTX 4050 (6 GB VRAM).
Configures specialized augmentations for Indian driving conditions (fliplr=0.0 to preserve sign semantics).
Monitors losses, logs metrics, and saves champion checkpoints.
"""

import os
import sys
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

def train_candidate(
    model_name: str,
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 512,
    patience: int = 5,
    project_dir: Path = REPO_ROOT / "models" / "checkpoints"
) -> dict:
    print(f"\n{'='*60}")
    print(f"TRAINING CANDIDATE: {model_name.upper()}")
    print(f"Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB)")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {imgsz}")
    print(f"{'='*60}")

    yaml_path = REPO_ROOT / "data" / "processed" / "yolo_indian_road" / "data.yaml"
    model = YOLO(f"{model_name}.pt")

    # Specialized training configuration for Indian traffic signs
    train_args = {
        "data": str(yaml_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": 0,
        "patience": patience,
        "save": True,
        "project": str(project_dir),
        "name": f"retrained_{model_name}",
        "exist_ok": True,
        # Augmentations: preserve sign semantics
        "fliplr": 0.0,       # CRITICAL: do not flip left/right directional signs
        "flipud": 0.0,
        "degrees": 5.0,      # slight camera tilt
        "hsv_h": 0.015,      # slight hue variation
        "hsv_s": 0.4,        # lighting/saturation variation
        "hsv_v": 0.4,        # brightness variation
        "translate": 0.1,    # camera shift
        "scale": 0.2,        # scale variations (distance)
        "mosaic": 0.5,       # multi-sign context
        "verbose": True
    }

    t0 = time.time()
    results = model.train(**train_args)
    duration = time.time() - t0

    print(f"\nTraining for {model_name} completed in {duration:.1f}s ({duration/60:.2f} min).")

    # Evaluate best checkpoint on validation partition
    best_pt = project_dir / f"retrained_{model_name}" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = project_dir / f"retrained_{model_name}" / "weights" / "last.pt"

    print(f"Validating best checkpoint: {best_pt}")
    best_model = YOLO(str(best_pt))
    val_results = best_model.val(data=str(yaml_path), split="val", imgsz=imgsz, device=0)
    
    # Evaluate on held-out test partition
    print(f"Evaluating on untouched test partition: {best_pt}")
    test_results = best_model.val(data=str(yaml_path), split="test", imgsz=imgsz, device=0)

    val_map50 = float(val_results.box.map50)
    val_map50_95 = float(val_results.box.map)
    val_p = float(val_results.box.mp)
    val_r = float(val_results.box.mr)
    val_f1 = float(2 * val_p * val_r / (val_p + val_r)) if (val_p + val_r) > 0 else 0.0

    test_map50 = float(test_results.box.map50)
    test_map50_95 = float(test_results.box.map)
    test_p = float(test_results.box.mp)
    test_r = float(test_results.box.mr)
    test_f1 = float(2 * test_p * test_r / (test_p + test_r)) if (test_p + test_r) > 0 else 0.0

    checkpoint_size_mb = best_pt.stat().st_size / (1024 * 1024) if best_pt.exists() else 0.0

    summary = {
        "candidate": model_name,
        "checkpoint_path": str(best_pt.relative_to(REPO_ROOT)),
        "checkpoint_size_mb": round(checkpoint_size_mb, 2),
        "training_duration_s": round(duration, 1),
        "epochs_completed": epochs,
        "validation_metrics": {
            "precision": round(val_p, 4),
            "recall": round(val_r, 4),
            "f1": round(val_f1, 4),
            "map50": round(val_map50, 4),
            "map50_95": round(val_map50_95, 4)
        },
        "test_metrics": {
            "precision": round(test_p, 4),
            "recall": round(test_r, 4),
            "f1": round(test_f1, 4),
            "map50": round(test_map50, 4),
            "map50_95": round(test_map50_95, 4)
        }
    }
    return summary

def main():
    os.makedirs(REPO_ROOT / "reports", exist_ok=True)
    results = {}

    # Primary candidate: YOLO26n
    results["yolo26n"] = train_candidate("yolo26n", epochs=10, batch_size=16, imgsz=512)

    # Comparison candidate: YOLO11n
    results["yolo11n"] = train_candidate("yolo11n", epochs=10, batch_size=16, imgsz=512)

    out_file = REPO_ROOT / "reports" / "retraining_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll training runs complete! Retraining report saved to {out_file}")
    print("\n--- RETRAINING RESULTS SUMMARY ---")
    for name, res in results.items():
        v = res["validation_metrics"]
        t = res["test_metrics"]
        print(f"[{name.upper()}] Val P: {v['precision']:.4f}, R: {v['recall']:.4f}, F1: {v['f1']:.4f}, mAP@50: {v['map50']:.4f} | Test mAP@50: {t['map50']:.4f}")

if __name__ == "__main__":
    main()
