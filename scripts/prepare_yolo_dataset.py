"""
Prepares the Indian Road Driving Detection Dataset in standard YOLO format.
Uses the sequence-partitioned splits from data/splits/indian_road_detection_expanded_splits.json:
- Validation: 174 frames (from data/processed/val_174/)
- Test: 200 frames (from data/processed/test_200/)
- Train: Extracted from tar archives (positive frames + sampled background frames)
Generates:
- images/train, images/val, images/test
- labels/train, labels/val, labels/test
- data.yaml
"""

import os
import sys
import json
import tarfile
import io
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

def prepare_yolo_dataset(max_train_positive: int = 1200, max_train_background: int = 600):
    print("Preparing YOLO Indian Road Detection Dataset...")
    
    yolo_root = REPO_ROOT / "data" / "processed" / "yolo_indian_road"
    for split in ["train", "val", "test"]:
        os.makedirs(yolo_root / "images" / split, exist_ok=True)
        os.makedirs(yolo_root / "labels" / split, exist_ok=True)
        
    # 1. Validation Set (already cached in val_174)
    val_meta_path = REPO_ROOT / "data" / "processed" / "val_174" / "val_meta.json"
    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_meta = json.load(f)
        
    print(f"Exporting {len(val_meta)} validation frames to YOLO format...")
    for item in val_meta:
        stem = item["stem"]
        src_img = REPO_ROOT / "data" / "processed" / "val_174" / item["filename"]
        dst_img = yolo_root / "images" / "val" / f"{stem}.jpg"
        if not dst_img.exists() and src_img.exists():
            # Create hard link or copy
            try:
                os.link(src_img, dst_img)
            except Exception:
                with open(src_img, "rb") as s, open(dst_img, "wb") as d:
                    d.write(s.read())
                    
        # Write YOLO label format: 0 x_center y_center width height
        dst_lbl = yolo_root / "labels" / "val" / f"{stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in item["boxes"]:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                w = max(0.001, x2 - x1)
                h = max(0.001, y2 - y1)
                cx = min(1.0, max(0.0, x1 + w / 2.0))
                cy = min(1.0, max(0.0, y1 + h / 2.0))
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 2. Test Set (already cached in test_200)
    test_meta_path = REPO_ROOT / "data" / "processed" / "test_200" / "test_meta.json"
    with open(test_meta_path, "r", encoding="utf-8") as f:
        test_meta = json.load(f)
        
    print(f"Exporting {len(test_meta)} test frames to YOLO format...")
    for item in test_meta:
        stem = item["stem"]
        src_img = REPO_ROOT / "data" / "processed" / "test_200" / item["filename"]
        dst_img = yolo_root / "images" / "test" / f"{stem}.jpg"
        if not dst_img.exists() and src_img.exists():
            try:
                os.link(src_img, dst_img)
            except Exception:
                with open(src_img, "rb") as s, open(dst_img, "wb") as d:
                    d.write(s.read())
                    
        dst_lbl = yolo_root / "labels" / "test" / f"{stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in item["boxes"]:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                w = max(0.001, x2 - x1)
                h = max(0.001, y2 - y1)
                cx = min(1.0, max(0.0, x1 + w / 2.0))
                cy = min(1.0, max(0.0, y1 + h / 2.0))
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 3. Training Set (Extract from tar archives)
    splits_path = REPO_ROOT / "data" / "splits" / "indian_road_detection_expanded_splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    train_samples = [s for s in splits.get("samples", []) if s.get("partition") == "TRAIN"]
    pos_train = [s for s in train_samples if s.get("hasSign")]
    bg_train = [s for s in train_samples if not s.get("hasSign")]
    
    selected_pos = pos_train[:max_train_positive]
    selected_bg = bg_train[:max_train_background]
    selected_train = selected_pos + selected_bg
    
    print(f"Selected {len(selected_pos)} positive + {len(selected_bg)} background frames = {len(selected_train)} total for training.")
    
    # Group by archive
    by_archive = {}
    for r in selected_train:
        arch = r.get("archiveName")
        stem = r.get("frameStem")
        if arch and stem:
            by_archive.setdefault(arch, {})[stem] = r
            
    base_shards = REPO_ROOT / "data" / "raw" / "indian-road-driving-sample"
    add_shards = base_shards / "additional-shards"
    
    extracted_count = 0
    for arch_name, stem_map in by_archive.items():
        tar_path = base_shards / arch_name
        if not tar_path.exists():
            tar_path = add_shards / arch_name
        if not tar_path.exists():
            continue
            
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isdir():
                    continue
                fname = Path(member.name).name
                if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
                    continue
                m_stem = Path(member.name).stem
                if m_stem in stem_map:
                    rec = stem_map[m_stem]
                    dst_img = yolo_root / "images" / "train" / f"{m_stem}.jpg"
                    if not dst_img.exists():
                        f_obj = tar.extractfile(member)
                        if f_obj is not None:
                            with open(dst_img, "wb") as out_f:
                                out_f.write(f_obj.read())
                                
                    dst_lbl = yolo_root / "labels" / "train" / f"{m_stem}.txt"
                    with open(dst_lbl, "w", encoding="utf-8") as out_lbl:
                        boxes = rec.get("boxes", [])
                        for b in boxes:
                            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                            if x2 > 1.0 or y2 > 1.0:
                                x1, y1, x2, y2 = x1 / 1920.0, y1 / 994.0, x2 / 1920.0, y2 / 994.0
                            w = max(0.001, x2 - x1)
                            h = max(0.001, y2 - y1)
                            cx = min(1.0, max(0.0, x1 + w / 2.0))
                            cy = min(1.0, max(0.0, y1 + h / 2.0))
                            out_lbl.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    extracted_count += 1
                    
    print(f"Extracted and formatted {extracted_count} training frames.")
    
    # 4. Write data.yaml
    yaml_content = f"""path: {yolo_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: traffic sign
"""
    yaml_path = yolo_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset config saved to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    prepare_yolo_dataset()
