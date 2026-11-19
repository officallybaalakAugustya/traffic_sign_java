"""
Efficient tar-archive reader for Indian Road Driving Sample shards.
Extracts and caches partitioned frames (train, validation, test) without unpacking the entire 10 GB.
"""

from __future__ import annotations
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torchvision.transforms as T

from trafficsign.data.bounding_box import BoundingBox

class IndianRoadTarLoader:
    def __init__(
        self,
        splits_json_path: Optional[Union[str, Path]] = None,
        base_shards_dir: Optional[Union[str, Path]] = None,
        additional_shards_dir: Optional[Union[str, Path]] = None
    ):
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.splits_path = Path(splits_json_path) if splits_json_path else repo_root / "data" / "splits" / "indian_road_detection_expanded_splits.json"
        self.base_shards_dir = Path(base_shards_dir) if base_shards_dir else repo_root / "data" / "raw" / "indian-road-driving-sample"
        self.additional_shards_dir = Path(additional_shards_dir) if additional_shards_dir else self.base_shards_dir / "additional-shards"
        
        with open(self.splits_path, "r", encoding="utf-8") as f:
            self.splits_data = json.load(f)
            
        self.samples = self.splits_data.get("samples", [])
        
    def find_shard_path(self, archive_name: str) -> Optional[Path]:
        p1 = self.base_shards_dir / archive_name
        if p1.exists():
            return p1
        p2 = self.additional_shards_dir / archive_name
        if p2.exists():
            return p2
        return None

    def load_partition(
        self,
        partition_name: str = "val",
        cache_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        Loads records for partition ('val'/'VALIDATION', 'test'/'TEST', or 'train'/'TRAIN').
        Returns list of dicts:
        {
            'stem': str,
            'archive': str,
            'image': PIL.Image,
            'has_sign': bool,
            'boxes': List[BoundingBox]
        }
        """
        p_upper = partition_name.upper()
        if p_upper == "VAL":
            p_upper = "VALIDATION"
            
        records = [s for s in self.samples if s.get("partition", "").upper() == p_upper]
        
        # Group by archive
        by_archive: Dict[str, Dict[str, Dict]] = {}
        for r in records:
            arch = r.get("archiveName") or r.get("archive")
            stem = r.get("frameStem") or r.get("stem")
            if arch and stem:
                by_archive.setdefault(arch, {})[stem] = r
                
        results = []
        
        # If cache_dir is provided and populated, load directly from cache
        if cache_dir:
            c_dir = Path(cache_dir)
            meta_file = c_dir / f"{partition_name}_meta.json"
            if c_dir.exists() and meta_file.exists():
                print(f"Loading {partition_name} from local cache {c_dir}...")
                with open(meta_file, "r", encoding="utf-8") as f:
                    cached_records = json.load(f)
                for item in cached_records:
                    img_path = c_dir / item["filename"]
                    if img_path.exists():
                        img = Image.open(img_path).convert("RGB")
                        boxes = [BoundingBox(b["x1"], b["y1"], b["x2"], b["y2"]) for b in item["boxes"]]
                        results.append({
                            "stem": item["stem"],
                            "archive": item["archive"],
                            "image": img,
                            "has_sign": item["has_sign"],
                            "boxes": boxes,
                            "cached_path": str(img_path)
                        })
                if len(results) == len(records):
                    return results
                else:
                    results.clear()

        print(f"Extracting {len(records)} {partition_name} frames from tar shards...")
        cached_meta = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        for arch_name, stem_map in by_archive.items():
            shard_path = self.find_shard_path(arch_name)
            if not shard_path or not shard_path.exists():
                print(f"Warning: Shard {arch_name} not found.")
                continue
                
            with tarfile.open(shard_path, "r:*") as tar:
                for member in tar.getmembers():
                    if member.isdir():
                        continue
                    fname = Path(member.name).name
                    if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
                        continue
                    m_stem = Path(member.name).stem
                    if m_stem in stem_map:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        img_bytes = f.read()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        
                        rec = stem_map[m_stem]
                        raw_boxes = rec.get("signBoxes") or rec.get("boxes", [])
                        norm_boxes = []
                        for b in raw_boxes:
                            # Standardize to normalized [0, 1]
                            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                            # Check if coordinates are in pixel space (e.g. > 1.0)
                            if x2 > 1.0 or y2 > 1.0:
                                box = BoundingBox(x1 / 1920.0, y1 / 994.0, x2 / 1920.0, y2 / 994.0)
                            else:
                                box = BoundingBox(x1, y1, x2, y2)
                            norm_boxes.append(box.clamp())
                            
                        has_sign = rec.get("hasSign", len(norm_boxes) > 0)
                        
                        save_name = f"{m_stem}.jpg"
                        if cache_dir:
                            img.save(Path(cache_dir) / save_name, quality=95)
                            cached_meta.append({
                                "stem": m_stem,
                                "archive": arch_name,
                                "filename": save_name,
                                "has_sign": has_sign,
                                "boxes": [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in norm_boxes]
                            })
                            
                        results.append({
                            "stem": m_stem,
                            "archive": arch_name,
                            "image": img,
                            "has_sign": has_sign,
                            "boxes": norm_boxes,
                            "cached_path": str(Path(cache_dir) / save_name) if cache_dir else None
                        })
                        
        if cache_dir and cached_meta:
            with open(Path(cache_dir) / f"{partition_name}_meta.json", "w", encoding="utf-8") as f:
                json.dump(cached_meta, f, indent=2)
                
        print(f"Successfully loaded {len(results)} {partition_name} frames.")
        return results
