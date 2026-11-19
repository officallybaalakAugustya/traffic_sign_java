"""
Dataset splitting, sequence grouping, deduplication, and validation
mirroring com.trafficsign.data.DatasetSplitter, GroupedDatasetSplitter,
DuplicateSafeSplitManifest, and DatasetValidator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from trafficsign.data.taxonomy import IndianTrafficSignTaxonomy


class PartitionCategory(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    CLEAN_TEST = "CLEAN_TEST"
    EXCLUDED_TRAIN_OVERLAP = "EXCLUDED_TRAIN_OVERLAP"


@dataclass
class PartitionEntry:
    sample_id: str
    relative_path: str
    class_id: int
    class_label: str
    content_hash: str
    partition: PartitionCategory


@dataclass
class ManifestSummary:
    total_train_files_scanned: int
    total_test_files_scanned: int
    unique_train_hashes: int
    unique_test_hashes: int
    overlapping_unique_hashes: int
    train_files_assigned: int
    val_files_assigned: int
    clean_test_files_assigned: int
    excluded_test_files_assigned: int
    random_seed: int
    val_ratio: float
    entries: List[PartitionEntry] = field(default_factory=list)


def compute_file_sha256(path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetSplitter:
    """
    Standard stratified train / validation dataset splitter.
    """

    @staticmethod
    def split(
        items: Sequence[Tuple[str, int]],
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        by_class: Dict[int, List[Tuple[str, int]]] = {}
        for path, label in items:
            by_class.setdefault(label, []).append((path, label))

        rng = random.Random(seed)
        train_list: List[Tuple[str, int]] = []
        val_list: List[Tuple[str, int]] = []

        for label, class_items in by_class.items():
            shuffled = list(class_items)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
            val_list.extend(shuffled[:n_val])
            train_list.extend(shuffled[n_val:])

        return train_list, val_list


class GroupedDatasetSplitter:
    """
    Sequence/group-aware dataset splitter preventing frame leakage across video clips.
    """

    @staticmethod
    def split_by_group(
        items: Sequence[Tuple[str, str, int]],  # (path, group_id, class_id)
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
        # Group by sequence ID
        by_group: Dict[str, List[Tuple[str, str, int]]] = {}
        for item in items:
            path, group_id, label = item
            by_group.setdefault(group_id, []).append(item)

        groups = list(by_group.keys())
        rng = random.Random(seed)
        rng.shuffle(groups)

        n_val_groups = max(1, int(round(len(groups) * val_ratio)))
        val_groups = set(groups[:n_val_groups])

        train_list: List[Tuple[str, str, int]] = []
        val_list: List[Tuple[str, str, int]] = []

        for gid, group_items in by_group.items():
            if gid in val_groups:
                val_list.extend(group_items)
            else:
                train_list.extend(group_items)

        return train_list, val_list


class DuplicateSafeSplitManifest:
    """
    Creates duplicate-safe, SHA256-verified partition manifests ensuring zero leakage.
    """

    @staticmethod
    def generate(
        dataset_root: Union[str, Path],
        val_ratio: float = 0.2,
        random_seed: int = 42,
        taxonomy: Optional[IndianTrafficSignTaxonomy] = None
    ) -> ManifestSummary:
        root = Path(dataset_root)
        train_dir = root / "train"
        test_dir = root / "test"

        train_files: List[Path] = []
        if train_dir.exists():
            for ext in ("*.png", "*.jpg", "*.ppm"):
                train_files.extend(train_dir.rglob(ext))

        test_files: List[Path] = []
        if test_dir.exists():
            for ext in ("*.png", "*.jpg", "*.ppm"):
                test_files.extend(test_dir.rglob(ext))

        train_hashes: Set[str] = set()
        test_hashes: Set[str] = set()

        for f in train_files:
            train_hashes.add(compute_file_sha256(f))

        overlap_count = 0
        for f in test_files:
            h = compute_file_sha256(f)
            test_hashes.add(h)
            if h in train_hashes:
                overlap_count += 1

        entries: List[PartitionEntry] = []
        rng = random.Random(random_seed)

        # Split train files
        shuffled_train = list(train_files)
        rng.shuffle(shuffled_train)
        n_val = int(round(len(shuffled_train) * val_ratio))
        val_set = set(shuffled_train[:n_val])

        n_train_assigned = 0
        n_val_assigned = 0

        for f in train_files:
            rel = str(f.relative_to(root))
            h = compute_file_sha256(f)
            # determine class
            cls_name = f.parent.name
            part = PartitionCategory.VALIDATION if f in val_set else PartitionCategory.TRAIN
            if part == PartitionCategory.VALIDATION:
                n_val_assigned += 1
            else:
                n_train_assigned += 1

            entries.append(PartitionEntry(
                sample_id=f.stem,
                relativePath=rel,
                class_id=0,
                class_label=cls_name,
                content_hash=h,
                partition=part
            ))

        n_clean_test = 0
        n_excluded_test = 0
        for f in test_files:
            rel = str(f.relative_to(root))
            h = compute_file_sha256(f)
            cls_name = f.parent.name
            if h in train_hashes:
                part = PartitionCategory.EXCLUDED_TRAIN_OVERLAP
                n_excluded_test += 1
            else:
                part = PartitionCategory.CLEAN_TEST
                n_clean_test += 1

            entries.append(PartitionEntry(
                sample_id=f.stem,
                relativePath=rel,
                class_id=0,
                class_label=cls_name,
                content_hash=h,
                partition=part
            ))

        return ManifestSummary(
            total_train_files_scanned=len(train_files),
            total_test_files_scanned=len(test_files),
            unique_train_hashes=len(train_hashes),
            unique_test_hashes=len(test_hashes),
            overlapping_unique_hashes=overlap_count,
            train_files_assigned=n_train_assigned,
            val_files_assigned=n_val_assigned,
            clean_test_files_assigned=n_clean_test,
            excluded_test_files_assigned=n_excluded_test,
            random_seed=random_seed,
            val_ratio=val_ratio,
            entries=entries
        )
