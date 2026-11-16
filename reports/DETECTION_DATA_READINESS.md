# Detection Data Readiness Report

**Dataset:** Indian Road Driving Dataset sample (`data/raw/indian-road-driving-sample/`)
**Target Category:** Generic `traffic sign` (IRC:67 single-class detection)
**Sequence Grouping Strategy:** Extracted source video UUID prefix (`<UUID>_<frame>`). Frames from the same video remain strictly within one partition.

## Shard & Frame Statistics

| Metric | Count |
| :--- | :---: |
| **TAR Archives Scanned** | 10 |
| **Total Image Frames** | 10000 |
| **Matched JPG/JSON Pairs** | 10000 |
| **Distinct Video Sequences** | 135 |
| **Total Sign Bounding Boxes** | 1032 |
| **Excluded Degenerate/Invalid Boxes** | 0 |

## Partition Distribution (Sequence-Level Isolation)

| Metric | Training Partition | Validation Partition |
| :--- | :---: | :---: |
| **Unique Sequences** | 108 | 27 |
| **Frames** | 8136 | 1864 |
| **Frames with Signs** | 667 | 87 |
| **Empty Frames (Background)** | 7469 | 1777 |
| **Sign Bounding Boxes** | 910 | 122 |

## Sequence Leakage Verification

- **Verification Result:** PASSED (0 sequences cross training and validation boundaries).
- **Exclusion Policy:** Only labels matching exactly `traffic sign` were ingested; all vehicle, pedestrian, rider, and animal categories were excluded from detection targets.
