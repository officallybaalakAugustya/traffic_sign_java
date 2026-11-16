# Classification Dataset Validation Report

**Dataset Root:** `data\raw\traffic_sign_classification_dataset`

## Summary Statistics

| Metric | Train Partition | Test Partition |
| :--- | :---: | :---: |
| **Total Images** | 4438 | 1288 |
| **Distinct Classes** | 85 | 85 |
| **Minimum Support** | 2 | 1 |
| **Maximum Support** | 242 | 81 |
| **Imbalance Ratio** | 121.00 | 81.00 |

## Class Consistency

- **Class Consistency Across Splits:** PASSED (Exact class match)

## Data Hygiene & Integrity

- **Unreadable / Corrupt Images:** 0
- **Unexpected File Extensions:** 0
- **Duplicate Hash Sets in Train:** 755
- **Duplicate Hash Sets in Test:** 80
- **Cross-Split Hash Overlap:** 846

## Deterministic Train/Validation Partitioning

- **Random Seed:** `42`
- **Derived Train Count:** 3551
- **Derived Validation Count:** 887
- **Preserved Held-Out Test Count:** 1288 (Strictly untouched)

## Recorded Anomalies & Warnings

- Duplicate content hashes detected within train: 755 duplicate sets
- Duplicate content hashes detected within test: 80 duplicate sets
- Cross-split hash leakage detected between generated train and val: 272 images
- Content hash leakage detected between raw train and raw test sets: 574 images

