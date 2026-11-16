# Experiment Tracking Report

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Run Timestamp:** 2026-10-19T21:02:57.514661500Z  
**Academic Notice:** Decision-support and educational prototype only; not certified for autonomous vehicular control.

## Dataset Split & Partition Summary

| Partition | Sample Count | Notes |
| :--- | :---: | :--- |
| **Training** | 3538 | Grouped by SHA-256 hash (80%) |
| **Validation** | 900 | Grouped by SHA-256 hash (20%) |
| **Clean Held-Out Test** | 714 | Hashes strictly absent from training |
| **Excluded Test Overlap** | 574 | Quarantined bit-for-bit duplicate images |

## Model Comparison Matrix

| Model | Params | Val Acc (%) | Val Macro-F1 | Test Acc (%) | Test Macro-F1 | Latency (ms) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 0 | 4.67% | 0.0010 | 6.72% | 0.0015 | 0.00 ms | 0.0 s |
| **Pure-Java MLP (3072-256-128-85)** | 830,549 | 52.78% | 0.3124 | 45.10% | 0.2571 | 13.75 ms | 109.4 s |
| **Compact CNN (Unweighted)** | 640,533 | 66.78% | 0.3840 | 52.66% | 0.2946 | 5.56 ms | 55.1 s |
| **Compact CNN (Class-Weighted)** | 640,533 | 42.67% | 0.2920 | 35.71% | 0.2540 | 5.56 ms | 39.2 s |

## Epoch Progression (Validation Loss & Macro-F1)

### Pure-Java MLP Baseline
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro-F1 | Time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3.9902 | 23.32% | 3.5570 | 21.67% | 0.0371 | 12677 |
| 2 | 3.1138 | 34.85% | 2.7752 | 33.56% | 0.1159 | 13703 |
| 3 | 2.4890 | 39.12% | 2.5082 | 36.00% | 0.1551 | 13808 |
| 4 | 2.1427 | 52.12% | 2.2487 | 47.89% | 0.2137 | 13385 |
| 5 | 1.8516 | 50.93% | 2.0967 | 44.11% | 0.1932 | 13297 |
| 6 | 1.6776 | 56.98% | 1.9522 | 50.11% | 0.2408 | 13171 |
| 7 | 1.5307 | 63.45% | 1.7437 | 57.56% | 0.3084 | 14041 |
| 8 | 1.3859 | 61.79% | 1.8737 | 52.78% | 0.3124 | 13545 |

### DJL Compact CNN (Unweighted)
| Epoch | Train Loss | Val Loss | Val Acc (%) | Val Macro-F1 | Time (ms) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 3.7681 | 2.8268 | 33.56% | 0.0846 | 7763 |
| 2 | 2.7541 | 2.2966 | 41.11% | 0.1381 | 6841 |
| 3 | 2.2709 | 1.8978 | 51.89% | 0.2227 | 6980 |
| 4 | 1.9294 | 1.6883 | 55.56% | 0.2475 | 6598 |
| 5 | 1.7069 | 1.5186 | 59.78% | 0.2929 | 6853 |
| 6 | 1.5520 | 1.3794 | 62.67% | 0.3256 | 7103 |
| 7 | 1.4037 | 1.3138 | 63.89% | 0.3525 | 6191 |
| 8 | 1.2750 | 1.2105 | 66.78% | 0.3840 | 6619 |

### DJL Compact CNN (Class-Weighted)
| Epoch | Train Loss | Val Loss | Val Acc (%) | Val Macro-F1 | Time (ms) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 4.8107 | 4.5180 | 4.67% | 0.0093 | 5217 |
| 2 | 4.1510 | 4.2603 | 9.67% | 0.0305 | 4970 |
| 3 | 3.8725 | 3.9684 | 21.56% | 0.0827 | 4884 |
| 4 | 3.5985 | 3.7311 | 28.67% | 0.1507 | 4812 |
| 5 | 3.3055 | 3.5669 | 31.89% | 0.1732 | 4730 |
| 6 | 3.0847 | 3.0757 | 35.33% | 0.2233 | 4854 |
| 7 | 2.9250 | 3.0254 | 36.11% | 0.2371 | 4843 |
| 8 | 2.6991 | 2.7536 | 42.67% | 0.2920 | 4812 |
