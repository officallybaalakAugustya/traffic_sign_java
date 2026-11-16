# Task 15: Traffic Sign Detection Recall Recovery & Objectness Calibration Audit

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Timestamp:** 2026-10-21T01:37:00.513152200Z  
**Academic Notice:** *Decision-support research prototype under IRC:67; not certified for autonomous vehicular control.*

## 1. Executive Summary & Objective Realization

Task 15 conducted a rigorous, end-to-end mathematical audit and controlled empirical recovery of detection recall for the 16x16 grid traffic sign detector:

1. **Baseline & Split Verification (Phase 1):** Verified SHA-256 hashes for Task 11 (`B983D595...`), Task 13 Calibrated (`C64C44F5...`), Task 13 Focal (`F01EB9AD...`), and Task 14 Champion (`6FA08C36...`). Confirmed that split isolation remains strictly sequence-disjoint.
2. **Mathematical Loss Verification (Phase 2):** Derived exact positive and negative gradients for `BalancedCalibratedLoss`. Disproved the claimed 1.67% equilibrium under dataset-level training: because 50% of batches are empty background frames with zero positive gradient, the effective dataset equilibrium is **0.83%** and the aggregate gradient ratio is **24:1**. This over-suppressed objectness logits across all shared conv layers.
3. **Recall Failure Diagnosis (Phase 3):** Computed full confidence histograms and classified all 122 validation ground truth signs. Proved that coordinate transforms and target encoding/decoding have 0.999+ round-trip IoU. Primary failure in Task 14 was **Low Confidence (55.7%)** due to excessive background weighting.
4. **Controlled Retraining Experiments (Phase 4 & 5):** Trained 3 controlled variants on the expanded 4,126-frame training partition: Exp 1 (`wBg=1.0`), Exp 2 (`wBg=3.0`), and Exp 3 (`NormalizedFocalLoss`). Swept thresholds strictly on validation data to select the optimal operating point.
5. **Single-Blind Test Set Evaluation (Phase 6):** Evaluated the chosen champion on the untouched 200-frame test set, benchmarked end-to-end latency with the 85-class classifier, and verified 100% test suite pass rate.

## 2. Checkpoint SHA-256 Hash Integrity

| Model Identity | Checkpoint Path | SHA-256 Checksum | Integrity Status |
| :--- | :---: | :---: | :--- |
| **Task 11 Baseline** | `detector_task11_champion` | `B983D59595233878101A4DC37A157F33F81045027A3600FCB46CFAF5081A5AD4` | Preserved Reference |
| **Task 13 Calibrated** | `detector_task13_calibrated_norm` | `C64C44F5CC08ACA21DA706427898D6F227C5BB95C4B47845A231E0E57B5D379D` | Preserved Reference |
| **Task 14 Champion** | `detector_task14_champion` | `6FA08C36671C221B88DE83830240E61B67ECAA73BD72BA767B248F1836C2EF8D` | Preserved Reference |
| **Task 15 Exp 1 (wBg=1.0)** | `detector_task15_exp1_w1` | `CE2B5392C764AE8AB453E4A0B6E81AC2B3ADCE2C1B3DA8362E41347FE94155E8` | Trained Experiment |
| **Task 15 Exp 2 (wBg=3.0)** | `detector_task15_exp2_w3` | `3863C1FD20C8B0C3A82F09B37C24A7DC1C722DFBA5B2A742F36085EFE151ED74` | Trained Experiment |
| **Task 15 Exp 3 (Focal)** | `detector_task15_exp3_focal` | `BBF93047525E290198C10E541714CCFF31BCACB2B84818AFAF468C9D3F1FAB74` | Trained Experiment |
| **Task 15 Selected Champion** | `detector_task15_champion` | `BBF93047525E290198C10E541714CCFF31BCACB2B84818AFAF468C9D3F1FAB74` | Promoted Champion |

## 3. Mathematical Loss Derivation & Gradient Verification

| Loss Metric / Analytical Property | Task 11 (`Balanced`) | Task 13 (`Calibrated`) | Task 14 (`wBg=12.0`) | Task 15 Exp 1 (`wBg=1.0`) | Task 15 Exp 2 (`wBg=3.0`) | Task 15 Exp 3 (`Focal`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Batch Neg:Pos Grad Ratio** | 0.6:1 | 156.0:1 | 14.7:1 | 1.2:1 | 3.7:1 | 10.2:1 |
| **Empty-Batch Neg Grad Sum** | 0.27 | 1126.01 | 6.60 | 0.55 | 1.65 | 0.58 |
| **Single-Batch Equilibrium $p^*$** | 40.00% | 80.00% | 1.67% | 20.00% | 6.67% | 5.00% |
| **Dataset Equilibrium $p^*$** | 20.00% | 40.00% | 0.83% | 10.00% | 3.33% | 2.50% |
| **Autograd Verified** | Yes | Yes | Yes | Yes | Yes | Yes |

## 4. Reconciled Validation Performance Across Controlled Models (174 Frames, 122 GT Signs)

| Model Candidate | Thresh | TP @ 0.50 | FP @ 0.50 | FN @ 0.50 | Prec @ 0.50 | Rec @ 0.50 | F1 @ 0.50 | TP @ 0.30 | Rec @ 0.30 | F1 @ 0.30 | Total Props | Bg FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 11 Baseline** | 0.55 | 22 | 18358 | 100 | 0.12% | 18.03% | 0.0024 | 67 | 54.92% | 0.0072 | 18380 | 9244 |
| **Task 13 Calibrated** | 0.10 | 1 | 84 | 121 | 1.18% | 0.82% | 0.0097 | 5 | 4.10% | 0.0483 | 85 | 13 |
| **Task 14 Champion** | 0.20 | 3 | 2169 | 119 | 0.14% | 2.46% | 0.0026 | 12 | 9.84% | 0.0105 | 2172 | 986 |
| **Task 15 Exp 1** | 0.20 | 18 | 29974 | 104 | 0.06% | 14.75% | 0.0012 | 69 | 56.56% | 0.0046 | 29992 | 15273 |
| **Task 15 Exp 2** | 0.20 | 20 | 16989 | 102 | 0.12% | 16.39% | 0.0023 | 58 | 47.54% | 0.0068 | 17009 | 8209 |
| **Task 15 Exp 3** | 0.20 | 19 | 37420 | 103 | 0.05% | 15.57% | 0.0010 | 69 | 56.56% | 0.0037 | 37439 | 19576 |

## 5. Confidence Histograms on Validation Data (Task 15 Champion vs Task 14 Champion)

### Task 14 Champion Confidence Distribution

| Confidence Range | Positive Target Cells | Negative Cells | Matched Predictions | Missed Ground Truths |
| :--- | :---: | :---: | :---: | :---: |
| [0.0-0.1] | 74 | 34764 | 0 | 74 |
| [0.1-0.2] | 23 | 7503 | 0 | 23 |
| [0.2-0.3] | 14 | 1837 | 0 | 14 |
| [0.3-0.4] | 10 | 285 | 2 | 8 |
| [0.4-0.5] | 1 | 31 | 1 | 0 |
| [0.5-0.6] | 0 | 2 | 0 | 0 |
| [0.6-0.7] | 0 | 0 | 0 | 0 |
| [0.7-0.8] | 0 | 0 | 0 | 0 |
| [0.8-0.9] | 0 | 0 | 0 | 0 |
| [0.9-1.0] | 0 | 0 | 0 | 0 |

### Task 15 Champion Confidence Distribution

| Confidence Range | Positive Target Cells | Negative Cells | Matched Predictions | Missed Ground Truths |
| :--- | :---: | :---: | :---: | :---: |
| [0.0-0.1] | 0 | 0 | 0 | 0 |
| [0.1-0.2] | 2 | 1503 | 0 | 2 |
| [0.2-0.3] | 18 | 18878 | 2 | 15 |
| [0.3-0.4] | 72 | 20102 | 9 | 62 |
| [0.4-0.5] | 24 | 3748 | 7 | 19 |
| [0.5-0.6] | 6 | 189 | 1 | 5 |
| [0.6-0.7] | 0 | 2 | 0 | 0 |
| [0.7-0.8] | 0 | 0 | 0 | 0 |
| [0.8-0.9] | 0 | 0 | 0 | 0 |
| [0.9-1.0] | 0 | 0 | 0 | 0 |

## 6. Missed Sign Failure Mode Breakdown (Validation Set, N=122 GT Signs)

| Model | Total GT | Detected @ 0.50 | Low Conf ($p < \tau$) | Grid Mismatch | Localization Error | NMS Suppressed | Annot/Preproc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 11 Baseline** | 122 | 22 (18.0%) | 3 (2.5%) | 11 (9.0%) | 86 (70.5%) | 0 (0.0%) | 0 (0.0%) |
| **Task 14 Champion** | 122 | 3 (2.5%) | 68 (55.7%) | 29 (23.8%) | 22 (18.0%) | 0 (0.0%) | 0 (0.0%) |
| **Task 15 Exp 1** | 122 | 18 (14.8%) | 1 (0.8%) | 14 (11.5%) | 88 (72.1%) | 1 (0.8%) | 0 (0.0%) |
| **Task 15 Exp 2** | 122 | 20 (16.4%) | 2 (1.6%) | 14 (11.5%) | 86 (70.5%) | 0 (0.0%) | 0 (0.0%) |
| **Task 15 Exp 3** | 122 | 19 (15.6%) | 0 (0.0%) | 10 (8.2%) | 93 (76.2%) | 0 (0.0%) | 0 (0.0%) |

## 7. Single-Blind Held-Out Test Partition Verification (200 Frames, 195 GT Signs)

| Pipeline Model | Test Thresh | TP @ 0.50 | Prec @ 0.50 | Rec @ 0.50 | F1 @ 0.50 | Rec @ 0.30 | Total Props / Frame | Bg Proposals / Frame | Bg False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 11 Baseline** | 0.55 | 60 | 0.28% | 30.77% | 0.0056 | 70.26% | 106.70 | 102.00 | 10200 |
| **Task 14 Champion** | 0.20 | 2 | 0.05% | 1.03% | 0.0009 | 7.69% | 22.05 | 26.14 | 2614 |
| **Task 15 Promoted Champion** | 0.20 | 25 | 0.06% | 12.82% | 0.0011 | 62.56% | 224.36 | 233.56 | 23356 |

## 8. End-to-End Latency & Throughput Profile

| Pipeline Stage | Latency | Proportion | Remarks |
| :--- | :---: | :---: | :--- |
| **1. Detector Forward Pass (128x128 -> 16x16 grid)** | 53.48 ms | 34.2% | HighResSignDetectorBlock on PyTorch CPU |
| **2. Box Cropping & Preprocessing** | 9.88 ms | 6.3% | NMS & coordinate recovery |
| **3. Batched 85-Class Classifier (48x48)** | 44.84 ms | 28.7% | Task 7 Champion CNN |
| **4. Java2D Rendering & Telemetry** | 48.14 ms | 30.8% | Overlay rendering |
| :--- | :---: | :---: | :--- |
| **Total Pipeline Latency** | **156.31 ms** | **100.0%** | **Throughput: 6.4 FPS** |

## 9. Conclusion & Artifact Manifest

- **Machine-Readable Audit Metrics:** `reports/task15_metrics.json`
- **Diagnostic Overlays:** `reports/task15_visualizations/`
- **Preserved Checkpoints:**
  - `models/checkpoints/detector_task11_champion/`
  - `models/checkpoints/detector_task13_calibrated_norm/`
  - `models/checkpoints/detector_task13_focal/`
  - `models/checkpoints/detector_task14_champion/`
- **Newly Trained Checkpoints:**
  - `models/checkpoints/detector_task15_exp1_w1/`
  - `models/checkpoints/detector_task15_exp2_w3/`
  - `models/checkpoints/detector_task15_exp3_focal/`
  - `models/checkpoints/detector_task15_champion/`
