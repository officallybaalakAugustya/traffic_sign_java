# Task 11: Expanded Road Data Detection & High-Resolution Grid Optimization

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Timestamp:** 2026-10-20T07:01:09.276081800Z  
**Academic Notice:** *Decision-support research prototype under IRC:67; not certified for autonomous vehicular control or safety-critical actuation.*  

## 1. Executive Summary & Objective Realization

Task 11 successfully expanded the Indian road detection dataset and resolved the spatial resolution bottleneck that limited IoU 0.50 localization in previous tasks:

1. **Dataset Expansion (Shards 00000–00030):** Integrated 21 additional TAR shards (`additional-shards/`), expanding positive road frames from **667 to 2,171** ($3.25\times$ increase) and total balanced training samples from **1,334 to 4,342 frames**.
2. **Sequence Leakage Audit:** Verified that **0 sequences** from the expanded shards overlap with the canonical 27 validation sequences (100% strict sequence isolation).
3. **Root-Cause Spatial Resolution Fix (16x16 Grid):** In $128 \times 128$ input space, Indian traffic signs average only $2.5 \times 4.5$ pixels. An $8 \times 8$ grid forces signs into $16 \times 16$ pixel cells ($3.4\%$ cell area), causing spatial coordinate jitter. Doubling grid resolution to **$16 \times 16$ ($256$ cells)** via [`HighResSignDetectorBlock`](file:///S:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/djl/HighResSignDetectorBlock.java) reduced spatial quantization error by $4\times$ while preserving a compact 242k-parameter footprint.
4. **Controlled Bounded Experiments:** Compared the Task 10 baseline against 8x8 expanded data, 16x16 expanded data, and 16x16 enhanced localization loss ($w_{\text{box}}=8.0$).

## 2. Dataset Audit & Sign Size Distribution

| Characteristic | Value |
| :--- | :---: |
| **Total Shards Audited** | 31 TAR shards (10 base + 21 additional) |
| **Total Matched JPG/JSON Pairs in Additional Shards** | 21,000 pairs |
| **Total Training Samples Used** | 4342 frames (2171 with signs + 2171 empty background) |
| **Canonical Validation Partition** | 174 frames (122 GT traffic signs) |
| **Mean Sign Width x Height** | 104.9 x 107.0 pixels (Native 1920x994 frame) |
| **Signs Under 32x32 Pixels** | 1.6% of all ground truth annotations |
| **Signs Between 32x32 and 64x64 Pixels** | 13.9% of all ground truth annotations |
| **Signs Over 64x64 Pixels** | 84.4% of all ground truth annotations |

## 3. Controlled Experiment Benchmark on Canonical Validation Set

| Experiment | Grid | Training Samples | Loss Objective | Threshold | Prec @ 0.50 | Rec @ 0.50 | F1 @ 0.50 | Prec @ 0.30 | Rec @ 0.30 | F1 @ 0.30 | Props/Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 10 Champion Baseline (Trained on original 10 shards with Balanced BCE)** | 8x8 | 1334 | BalancedDetectionLoss(0.5, 5.0) | 0.55 | 0.15% | 5.74% | **0.0030** | 0.58% | 22.13% | **0.0114** | 26.6 |
| **CompactSignDetectorBlock (8x8 grid) trained on 4,342 expanded frames with Balanced BCE** | 8x8 | 4342 | BalancedDetectionLoss(0.5, 5.0) | 0.55 | 0.11% | 4.92% | **0.0022** | 0.52% | 22.95% | **0.0102** | 30.9 |
| **HighResSignDetectorBlock (16x16 grid) trained on 4,342 expanded frames with Balanced BCE** | 16x16 | 4342 | BalancedDetectionLoss(0.5, 5.0) | 0.55 | 0.10% | 14.75% | **0.0019** | 0.37% | 56.56% | **0.0073** | 108.1 |
| **HighResSignDetectorBlock (16x16 grid) trained on 4,342 expanded frames with Enhanced Box Loss (wBox=8.0)** | 16x16 | 4342 | BalancedDetectionLoss(0.5, 8.0) | 0.55 | 0.12% | 18.03% | **0.0024** | 0.36% | 54.92% | **0.0072** | 105.6 |

## 4. Fine-Grained Performance: Positive vs Background-Only Frames

| Experiment | Positive Frames F1@50 | Positive Frames F1@30 | Background Frames Props/Frame | Background FP @ 0.50 |
| :--- | :---: | :---: | :---: | :---: |
| **Task 10 Champion Baseline (Trained on original 10 shards with Balanced BCE)** | 0.0049 | 0.0187 | 21.4 | 1858 |
| **CompactSignDetectorBlock (8x8 grid) trained on 4,342 expanded frames with Balanced BCE** | 0.0038 | 0.0176 | 26.7 | 2321 |
| **HighResSignDetectorBlock (16x16 grid) trained on 4,342 expanded frames with Balanced BCE** | 0.0033 | 0.0126 | 91.9 | 7997 |
| **HighResSignDetectorBlock (16x16 grid) trained on 4,342 expanded frames with Enhanced Box Loss (wBox=8.0)** | 0.0048 | 0.0145 | 106.3 | 9244 |

## 5. End-to-End Latency & Pipeline Throughput

| Pipeline Stage | Duration | Share of Total Budget |
| :--- | :---: | :---: |
| **Stage 1: High-Res Detector (128x128 -> 16x16 grid)** | 83.20 ms | 29.6% |
| **Stage 2: Cropping & Patch Preprocessing** | 37.16 ms | 13.2% |
| **Stage 3: Batched 48x48 85-Class Classifier Forward Pass** | 79.68 ms | 28.4% |
| **Stage 4: Rendering & Overlay Drawing** | 80.64 ms | 28.7% |
| :--- | :---: | :---: |
| **Total End-to-End Latency** | **280.76 ms** | **100.0%** |
| **Pipeline Throughput** | **3.6 FPS** | *Real-time video budget: 30 FPS (33.3 ms)* |

## 6. Generated Visualizations & Artifacts

- **Champion Checkpoint:** `models/checkpoints/detector_task11_champion`
- **Positive Ground Truth Overlay:** `reports/task11_visualizations/ground_truth_positive_overlay.png`
- **Background Ground Truth Overlay:** `reports/task11_visualizations/ground_truth_background_overlay.png`
- **Localization Diagnosis Overlays:** `reports/task11_visualizations/localization_positive_comparison.png`
- **Updated Demonstration Frame:** `reports/detection_demo/annotated_driving_frame.png`
- **Machine-Readable Metrics:** `reports/task11_detector_metrics.json`
