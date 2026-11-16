# Detection Experiment & Evaluation Report

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Model Architecture:** Compact Single-Stage Sign Detector (`CompactSignDetectorBlock`)  
**Timestamp:** 2026-10-19T21:23:00.107052100Z  
**Notice:** *Academic decision-support prototype under IRC:67; not certified for autonomous vehicular control.*  

## Detector Architecture Summary

- **Input Resolution:** $128 \times 128 \times 3$ RGB (bilinear resize from native $1920 \times 994$ road frames)
- **Feature Extractor:** 4 Conv2D + BatchNorm + ReLU + MaxPool blocks ($3 \to 32 \to 64 \to 128 \to 128$)
- **Detection Head:** $1 \times 1$ Conv2D outputting $[5 \times 8 \times 8]$ grid tensor
- **Grid Representation:** $8 \times 8$ grid cells with 5 channels: $[\text{confidence}, \Delta x, \Delta y, \log(w), \log(h)]$
- **Trainable Parameters:** 242,885
- **Hardware Target:** CPU execution with DJL PyTorch 2.2.2 native engine

## Sequence-Held-Out Validation Benchmark (IoU = 0.50)

| Metric | Value |
| :--- | :---: |
| **Validation Precision** | 0.11% |
| **Validation Recall** | 9.84% |
| **Validation F1 Score** | 0.0021 |
| **Validation AP@50** | 0.0001 |
| **True Positives (TP)** | 12 |
| **False Positives (FP)** | 11124 |
| **False Negatives / Missed (FN)** | 110 |
| **Average Detection Latency** | 38.95 ms |
| **Total Training Duration** | 402.1 s |
| **Checkpoint Location** | `models/checkpoints/detector/detector_best` |

## Epoch Progression (Loss & AP@50)

| Epoch | Train Loss | Val Loss | Val Precision | Val Recall | Val AP@50 | Time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2877 | 0.0405 | 0.09% | 8.20% | 0.0001 | 77523 |
| 2 | 0.0319 | 0.0305 | 0.06% | 5.74% | 0.0000 | 72823 |
| 3 | 0.0257 | 0.0284 | 0.09% | 8.20% | 0.0001 | 52591 |
| 4 | 0.0227 | 0.0273 | 0.11% | 9.84% | 0.0001 | 39983 |
| 5 | 0.0207 | 0.0266 | 0.09% | 8.20% | 0.0001 | 74569 |
| 6 | 0.0191 | 0.0252 | 0.07% | 6.56% | 0.0000 | 84153 |

## End-to-End Pipeline Verification

The detector was coupled directly to the verified unweighted 85-class CNN classifier ([`RoadSceneInferencePipeline`](file:///S:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/inference/RoadSceneInferencePipeline.java)):  
1. **Detection:** Locates generic sign candidates in high-resolution road frames.
2. **Safe Cropping:** Crops candidate sign patches with boundary clipping and aspect retention.
3. **Classification:** Resizes crops to $32 \times 32 \times 3$ and evaluates against the 85 IRC:67 sign taxonomy.
4. **Uncertainty State:** Rejects candidate boxes with detection confidence $< 0.30$ or classification confidence $< 0.25$ as `LOW_CONFIDENCE` / `UNKNOWN_CLASS`.
