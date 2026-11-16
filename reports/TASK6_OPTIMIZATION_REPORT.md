# Task 6 Optimization Report: Error-Driven CNN Refinement

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor
**Target Classes:** 85 IRC:67 Traffic Sign Classes
**Timestamp:** 2026-10-19T22:03:33.950670500Z
**Notice:** *Academic decision-support prototype under IRC:67; not certified for autonomous vehicular control.*

## 1. Controlled Experiment Suite Comparison

| Experiment Strategy | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val 0-Recall | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task6_C_Repro** (Uniform Balanced Sampling (repro lr=1e-3)) | 61.89% | 0.4977 | 0.6084 | 13 / 85 | 37.4 s |
| **Task6_C1_Sqrt** (Square-Root Smoothed Sampling (power=0.5)) | 65.56% | 0.4823 | 0.6472 | 17 / 85 | 37.3 s |
| **Task6_C2_Photo** (Balanced Sampling + Photometric Jitter (+-10%)) | 62.44% | 0.5307 | 0.6146 | 12 / 85 | 40.3 s |
| **Task6_C3_Lr5e4** (Balanced Sampling with tuned lr=5e-4) | 58.89% | 0.4800 | 0.5736 | 18 / 85 | 32.6 s |

## 2. Winning Model Selection (Validation-Driven)

- **Selected Champion:** **Task6_C2_Photo**
- **Validation Macro-F1:** 0.5307
- **Validation Accuracy:** 62.44%
- **Checkpoint Location:** `models/checkpoints/task6_champion/`

## 3. Held-Out Clean Test Set Evaluation (Evaluated Once After Selection)

- **Clean Held-Out Test Accuracy:** 58.68%
- **Clean Held-Out Test Macro-F1:** 0.5006
- **Clean Held-Out Test Weighted-F1:** 0.5787
- **Clean Held-Out Test Macro-Recall:** 0.5250
- **Zero-Recall Classes on Held-Out Test:** 14 / 85

## 4. Hardware and Runtime Diagnostics

- **Host Processor:** Intel Core i7-13550H (12 physical cores, 16 threads)
- **Dedicated GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 595.97, CUDA 13.2)
- **DJL Engine:** PyTorch 2.2.2 Engine (Native CPU DLLs cached at `C:\Users\Acer\.djl.ai\pytorch\2.2.2-cpu-win-x86_64`)
- **CPU Fallback:** Multi-threaded PyTorch backend (12 inter-op threads, 8 intra-op threads). Measured inference latency: 5.56 ms/image.

## 5. Strategic Recommendations for Next Development Stage

1. **Sign-Crop Resolution Upgrade:** Increase sign classifier crop resolution from 32x32 to 48x48 or 64x64 to resolve numerical ambiguities in speed limit and restriction signs.
2. **Detection-to-Classification Cascading:** Integrate this champion classifier directly with the road-scene detection pipeline to filter out false-positive bounding boxes before safety risk alerting.
3. **Class-Aware Active Learning:** Collect additional real-world driving footage for the 12 persistently scarce classes (e.g. Bullock Carts Prohibited, Ferry, Barrier Ahead) to address data scarcity at the source.
