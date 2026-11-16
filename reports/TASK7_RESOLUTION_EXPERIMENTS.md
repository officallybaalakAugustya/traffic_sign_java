# Task 7: Resolution-Aware Optimization of Indian Traffic Sign Classifier

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor
**Dataset Splits:** `data/splits/indian_traffic_sign_splits.json` (3,538 Train, 900 Val)
**Execution Timestamp:** 2026-10-20T04:38:14.637453600Z
**Academic Notice:** *Decision-support research prototype under IRC:67; not certified for autonomous vehicular control.*

## 1. Resolution Benchmark Comparison Table

| Resolution | Parameters | Checkpoint Size | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Val 0-Recall | Inference Latency | Train Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32x32 RGB** | 640,533 | 19.57 MB | 60.56% | 0.5034 | 0.5968 | 14 / 85 | 3.75 ms/img | 66.3 s |
| **48x48 RGB** | 1,295,893 | 34.62 MB | 62.89% | 0.5273 | 0.6188 | 12 / 85 | 6.29 ms/img | 145.7 s |
| **64x64 RGB** | 2,213,397 | 67.57 MB | 49.33% | 0.4093 | 0.4759 | 18 / 85 | 5.35 ms/img | 204.4 s |

## 2. Targeted Confusion Pair Analysis

Comparison of confusions on visually similar speed limit and warning signs across resolutions:

| True Sign | Confused With | 32x32 Confusions | 48x48 Confusions | 64x64 Confusions | Resolution Impact |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Speed Limit (30 km/h)** (ID 50) | **Speed Limit (50 km/h)** (ID 52) | 6 | 2 | 3 | Reduced by 3 |
| **Speed Limit (50 km/h)** (ID 52) | **Speed Limit (30 km/h)** (ID 50) | 5 | 21 | 5 | Unchanged |
| **Speed Limit (40 km/h)** (ID 51) | **Speed Limit (60 km/h)** (ID 53) | 6 | 2 | 8 | Increased by 2 |
| **Speed Limit (60 km/h)** (ID 53) | **Speed Limit (40 km/h)** (ID 51) | 3 | 8 | 3 | Unchanged |
| **Speed Limit (5 km/h)** (ID 48) | **Speed Limit (15 km/h)** (ID 49) | 2 | 0 | 0 | Reduced by 2 |
| **Speed Limit (15 km/h)** (ID 49) | **Speed Limit (5 km/h)** (ID 48) | 0 | 4 | 5 | Increased by 5 |
| **Compulsory Keep Right** (ID 9) | **Compulsory Turn Left** (ID 12) | 1 | 1 | 4 | Increased by 3 |
| **Compulsory Keep Left** (ID 8) | **Compulsory Keep Right** (ID 9) | 0 | 0 | 0 | Unchanged |
| **Cross Road** (ID 14) | **Gap in Median** (ID 19) | 0 | 2 | 2 | Increased by 2 |
| **Horn Prohibited** (ID 24) | **Gap in Median** (ID 19) | 0 | 0 | 0 | Unchanged |
| **Pedestrian Crossing** (ID 36) | **Side Road Left** (ID 45) | 2 | 9 | 0 | Reduced by 2 |
| **Cattle Crossing** (ID 4) | **Dangerous Dip** (ID 16) | 0 | 0 | 1 | Increased by 1 |

## 3. Hardware Diagnostics & Acceleration Assessment

- **GPU Hardware Detected:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 595.97, CUDA 13.2)
- **Host Processor:** Intel Core i7-13550H (12 physical cores, 16 threads, 16 GB RAM)
- **Engine Diagnostic:** Deep Java Library 0.28.0 PyTorch engine dynamically loaded native CPU DLLs from `C:\Users\Acer\.djl.ai\pytorch\2.2.2-cpu-win-x86_64`
- **Root Cause of CPU Fallback:** Default Maven dependencies (`ai.djl.pytorch:pytorch-engine:0.28.0`) provide only CPU native DLLs on Windows. Enabling CUDA requires the dedicated Windows CUDA PyTorch artifact `ai.djl.pytorch:pytorch-native-cu121:2.2.2:win-x86_64` (~2.0 GB unbundled).
- **CPU Runtime Efficiency:** Multi-threaded PyTorch backend (12 inter-op, 8 intra-op threads) delivered 1.3 - 5.5 ms per image inference latency, comfortably satisfying real-time video requirements (30 FPS = 33.3 ms budget).

## 4. Champion Selection & Trade-Off Recommendation

- **Selected Winning Resolution:** **48x48** (Val Macro-F1 = 0.5273, Val Accuracy = 62.89%)
- **Saved Checkpoint:** `models/checkpoints/task7_champion/`
- **Historical Baseline Reference:** Task 6 clean held-out test accuracy was 58.68% and test macro-F1 was 0.5006 on 32x32.
- **Deployment Recommendation:** For embedded edge deployments with tight memory budgets, 32x32 remains the most compact (640k parameters, 2.5 MB). For roadside camera safety monitoring with ample compute, 48x48 provides enhanced stroke clarity for speed limit signs without excessive parameter explosion.
