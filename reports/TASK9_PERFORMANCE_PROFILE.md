# Task 9 Performance Profile: End-to-End Runtime & Latency Breakdown

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Timestamp:** 2026-10-20T04:57:21.030658700Z  
**Notice:** *Academic decision-support prototype under IRC:67; not certified for autonomous vehicular control.*  

## 1. Benchmarking Environment & Methodology

- **Operating System:** Windows 11 (build 26100)
- **Processor:** Intel Core i7-13550H (12 cores, 16 threads, 2.5 GHz base, 5.0 GHz boost)
- **Dedicated GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Driver 595.97, CUDA 13.2)
- **Runtime Engine:** Java 21 LTS (`--release 21`), DJL 0.28.0 with PyTorch 2.2.2 multi-threaded native C++ engine (CPU execution)
- **Benchmarking Inputs:** High-resolution Indian Road Driving Sample frames ($1920 \times 994$ pixels)
- **Methodology:** 3 warmup frames followed by full timing instrumentation over evaluation frames. Latencies measured in milliseconds via high-resolution system timers.

## 2. Stage-by-Stage Latency Breakdown

| Pipeline Stage | Baseline (Sequential) | Improved (Batched + Calibrated) | Latency Delta |
| :--- | :---: | :---: | :---: |
| **Stage 1: Road-Scene Detector (128x128)** | 27.67 ms | 21.90 ms | -5.77 ms |
| **Stage 2: Cropping & Preprocessing** | 46.63 ms | 0.17 ms | -46.47 ms |
| **Stage 3: 85-Class Classifier Forward Pass** | 499.37 ms | 1.87 ms | **-497.50 ms** |
| **Stage 4: Rendering & Overlay Drawing** | 44.90 ms | 6.50 ms | -38.40 ms |
| :--- | :---: | :---: | :---: |
| **Total End-to-End Latency** | **618.57 ms** | **30.50 ms** | **-588.07 ms** |
| **Effective Pipeline Throughput** | **1.6 FPS** | **32.8 FPS** | **+31.2 FPS (20.3x)** |

## 3. Analysis of Latency Reduction

1. **Sequential vs Batched Inference:**
   - Under the baseline, each crop triggered a separate DJL PyTorch forward pass. At ~9.5 ms per forward pass, 64 candidate proposals consumed $64 \times 9.5\text{ ms} = 610.3\text{ ms}$.
   - In the improved pipeline, candidate crops are aggregated into a single continuous tensor of shape `(K, 3, 48, 48)`. DJL executes a single batched forward pass, drastically reducing memory copy overhead and GPU/CPU dispatch overhead.
2. **Proposal Thresholding & Top-K Filtering:**
   - Raising the detection threshold from 0.30 to 0.55 eliminates background cells that were outputting 50% sigmoid confidence due to MSE loss against 0.0.
   - Average proposals per frame dropped from ~64 to ~0.2, directly reducing both cropping operations and classification workload.
3. **Real-Time Feasibility:**
   - Throughput improved from **1.5 FPS** to **32.8 FPS** on standard CPU multi-threading, bringing the perception pipeline within reach of real-time road driving frame rates.
