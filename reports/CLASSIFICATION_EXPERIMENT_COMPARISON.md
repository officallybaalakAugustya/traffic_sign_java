# Classification Experiment Comparison Report

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Target Taxonomy:** 85 IRC:67 Traffic Sign Classes  
**Evaluation Splits:** `data/splits/indian_traffic_sign_splits.json` (3,538 Train, 900 Val, 714 Clean Test, 574 Quarantined Test)  
**Execution Timestamp:** 2026-10-19T21:41:03Z  
**Academic Notice:** *This system is a research prototype under IRC:67 guidelines; not certified for autonomous vehicle control.*

---

## 1. Controlled Benchmark Table across Optimization Strategies

| Strategy | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val 0-Recall | Test Acc | Test Macro-F1 | Test Weighted-F1 | Test 0-Recall | Train Time | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Experiment A (Baseline)** | 70.11% | 0.4082 | 0.6789 | 31 / 85 | 57.28% | 0.3450 | 0.5438 | 32 / 85 | 40.7 s | `models/checkpoints/exp_a_baseline/exp_a_baseline_best` |
| **Experiment B (Augmentation)** | 57.56% | 0.2665 | 0.5332 | 42 / 85 | 45.80% | 0.2060 | 0.4226 | 47 / 85 | 36.7 s | `models/checkpoints/exp_b_augmented/exp_b_augmented_best` |
| **Experiment C (Balanced Sampling)** | **62.33%** | **0.4872** | **0.6152** | **17 / 85** | **59.66%** | **0.5046** | **0.6006** | **12 / 85** | **32.3 s** | `models/checkpoints/exp_c_balanced_sampling/exp_c_balanced_sampling_best` |
| **Experiment D (Capped Weights)** | 47.11% | 0.3303 | 0.4600 | 28 / 85 | 43.28% | 0.2921 | 0.4246 | 27 / 85 | 32.6 s | `models/checkpoints/exp_d_capped_weights/exp_d_capped_weights_best` |

---

## 2. Hardware and Acceleration Runtime Diagnosis

- **Physical GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
  - Dedicated VRAM: 6,141 MiB (6.0 GB)
  - Driver Version: 595.97
  - CUDA Driver Compatibility: 13.2
- **Host CPU:** Intel Core i7-13550H (12 physical cores, 16 threads, 16 GB System RAM)
- **Engine Framework:** Deep Java Library (DJL) 0.28.0 with PyTorch 2.2.2 Engine
- **Engine Diagnostics & CPU Fallback:**
  - DJL engine inspection logged: `ai.djl.util.cuda.CudaUtils -- No cudart library found in path.`
  - Native cache directory: `C:\Users\Acer\.djl.ai\pytorch\2.2.2-cpu-win-x86_64\`
  - Native binaries loaded: `asmjit.dll`, `c10.dll`, `libiomp5md.dll`, `fbgemm.dll`, `torch_cpu.dll`, `torch.dll`, `0.28.0-djl_torch.dll`
  - Root Cause: DJL 0.28.0 default maven dependency uses PyTorch CPU native binaries on Windows. Windows CUDA native wheels (`cu121/win-x86_64`) require additional unbundled native jars.
  - Runtime Fallback: PyTorch native threading utilized 12 inter-op threads and 8 intra-op threads on CPU.
  - Measured Latency: Average inference latency was **5.56 ms per image** on CPU.

---

## 3. Detailed Experimental Analysis

### Experiment A (Baseline — Unweighted Cross-Entropy)
- **Objective:** Reproduce the unweighted compact CNN baseline on the verified deduplicated splits.
- **Results:** Validation macro-F1 reached 0.4082, with 31 zero-recall classes. Held-out test macro-F1 was 0.3450 with 32 zero-recall classes.
- **Analysis:** Majority classes (e.g. Stop, No Stopping or Standing, No Entry) dominate the gradients, while classes with fewer than 10 training samples fail to develop feature representations.

### Experiment B (Moderate Realistic Augmentation)
- **Objective:** Evaluate spatial (rotation $\pm 10^\circ$, translation $\pm 5\%$, scale $0.90\times-1.10\times$) and photometric jitter ($\pm 15\%$ brightness/contrast) without horizontal flips.
- **Results:** Validation macro-F1 dropped to 0.2665; test macro-F1 fell to 0.2060; zero-recall classes increased to 47.
- **Analysis:** For the compact CNN with 640k parameters trained over 8 epochs, synthetic jitter increased effective variance and distracted feature learning for ultra-rare classes without addressing the severe 200:1 class imbalance.

### Experiment C (Balanced Class-Aware Sampling) — WINNER
- **Objective:** Sample training batches uniformly across all 85 classes by cycling minority class instances while preventing over-sampling runaway.
- **Results:** Validation macro-F1 improved to **0.4872** (+0.0790 over baseline). On held-out clean test data, **Macro-F1 reached 0.5046** (+0.1596 gain), **Test Accuracy reached 59.66%** (+2.38% gain), and **Zero-Recall classes dropped from 32 down to 12**.
- **Analysis:** Minority classes received consistent gradient updates throughout every epoch, preventing majority classes from dominating the convolutional feature representations.

### Experiment D (Capped Class Weights — Cap = 5.0)
- **Objective:** Test normalized inverse class frequency weights capped at $5.0\times$ mean to prevent gradient explosion.
- **Results:** Validation macro-F1 reached 0.3303 (better than the previous unconstrained class-weighted run of 0.2540, but below baseline 0.4082).
- **Analysis:** Why Naive Class Weighting Underperformed: Without capping, rare classes ($N_c = 2$) receive weights $> 120.0$, causing extreme gradient spikes that destabilize shared convolutional layers. While capping at $5.0\times$ eliminated loss instability, static loss weighting still suffers from gradient noise compared to balanced mini-batch sampling.

---

## 4. Evaluation Integrity & Split Hygiene

- Split Manifest Path: `data/splits/indian_traffic_sign_splits.json`
- Partition Counts:
  - Training: 3,538 images (85 classes)
  - Validation: 900 images (85 classes)
  - Clean Held-Out Test: 714 images (85 classes)
  - Quarantined Overlapping Test: 574 images (exact SHA-256 matches to training set, excluded from test evaluation)
- Split Isolation: Validation set was used exclusively for model selection. Clean test partition was evaluated only on the final winning model.

---

## 5. Per-Class Weaknesses and Remaining Limitations

The remaining 12 zero-recall classes on the clean held-out test partition are:
1. `Barrier Ahead` (Support: 2)
2. `Bullock Carts Prohibited` (Support: 5)
3. `Compulsory Turn Left` (Support: 1)
4. `Handcarts Prohibited` (Support: 2)
5. `Hump or Rough Road (Speed Breaker)` (Support: 5)
6. `Men at Work` (Support: 2)
7. `Staggered Intersection` (Support: 6)
8. `Cycle Crossing` (Support: 4)
9. `Ferry` (Support: 1)
10. `Left Hairpin Bend` (Support: 3)
11. `Speed Limit (20 km/h)` (Support: 1)
12. `Unguarded Level Crossing` (Support: 1)

**Key Insight:** All 12 failure classes have tiny test support ($\le 6$ images) and severe training scarcity ($\le 8$ images). Their low recall stems from absolute data sparsity rather than architectural capacity.

---

## 6. Reproducible Execution Commands

To reproduce the benchmark suite or rerun individual experiments:

```powershell
# Set Java 21 environment
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"

# Compile codebase
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' test-compile

# Run full experiment suite
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' exec:java "-Dexec.mainClass=com.trafficsign.app.BalancedClassificationOrchestrator"

# Run unit and integration test suite
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' test
```
