# Experiment Log & Reproducibility Registry

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** CSAI3027P  
**Standard Random Seed:** `42`  

This document serves as the immutable audit log for all controlled training, ablation, and evaluation runs. Every experiment must be assigned a unique identifier (`EXP-XXX`) and record all required parameters.

---

## 1. Experiment Record Schema

Every experiment must record:
1. **Experiment ID:** Unique sequential identifier (e.g., `EXP-001`).
2. **Date & Timestamp:** ISO 8601 execution time.
3. **Hypothesis / Objective:** Exact question tested (e.g., "Compare SGD with Momentum vs. Adam on 43-class GTSRB").
4. **Dataset & Split Hash:** Dataset version, split manifest hash, total samples.
5. **Model Architecture:** Layer definitions, number of trainable parameters.
6. **Hyperparameters:** Batch size, learning rate ($\eta$), learning rate decay schedule, optimizer parameters ($\beta_1, \beta_2, \epsilon$), weight decay, dropout rate.
7. **Random Seed:** Fixed seed for weight initialization and dataset shuffling.
8. **Hardware & Environment:** CPU/GPU model, OS, Java version, runtime flags.
9. **Empirical Results:** Training loss, validation loss, test accuracy, macro-F1, inference latency (ms).
10. **Conclusions & Next Steps:** Analysis of outcome.

---

## 2. Experiment Run Registry

| Exp ID | Date | Objective | Model / Optimizer | Seed | Train Loss | Val Loss | Test Acc | Macro F1 | Latency | Status |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-000` | 2026-10-19 | Project Initialization & Test Harness Smoke Test | Smoke Test / None | 42 | N/A | N/A | N/A | N/A | < 50ms | **COMPLETE** |
| `EXP-001` | *Pending* | Stage 1 Baseline Classifier on GTSRB | Baseline / Heuristic | 42 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Planned* |
| `EXP-002` | *Pending* | Stage 2 Pure Java MLP (SGD) | 2-Layer MLP / SGD | 42 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Planned* |
| `EXP-003` | *Pending* | Stage 2 Pure Java MLP (Momentum vs. NAG) | 2-Layer MLP / Momentum | 42 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Planned* |
| `EXP-004` | *Pending* | Stage 3 CNN Classifier Benchmark | Deep CNN / Adam | 42 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Planned* |
| `EXP-005` | *Pending* | Stage 4 Data Augmentation Ablation | Deep CNN / Augmentation | 42 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Planned* |

---

## 3. Detailed Experiment Notes Template

```markdown
### [EXP-XXX] Title / Hypothesis
* **Date:** YYYY-MM-DD
* **Goal:** 
* **Configuration:**
  - Architecture: 
  - Optimizer: 
  - Learning Rate: 
  - Batch Size: 
  - Epochs: 
  - Seed: 42
* **Hardware & Runtime:** Java 25.0.2 / Windows 11 / AMD64
* **Observations:**
  - Epoch convergence behavior:
  - Overfitting observed at epoch:
* **Quantitative Findings:**
  - Test Accuracy: 
  - Test Macro-F1: 
  - Average Latency: 
* **Decision:** 
```
