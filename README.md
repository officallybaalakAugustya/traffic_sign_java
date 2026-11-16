# Indian Traffic Sign Recognition and Road Safety Monitor

**Course:** Deep Learning (CSAI3027P) — PBL & Individual Hackathon  
**Language:** Java 21 LTS  
**Build System:** Apache Maven  
**Current Milestone:** Indian Road Reorientation Complete — Taxonomy, Data Abstractions & Pre-Flight Validation  

---

> [!NOTE]
> **Strict Java-Only Implementation:** In compliance with course CSAI3027P requirements, this project contains **zero Python source files, Python scripts, Python notebooks, or Python runtime dependencies**. All mathematical operations, neural network fundamentals (MLP, backpropagation, optimizers, gradient checking), dataset splitting, image processing, taxonomy mapping, CNN construction, training orchestration, and multi-class evaluations are executed 100% in Java 21 via Apache Maven.

---

## 1. Project Identity and Scope

The **Indian Traffic Sign Recognition and Road Safety Monitor** is an academic computer-vision and deep learning system specifically reoriented toward **Indian roads, Indian traffic signs, and Indian driving conditions**.

### Key Subsystems
1. **First-Principles Neural Network Mechanics:** Pure Java 21 implementations of McCulloch-Pitts neurons, Sigmoid units, multi-layer perceptrons (MLP), analytical backpropagation, numerical gradient checking, and first-order optimizers (BGD, SGD, Momentum, Nesterov).
2. **IRC:67 Indian Traffic Sign Taxonomy:** Comprehensive 85-class mapping conforming to the **Indian Roads Congress (IRC:67)** standards across Mandatory (Regulatory), Cautionary (Warning), and Informatory categories, with machine-readable JSON mapping (`src/main/resources/taxonomies/indian_traffic_signs_irc67.json`).
3. **Dataset-Independent Pipeline:** High-integrity data pipeline supporting generic directory structures (`ImageFolderDatasetLoader`), SHA-256 duplicate detection, cross-split leakage checks (`DatasetIntegrityChecker`), bounding box validation with IoU (`BoundingBox`), and class imbalance audits (`ClassFrequencyReport`).
4. **Deep Java Library (DJL) Integration:** PyTorch 2.2.2 engine integration with GPU detection and CPU fallback (`GpuVerifier`).
5. **Road Safety Decision-Support Framework:** Translates recognized signs (e.g. unmarked speed breakers, cattle crossings, silence zones) into actionable driver assistance alerts.

> [!WARNING]
> **Academic Prototype Notice:** This project is an educational proof-of-concept for university evaluation. It is **not** a certified autonomous-driving, collision-avoidance, or life-critical safety system. Drivers retain full legal and operational responsibility under the Indian Motor Vehicles Act.

---

## 2. Prerequisites & Environment

* **Java Development Kit (JDK):** Version 21 LTS or newer (tested with Oracle JDK 25.0.2 LTS, compiling with `--release 21`).
* **Apache Maven:** Version 3.8.0 or newer (tested with Maven 3.9.11).
* **Python Runtime:** **NONE REQUIRED.** The project does not execute or rely on Python.
* **Operating System:** Windows, Linux, or macOS.

---

## 3. Current Implementation Status

| Stage / Component | Status | Verification Evidence |
| :--- | :---: | :--- |
| **Neural Fundamentals** | **COMPLETE** | 74 foundational tests passing (MLP backprop, numerical gradient checks, optimizers, learning rules, Matrix, Tensor). |
| **Indian Road Project Audit** | **COMPLETE** | `docs/INDIAN_ROAD_PROJECT_AUDIT.md` isolating legacy GTSRB code and identifying Indian requirements. |
| **Indian Dataset Research** | **COMPLETE** | `docs/INDIAN_TRAFFIC_DATASET_RESEARCH.md` evaluating 5 candidates and recommending `kannanwisen/Indian-Traffic-Sign-Classification` (CC-BY-4.0). |
| **Indian Sign Taxonomy (IRC:67)** | **COMPLETE** | `docs/INDIAN_TRAFFIC_SIGN_TAXONOMY.md` + `src/main/resources/taxonomies/indian_traffic_signs_irc67.json` (85 classes). |
| **Dataset-Independent Pipeline** | **COMPLETE** | `BoundingBox`, `ClassFrequencyReport`, `ImageFolderDatasetLoader`, `DatasetIntegrityChecker`, `GroupedDatasetSplitter`. |
| **Unit & Integration Tests** | **COMPLETE** | **94/94 tests passing cleanly** via Apache Maven. |
| **Dataset Acquisition** | **PENDING APPROVAL** | `docs/INDIAN_DATA_READINESS.md` establishes pre-flight checks, storage verification, and safeguards; **no real data downloaded yet**. |
| **Real Indian CNN Training** | **PLANNED** | `docs/INDIAN_ROAD_EXPERIMENT_PLAN.md` establishes MLP baseline, compact CNN, and residual comparison protocol. |

---

## 4. Documentation Suite

- [`INDIAN_ROAD_PROJECT_AUDIT.md`](docs/INDIAN_ROAD_PROJECT_AUDIT.md) — Complete repository audit and isolation strategy.
- [`INDIAN_TRAFFIC_DATASET_RESEARCH.md`](docs/INDIAN_TRAFFIC_DATASET_RESEARCH.md) — Comprehensive comparative analysis of Indian datasets and licensing.
- [`INDIAN_TRAFFIC_SIGN_TAXONOMY.md`](docs/INDIAN_TRAFFIC_SIGN_TAXONOMY.md) — Detailed IRC:67 taxonomy, shapes, colors, and safety decision-support alerts.
- [`INDIAN_DATA_READINESS.md`](docs/INDIAN_DATA_READINESS.md) — Safeguards, pre-flight checks, and acquisition plan.
- [`INDIAN_ROAD_EXPERIMENT_PLAN.md`](docs/INDIAN_ROAD_EXPERIMENT_PLAN.md) — Multi-model evaluation and Indian road robustness protocol.

---

## 5. Verification Commands

### Run Full Test Suite
Ensure `JAVA_HOME` points to your JDK 21+ installation:
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' test
```

### Verify GPU Runtime Status
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' test -Dtest=GpuVerifierTest
```
*(Honest fallback: Reports CPU execution if native CUDA DLLs are not in system path).*
