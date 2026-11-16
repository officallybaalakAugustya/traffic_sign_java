# Implementation Roadmap: Stages 0 through 8

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Target Delivery:** University Deep Learning PBL & Hackathon  
**Language:** Java 21  

This roadmap details the sequential, gated implementation stages. Work must not advance to subsequent stages until the acceptance criteria of the active stage are verified by tests or reproducible logs.

---

## Stage Progression Table

| Stage | Name | Primary Objective | Deliverables | Key Verification / Acceptance Criteria | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **0** | **Project Foundation** | Establish folder hierarchy, Maven configuration, Git repository, environment check, and architectural documentation. | Maven `pom.xml`, documentation suite in `docs/`, directory structure, `.gitignore`, initial smoke test. | Java 21 compilation succeeds; initial Maven build executes cleanly. | **COMPLETE** |
| **1** | **Dataset & Baseline** | Audit traffic sign datasets, implement image preprocessing, generate non-leaking split manifests, and establish a baseline classifier. | `com.trafficsign.data.*`, `data/splits/manifest.json`, baseline classifier architecture plan, initial evaluation metrics. | Zero-leakage split verification test passes; unit tests verify PPM loading, resizing, normalization. | **COMPLETE** |
| **2** | **Java Neural Fundamentals** | Implement core neural network mathematics from scratch in pure Java 21. | Matrix and Tensor engine, McCulloch-Pitts & Sigmoid neurons, MLP forward/backward pass, loss functions, optimizers (BGD, SGD, Momentum, NAG). | Automated unit tests pass; numerical gradient checker confirms analytical backpropagation gradients ($< 10^{-6}$ error). | **COMPLETE** |
| **3** | **CNN Classification Pipeline** | Implement or integrate CNN inference engine in Java 21 and compare against baseline. | CNN architecture / inference adapter, forward pass execution, learning curve tracker. | Outperforms baseline classifier; confusion matrix and per-class precision/recall generated. | **NEXT** |
| **4** | **Controlled Improvements** | Systematically evaluate optimizers, activations, regularizers, and data augmentations. | Ablation experiment runner, hyperparameter logs, comparative evaluation report. | Controlled experiments demonstrate quantifiable gains/losses with fixed random seeds. | **PENDING** |
| **5** | **Object Detection Module** | Implement sign localization and bounding box extraction on full scene road images. | Sign detector integration (YOLO / SSD), bounding box parsing, non-maximum suppression (NMS). | Intersection-over-Union (IoU) calculation passes; detection visualized on sample test images. | **PENDING** |
| **6** | **Advanced Components** | Assess feasibility, implement and evaluate temporal risk modeling and generative augmentation. | Frame sequence sliding-window risk scorer; comparative analysis of synthetic vs. standard augmentation. | Sequence risk tests verify progressive hazard scoring; comparative report completed. | **PENDING** |
| **7** | **Interactive Web Product** | Connect Java web service to the model pipeline, providing drag-and-drop preview and predictions. | Web UI dashboard, REST controllers for image upload and experiment viewing. | Unseen images upload seamlessly; predictions, confidence, bounding boxes, and warnings display clearly. | **PENDING** |
| **8** | **Final Validation & Viva** | Comprehensive test execution, documentation finalization, and viva defense rehearsal. | Technical report, viva question bank, clean build verification from scratch. | Full test suite passes; project successfully presented and defended. | **PENDING** |

---

## Detailed Stage Breakdown

### Stage 0: Project Foundation (Current)
* **Tasks:**
  1. Inspect root directory to prevent collisions.
  2. Create modular directory layout matching architectural specification.
  3. Author project documentation (`PROJECT_OVERVIEW.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_ROADMAP.md`, etc.).
  4. Configure Maven `pom.xml` with Java 21 release targeting.
  5. Author `.gitignore` and `LICENSE`.
  6. Verify environment JDK and Maven availability; execute clean build.
* **Exit Gate:** Clean `mvn test` execution and report generated.

### Stage 1: Dataset & Baseline
* **Tasks:**
  1. Investigate German Traffic Sign Recognition Benchmark (GTSRB) and Chinese Traffic Sign Database (TSRD).
  2. Implement `ImagePreprocessor` (PPM/PNG reader, bilinear interpolation to $32 \times 32$, channel normalization).
  3. Implement track-aware `SplitManager` ensuring images from the same physical road track remain in the same partition.
  4. Build a deterministic baseline (e.g., k-Nearest Neighbors or simple logistic baseline) to provide an empirical floor.
* **Exit Gate:** Reproducible `manifest.json` and baseline classification metrics recorded in `results/`.

### Stage 2: Java Neural Fundamentals
* **Tasks:**
  1. Build `Matrix` and `Tensor` classes in `com.trafficsign.math` with shape validation and broadcasting.
  2. Implement `McCullochPittsNeuron` and `SigmoidNeuron` demonstrating foundational logic operations.
  3. Implement `DenseLayer` and `MLPNetwork` supporting forward propagation, ReLU/Sigmoid/Softmax activations, and backpropagation.
  4. Implement `CategoricalCrossEntropy` and `MeanSquaredError` loss.
  5. Implement optimizers: Batch GD, Mini-batch SGD, Momentum, and Nesterov Accelerated Gradient.
  6. Implement `GradientChecker` to verify analytical derivatives against central finite differences.
* **Exit Gate:** 100% pass on mathematical unit tests and gradient checks.

### Stage 3: CNN Classification Pipeline
* **Tasks:**
  1. Define `SignClassifier` interface in `com.trafficsign.inference`.
  2. Configure CNN model architecture (Convolutional layers, pooling, flattening, dense layers, dropout).
  3. Implement Java inference engine (via ONNX Runtime Java bindings or native Java CNN forward pass).
  4. Benchmark classification against the Stage 1 baseline on the held-out test split.
* **Exit Gate:** CNN classifier running in Java; accuracy and macro-F1 surpass the baseline.

### Stage 4: Controlled Improvements & Ablation
* **Tasks:**
  1. Run controlled experiments comparing optimizers (SGD vs. Momentum vs. Adam).
  2. Compare activation functions (Sigmoid vs. ReLU vs. LeakyReLU).
  3. Implement geometric data augmentations (random rotations $\pm 10^\circ$, translation, contrast jitter).
  4. Evaluate L2 weight regularization and Dropout.
* **Exit Gate:** All runs recorded in `experiments/` and summarized in `results/` with fixed seeds.

### Stage 5: Object Detection Module
* **Tasks:**
  1. Integrate bounding box detection pipeline for full road images ($1280 \times 720$ or $1920 \times 1080$).
  2. Implement Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.
  3. Calculate Intersection-over-Union (IoU) and evaluate localization accuracy.
* **Exit Gate:** Full scene image produces localized bounding boxes around traffic signs.

### Stage 6: Advanced Components
* **Tasks:**
  1. Assess temporal risk modeling: implement sliding-window hazard tracking over consecutive video frames.
  2. Evaluate generative data augmentation: document feasibility and comparative impact of synthetic data.
* **Exit Gate:** Documented empirical analysis of advanced components.

### Stage 7: Interactive Web Product
* **Tasks:**
  1. Implement lightweight embedded web service (`SignRecognitionController`) using Javalin.
  2. Build clean frontend dashboard for drag-and-drop image upload and visual bounding box overlay.
  3. Display prediction confidence, latency, uncertainty warnings, and confusion matrix views.
* **Exit Gate:** Manual browser test validating end-to-end user workflow with unseen road images.

### Stage 8: Final Validation & Viva Preparation
* **Tasks:**
  1. Execute complete test suite from scratch on a clean environment.
  2. Finalize all documentation cards and audit logs.
  3. Review and practice viva defense questions in `docs/VIVA_PREPARATION.md`.
* **Exit Gate:** Project builds cleanly, runs reliably, and is ready for submission and viva presentation.
