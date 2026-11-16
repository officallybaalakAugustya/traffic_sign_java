# Requirement-to-Evidence Matrix

**Course:** Deep Learning (CSAI3027P)  
**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Last Updated:** Stage 0 (Initialization)  

This matrix maps every functional, technical, and academic requirement from the PBL specification and hackathon guidelines to the planned implementation, target source files, verification evidence, current status, and required faculty clarifications.

---

## Deliverable Classification Legend

* **[CORE]**: Mandatory requirement that directly impacts the core grading rubric. Must be implemented in pure Java 21 and verified with automated tests.
* **[ADVANCED]**: High-value extension that enhances the product capabilities (e.g., Web UI, Object Detection).
* **[NEEDS CLARIFICATION]**: Requirement where specification boundaries (e.g., Python training vs. pure Java training, VAE/GAN scope) must be confirmed with faculty.
* **[PROPOSED SCOPE REDUCTION]**: Items requiring explicit faculty sign-off if deemed beyond single-student timeline constraints.

---

## Requirement Mapping Table

| Req ID | Category | Requirement Description | Class | Planned Implementation & Architecture | Target Source Files | Verification & Evidence | Status | Faculty Clarification Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-01** | Platform | Primary implementation in Java 21 with standard Maven build. | [CORE] | Standard Maven project structure targeting Java 21 bytecode release. | `pom.xml`, `src/main/java/**` | `mvn clean test` succeeds cleanly on command line. | **COMPLETE (Stage 0)** | None. |
| **REQ-02** | Math | Tensor/matrix math fundamentals in pure Java (multiplication, dot product, reshaping, broadcasting). | [CORE] | Pure Java 2D/3D matrix and tensor math engine with shape validation. | `com.trafficsign.math.Matrix`, `com.trafficsign.math.Tensor` | Unit tests in `test/com/trafficsign/math/*Test.java` verifying matrix arithmetic against analytical references. | **COMPLETE (Stage 2)** | None. |
| **REQ-03** | Math | Numerical gradient checking in pure Java. | [CORE] | Finite difference approximation: $\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$ compared against backpropagation gradients. | `com.trafficsign.math.GradientChecker` | Automated gradient check tests verifying relative difference $< 10^{-6}$. | **COMPLETE (Stage 2)** | None. |
| **REQ-04** | DL Fundamentals | McCulloch-Pitts and Sigmoid artificial neuron models. | [CORE] | Pure Java implementation demonstrating threshold activations and logistic sigmoids. | `com.trafficsign.neural.neuron.McCullochPittsNeuron`, `SigmoidNeuron` | Unit tests verifying truth-table behaviors (AND, OR, NOT) and non-linear outputs. | **COMPLETE (Stage 2)** | None. |
| **REQ-05** | DL Fundamentals | Multi-Layer Perceptron (MLP) forward and backward propagation. | [CORE] | Full forward and backprop loop with configurable hidden layer sizes, activations, and weight initialization (Xavier/He). | `com.trafficsign.neural.mlp.DenseLayer`, `com.trafficsign.neural.mlp.MLPNetwork` | Gradient checking tests and synthetic non-linear classification tests (XOR, concentric circles). | **COMPLETE (Stage 2)** | None. |
| **REQ-06** | Optimization | Gradient descent optimizers: Batch GD, Mini-batch SGD, Momentum, and Nesterov. | [CORE] | Parameter update rule abstractions with tracking for velocity vectors and learning rate schedules. | `com.trafficsign.neural.optimizer.*` | Benchmark comparisons plotting convergence rate and loss decay over identical test problems. | **COMPLETE (Stage 2)** | None. |
| **REQ-07** | Data Pipeline | Dataset ingestion, preprocessing, normalization, and split manifests. | [CORE] | Buffered image loaders, bilinear resizing, mean/std normalization, and reproducible group-aware split generator. | `com.trafficsign.data.*` | Split manifest JSON files in `data/splits/`; zero-leakage test ensuring no scene overlap between train and test. | **COMPLETE (Stage 1)** | Confirm approved dataset (GTSRB recommended). |
| **REQ-08** | CNN Classifier | Convolutional Neural Network for traffic sign classification. | [CORE] | Explainable CNN architecture (Conv2D, BatchNorm, ReLU, MaxPool, Dense, Softmax) trained and executed in Java. | `com.trafficsign.neural.djl.GtsrbCnnBlock`, `com.trafficsign.neural.djl.CnnTrainer` | Parameter calculation (629,739 params), synthetic smoke test, checkpoint serialization, confusion matrix, macro-F1. | **IN PROGRESS (Pre-Flight Verified)** | None (100% Java via DJL PyTorch engine). |
| **REQ-09** | Controlled Experiments | Ablation studies comparing optimizers, activations, data augmentation, and regularizers. | [CORE] | Experiment runner orchestrating runs with fixed random seeds; structured JSON logging. | `com.trafficsign.training.ExperimentRunner`, `docs/EXPERIMENT_LOG.md` | Logged comparison tables and loss/accuracy curves in `experiments/` and `results/`. | **PLANNED (Stage 4)** | None. |
| **REQ-10** | Evaluation | Comprehensive metrics: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix, and Latency. | [CORE] | Pure Java evaluation metrics calculation without third-party black-box metrics libraries. | `com.trafficsign.evaluation.ConfusionMatrix` | Unit tests with known confusion matrices (`ConfusionMatrixTest.java`); generated evaluation reports. | **COMPLETE (Stage 2/3)** | None. |
| **REQ-11** | Object Detection | Traffic sign detection (bounding boxes on full scenes). | [ADVANCED] | Integration of YOLO / object detection pipeline via Java ONNX Runtime. | `com.trafficsign.detection.SignDetector` | mAP@0.50 evaluation on GTSDB or full road scenes; annotated preview images. | **PLANNED (Stage 5)** | Verify GTSDB dataset compatibility. |
| **REQ-12** | Temporal Risk Model | Temporal risk modeling across sequential road frames. | [NEEDS CLARIFICATION] | Sliding window risk scoring based on sign sequence transitions and detected proximity. | `com.trafficsign.neural.temporal.RiskScorer` | Frame sequence simulation tests with simulated vehicle speeds and sign encounters. | **PLANNED (Stage 6)** | **CLARIFICATION NEEDED:** Is an explicit RNN/LSTM mandatory, or is a rule-and-proximity sliding risk model accepted? |
| **REQ-13** | Generative Augmentation | Generative model (VAE / GAN) data augmentation. | [PROPOSED SCOPE REDUCTION] | Comparative study evaluating synthetic sign generation vs. geometric/photometric augmentation. | `docs/GENERATIVE_AUGMENTATION_STUDY.md` | Report comparing classification accuracy with and without synthetic data. | **PLANNED (Stage 6)** | **PROPOSAL:** Propose treating VAE/GAN as an offline comparative analysis to maintain clean Java runtime. |
| **REQ-14** | Web Product | Interactive web interface for image upload, visual annotations, confidence, and metrics display. | [ADVANCED] | Lightweight Java web service (Spring Boot or Javalin) serving REST endpoints and responsive web frontend. | `com.trafficsign.web.*` | End-to-end integration test; browser upload demonstration. | **PLANNED (Stage 7)** | **CLARIFICATION NEEDED:** Confirm whether an embedded Java web server (e.g., Javalin/Spring Boot) is acceptable. |
| **REQ-15** | Academic Integrity | Documentation of sources, third-party code, and individual contribution. | [CORE] | Maintained source log, dataset card, model card, and viva preparation guide. | `docs/SOURCE_AND_CONTRIBUTION_LOG.md`, `docs/DATASET_CARD.md`, `docs/MODEL_CARD.md` | Complete citation of all academic papers, datasets, and libraries used. | **ACTIVE (Stage 0)** | None. |

---

## Action Items for Faculty Consultation

Before advancing past Stage 1, the student should present the following specific questions to the course coordinator:

1. **Training Pipeline Architecture Confirmation:**
   * *"Confirming that the training pipeline is 100% Java 21 using Deep Java Library (DJL) with PyTorch C++ native bindings, eliminating Python runtime dependencies."*
2. **Web Framework Selection:**
   * *"Is an embedded Java lightweight HTTP server (such as Javalin or Spring Boot) acceptable for the web presentation interface?"*
3. **Temporal Modeling Deliverable Definition:**
   * *"Can the temporal risk model be demonstrated via a multi-frame sliding window risk scoring module over simulated dashcam sequences?"*
