# Architecture Specification: Traffic Sign Recognition & Road Safety Monitor

**Target Platform:** Java 21 (LTS)  
**Build Tool:** Apache Maven  
**Architecture Style:** Modular, Layered, Interface-Driven  

---

## 1. System Architectural Overview

The application architecture enforces strict decoupling between data ingestion, mathematical execution, neural abstractions, model inference, evaluation, and user presentation. This ensures:
1. Pure Java pedagogical transparency for foundational neural network mechanics.
2. Extensibility for high-performance inference engines.
3. Testability of individual mathematical components without requiring UI or network dependencies.

```mermaid
flowchart TD
    subgraph PresentationLayer [Presentation Layer]
        UI[Web UI: HTML5 / CSS / Vanilla JS]
        WebController[com.trafficsign.web.SignRecognitionController]
    end

    subgraph ApplicationLayer [Application Layer]
        AppEntry[com.trafficsign.app.Main]
        Config[com.trafficsign.config.AppConfig]
        Service[com.trafficsign.inference.InferenceService]
    end

    subgraph DetectionAndInference [Inference & Detection Layer]
        InferenceEngine[<<interface>> com.trafficsign.inference.SignClassifier]
        JavaClassifier[com.trafficsign.neural.mlp.JavaMLPClassifier]
        ONNXClassifier[com.trafficsign.inference.ONNXCNNClassifier]
        Detector[com.trafficsign.detection.YOLOSignDetector]
    end

    subgraph NeuralAndMathLayer [Neural & Mathematical Core]
        MLP[com.trafficsign.neural.mlp.MLPNetwork]
        Neuron[com.trafficsign.neural.neuron.*]
        Optimizer[com.trafficsign.neural.optimizer.*]
        Loss[com.trafficsign.neural.loss.*]
        GradCheck[com.trafficsign.math.GradientChecker]
        Matrix[com.trafficsign.math.Matrix]
        Tensor[com.trafficsign.math.Tensor]
    end

    subgraph DataAndEvaluation [Data & Evaluation Layer]
        DataLoader[com.trafficsign.data.DatasetLoader]
        Preprocessor[com.trafficsign.data.ImagePreprocessor]
        Splitter[com.trafficsign.data.SplitManager]
        Evaluator[com.trafficsign.evaluation.MetricsCalculator]
        ConfusionMatrix[com.trafficsign.evaluation.ConfusionMatrix]
        ExperimentTracker[com.trafficsign.evaluation.ExperimentTracker]
    end

    UI -->|HTTP / REST multipart upload| WebController
    WebController --> Service
    AppEntry --> Config
    AppEntry --> Service
    Service --> Preprocessor
    Service --> InferenceEngine
    Service --> Detector
    InferenceEngine <|.. JavaClassifier
    InferenceEngine <|.. ONNXClassifier
    JavaClassifier --> MLP
    MLP --> Matrix
    MLP --> Optimizer
    MLP --> Loss
    GradCheck --> Loss
    GradCheck --> MLP
    DataLoader --> Preprocessor
    DataLoader --> Splitter
    Evaluator --> ConfusionMatrix
    Evaluator --> ExperimentTracker
```

---

## 2. Layer Specifications and Interfaces

### Module A: Application Layer (`com.trafficsign.app`, `com.trafficsign.config`)
* **Role:** Bootstrapping, configuration ingestion, and dependency wiring.
* **Key Components:**
  * `Main.java`: Command-line launcher supporting CLI mode (training, evaluation, benchmarking) and Web server mode.
  * `AppConfig.java`: Strongly-typed configuration parser loading from `config/application.properties`.

### Module B: Dataset Layer (`com.trafficsign.data`)
* **Role:** Image decoding, color space transformation, bilinear resizing, and split management.
* **Core Contract:**
```java
public record ImageSample(int id, int classId, String classLabel, int width, int height, float[][][] rgbChannels) {}

public interface ImageTransformer {
    float[][][] transform(float[][][] input);
}
```
* **Responsibilities:**
  * Load PPM / PNG / JPEG images into standardized normalized arrays $[0.0, 1.0]$.
  * Prevent data leakage by respecting track/sequence IDs when generating train/val/test splits.
  * Save and load deterministic split manifests in JSON format.

### Module C: Mathematical Fundamentals (`com.trafficsign.math`)
* **Role:** Educational, transparent tensor and matrix operations implemented in pure Java.
* **Core Contracts:**
```java
public final class Matrix {
    public static Matrix multiply(Matrix a, Matrix b) { ... }
    public static Matrix add(Matrix a, Matrix b) { ... }
    public Matrix transpose() { ... }
    public Matrix applyFunction(DoubleUnaryOperator func) { ... }
}
```
* **Capabilities:**
  * 2D Matrix multiplication and dot products.
  * Reshaping and broadcasting for tensor operations.
  * Finite-difference numerical gradient checking (`GradientChecker.java`) ensuring analytical backward passes have relative error $< 10^{-6}$.

### Module D: Neural Network Fundamentals (`com.trafficsign.neural`)
* **Role:** Ground-up implementation of artificial neural network mechanics.
* **Components:**
  * `McCullochPittsNeuron`: Threshold logic unit demonstrating binary classification (AND, OR, NOT gates).
  * `SigmoidNeuron`: Differentiable single-unit model with logistic activation.
  * `DenseLayer`: Affine transformation $Z = W \cdot X + b$ and activation $A = \sigma(Z)$.
  * `MLPNetwork`: Multi-layer perceptron supporting arbitrary layer configurations, forward propagation, and backpropagation.
  * `LossFunction`: Interface for `CategoricalCrossEntropy` and `MeanSquaredError`.
  * `Optimizer`: Abstraction for `BatchGradientDescent`, `StochasticGradientDescent`, `MomentumOptimizer`, and `NesterovOptimizer`.

### Module E: Main Model & Inference Abstraction (`com.trafficsign.inference`, `com.trafficsign.detection`)
* **Role:** Common inference interface isolating model execution mechanics from upstream consumers.
* **Core Contract:**
```java
public record ClassificationResult(
    int classId,
    String labelName,
    double confidence,
    double[] classProbabilities,
    long inferenceLatencyMs,
    boolean isUncertain
) {}

public interface SignClassifier {
    ClassificationResult classify(float[][][] preprocessedImage);
    void loadModel(String modelPath);
}
```
* **Inference Strategy:**
  1. *Pedagogical Classifier (`JavaMLPClassifier`):* Executes pure Java forward passes using weights learned by the ground-up MLP engine.
  2. *High-Accuracy CNN Classifier (`ONNXCNNClassifier`):* Executes trained CNN weights (e.g., LeNet-5, ResNet-18) via Microsoft ONNX Runtime for Java (`com.microsoft.onnxruntime:onnxruntime`). This provides native Java execution without external runtime servers.
  3. *Sign Detector (`YOLOSignDetector`):* Performs multi-scale bounding box detection and class classification on high-resolution road scenes.

### Module F: Evaluation Layer (`com.trafficsign.evaluation`)
* **Role:** Unbiased, objective verification of model quality across evaluation splits.
* **Metrics Computed in Pure Java:**
  * Confusion Matrix ($C \times C$).
  * Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$.
  * Precision per class $c$: $\frac{TP_c}{TP_c + FP_c}$.
  * Recall per class $c$: $\frac{TP_c}{TP_c + FN_c}$.
  * Macro-F1: $\frac{1}{C} \sum_{c=1}^C 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$.
  * Inference Latency: p50, p95, p99 distribution over $N$ runs.

### Module G: Web Presentation Layer (`com.trafficsign.web`)
* **Framework Recommendation:**
  * *Option 1: Spring Boot:* Heavyweight, high startup overhead, extensive boilerplate.
  * *Option 2: Pure JDK `com.sun.net.httpserver.HttpServer`:* Zero external dependencies, but cumbersome multipart file upload handling.
  * *Option 3: Javalin (Recommended):* Lightweight (~100KB), embeddable microframework running on embedded Jetty. Extremely fast, supports Java 21 virtual threads, and provides clean REST routing and multipart upload handling.
* **API Endpoints:**
  * `POST /api/predict/classify`: Upload single cropped image $\rightarrow$ JSON classification result with confidence and latency.
  * `POST /api/predict/detect`: Upload road scene $\rightarrow$ JSON bounding boxes and annotated image.
  * `GET /api/experiments/summary`: Fetch JSON summary of benchmark experiments and confusion matrices.
  * `GET /`: Static single-page dashboard for image drag-and-drop, preview, and results display.

---

## 3. Technology Stack & Justification

| Component | Selected Technology | Technical Rationale |
| :--- | :--- | :--- |
| **Language** | **Java 21 (LTS)** | Required by university PBL course; provides modern language features (records, pattern matching, virtual threads, sequence collections). |
| **Build Tool** | **Apache Maven** | Standard enterprise build management; deterministic dependency resolution and reproducible packaging. |
| **Logging** | **SLF4J + Logback** | Clean separation of logging API from logging implementation; high throughput. |
| **Serialization** | **Jackson 2.17** | Type-safe JSON serialization for dataset split manifests and experiment metrics. |
| **Testing** | **JUnit 5 + AssertJ** | Modern test harness with parameterized testing support for mathematical gradient checks and confusion matrix assertions. |
| **Inference Runtime** | **ONNX Runtime (Java)** | Allows state-of-the-art CNN/YOLO models to run directly inside the Java Virtual Machine (JVM) at native speeds without Python inter-process communication. |
| **Web Server** | **Javalin / Embedded Jetty** | Minimalist, robust, low overhead, and fully self-contained inside the single Maven JAR. |
