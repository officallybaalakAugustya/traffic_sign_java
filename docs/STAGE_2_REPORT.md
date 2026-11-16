# Stage 2 Milestone Report: Java Neural-Network Fundamentals

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** Deep Learning (CSAI3027P) — PBL & Hackathon  
**Target Platform:** Java 21 LTS | Apache Maven  
**Date:** October 2026  
**Status:** COMPLETE (60/60 Tests Passing)  

---

## 1. Executive Summary

Stage 2 establishes the core mathematical and deep-learning fundamentals implemented ground-up in pure Java 21 without reliance on high-level ML frameworks (no PyTorch, Deeplearning4j, or DJL for foundational mechanics).

All mathematical formulations, neuron abstractions, multi-layer forward/backward passes, four optimizers, finite-difference gradient checks, SVD low-rank approximations, and four classical learning rules have been implemented, verified, and integrated with the Stage 1 data pipeline.

---

## 2. Inventory of Implemented Components

### A. Linear Algebra & Numerical Mathematics (`com.trafficsign.math`)
1. [`Matrix.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/math/Matrix.java): Pure Java 2D matrix engine with strict dimension checks, defensive array copying, matrix multiplication, transposition, vector dot products, and column sums.
2. [`Tensor.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/math/Tensor.java): N-D tensor abstraction with flat buffer storage, stride calculation, reshaping, slicing, and explicit bias broadcasting.
3. [`SvdDemonstrator.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/math/SvdDemonstrator.java): Singular Value Decomposition and rank-$k$ low-rank approximation using Apache Commons Math3, calculating singular values, energy preservation, and Frobenius reconstruction error.
4. [`GradientChecker.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/math/GradientChecker.java): Central finite-difference gradient checker comparing analytical backpropagation against numerical approximations ($\epsilon = 10^{-5}$).

### B. Artificial Neurons (`com.trafficsign.neural.neuron`)
1. [`McCullochPittsNeuron.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/neuron/McCullochPittsNeuron.java): 1943 threshold logic unit evaluating binary logic gates (AND, OR, NOT).
2. [`SigmoidNeuron.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/neuron/SigmoidNeuron.java): Continuous unit with numerical saturation clamping outside $[-45.0, +45.0]$ to eliminate subnormal underflows and overflow.

### C. Activations & Loss Functions (`com.trafficsign.neural.activation`, `com.trafficsign.neural.loss`)
1. [`ActivationFunction.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/activation/ActivationFunction.java), [`ReLUActivation.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/activation/ReLUActivation.java), [`SigmoidActivation.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/activation/SigmoidActivation.java): Modular activation implementations.
2. [`Softmax.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/activation/Softmax.java): Numerically stable max-shift categorical probability engine.
3. [`CategoricalCrossEntropyLoss.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/loss/CategoricalCrossEntropyLoss.java): Multi-class cross-entropy loss with analytical $\frac{1}{N}(P - Y)$ gradients.

### D. Multilayer Perceptron (`com.trafficsign.neural.mlp`)
1. [`MLPNetwork.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/mlp/MLPNetwork.java): Configurable feed-forward network with Xavier/Glorot initialization, forward propagation, analytical backpropagation, and multi-class classification.

### E. Optimizers (`com.trafficsign.neural.optimizer`)
1. [`Optimizer.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/optimizer/Optimizer.java): Common decoupled interface.
2. [`BatchGradientDescent.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/optimizer/BatchGradientDescent.java): Full-batch gradient descent.
3. [`StochasticGradientDescent.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/optimizer/StochasticGradientDescent.java): Mini-batch SGD with inverse time decay.
4. [`MomentumOptimizer.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/optimizer/MomentumOptimizer.java): Polyak momentum with persistent velocity state.
5. [`NesterovOptimizer.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/optimizer/NesterovOptimizer.java): Nesterov look-ahead momentum.

### F. Learning Rules (`com.trafficsign.neural.rules`)
1. [`HebbianLearning.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/rules/HebbianLearning.java): Correlation learning and Oja's bounded normalization rule.
2. [`CompetitiveLearning.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/rules/CompetitiveLearning.java): Winner-Take-All vector quantization clustering.
3. [`ErrorCorrectionLearning.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/rules/ErrorCorrectionLearning.java): Perceptron / Widrow-Hoff delta rule.
4. [`HopfieldEnergyModel.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/neural/rules/HopfieldEnergyModel.java): Recurrent associative memory network minimizing Lyapunov energy.

---

## 3. Test Verification & Results

### Execution Command
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"; & "C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd" test
```

### Exact Results Summary
* **Result:** `BUILD SUCCESS`
* **Total Tests Run:** **60**
* **Failures:** **0** | **Errors:** **0** | **Skipped:** **0**
* **Total Execution Time:** **8.512 seconds**

| Test Class | Tests | Domain Verified |
| :--- | :---: | :--- |
| `AppTest` | 1 | Application metadata & Java runtime environment |
| `DatasetSplitterTest` | 3 | Zero track leakage, manifest JSON round-trip, leakage detection |
| `ImagePreprocessorTest` | 5 | Bilinear resizing, ROI crop, [0,1] normalization, CHW/HWC shapes |
| `PpmImageReaderTest` | 3 | Binary P6 reading, ASCII P3 reading, comment stripping |
| `TrafficSignClassTest` | 6 | All 43 continuous classes, ID lookups, speed limit mappings |
| `MatrixTest` | 7 | Matrix multiplication, addition, subtraction, transpose, dot products |
| `TensorTest` | 6 | Shapes, strides, slicing, bias broadcasting, reshape validation |
| `NeuronTest` | 11 | Logic gates (AND, OR, NOT), sigmoid derivatives, extreme input stability |
| `OptimizerTest` | 5 | BGD direction, momentum accumulation, Nesterov lookahead, isolated states |
| `MLPNetworkTest` | 4 | Configurable layers, probability distributions, toy problem convergence |
| `GradientCheckerTest` | 1 | Finite-difference relative error $< 10^{-6}$ across all 31 parameters |
| `SvdDemonstratorTest` | 3 | Thin SVD reconstruction, Frobenius error reduction, energy preservation |
| `LearningRulesTest` | 4 | Oja weight normalization, competitive WTA clustering, Hopfield recall |
| `MLPDatasetIntegrationTest` | 1 | Preprocessor CHW tensors seamlessly feeding into 3072-input MLP |

---

## 4. Environment Disclosure

* **Active JDK:** Oracle JDK 25.0.2 LTS (`C:\Program Files\Java\jdk-25.0.2`).
* **Target Bytecode:** Java 21 (`--release 21`).
* **Maven Version:** Apache Maven 3.9.11.
* *Note on Compatibility:* JDK 21 is not installed as a standalone folder on this system; compilation targets Java 21 bytecode release using the installed JDK 25.0.2 LTS compiler.

---

## 5. Status & Next Actions

* **Stage 2 Status:** **COMPLETE**.
* **What is Ready for Stage 3:**
  * Dataset pipeline output shapes match the MLP input dimension ($32 \times 32 \times 3 = 3072$).
  * The mathematical and neural engine is fully tested and capable of training baseline MLPs.
* **What Remains Blocked / Blocked Actions:**
  * Downloading the physical 300MB GTSRB archive requires user confirmation before full training.
  * Training pipeline architecture resolved: 100% Java-only training using DJL PyTorch C++ native engine (no Python training needed).
