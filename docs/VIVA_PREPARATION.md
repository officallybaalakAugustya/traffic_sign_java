# Viva Voce & Technical Defense Preparation Guide

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** Deep Learning (CSAI3027P)  
**Implementation:** Java 21  

This guide provides rigorous, mathematically sound answers to anticipated viva questions from evaluators and hackathon judges.

---

## 1. Architectural & Language Justifications

### Q1: "Why did you implement this deep learning project in Java 21 rather than Python with PyTorch?"
* **Answer:**
  1. **Course Specification Compliance:** The course assignment explicitly mandates C, C++, or Java. Implementing in Java ensures 100% compliance without violating project constraints.
  2. **Pedagogical Value:** In Python, calling `nn.Linear()` or `F.cross_entropy()` hides the actual matrix multiplications, pointer traversals, and gradient derivations behind C++ backends. Implementing matrix math, forward passes, backpropagation, and finite-difference gradient checking from scratch in pure Java proves complete understanding of the underlying mechanics.
  3. **Production Robustness & Type Safety:** Java 21 LTS brings modern language constructs (strongly-typed `record` classes for tensors and prediction results, pattern matching, sealed interfaces, and lightweight Virtual Threads) which prevent entire classes of runtime type and dimension mismatch errors.
  4. **Clean Inference Execution:** For high-performance CNN execution, Java 21 interfaces with ONNX Runtime via official native JVM bindings, delivering zero-copy C++ execution speed inside an enterprise-grade JVM web service without requiring brittle inter-process communication with Python.

---

## 2. Neural Network Fundamentals & Mathematics

### Q2: "How does a McCulloch-Pitts neuron differ from a Sigmoid neuron?"
* **Answer:**
  * **McCulloch-Pitts Neuron (1943):**
    * Operates on binary inputs $x_i \in \{0, 1\}$.
    * Computes a weighted sum and applies a hard step threshold function:
      $$y = \begin{cases} 1 & \text{if } \sum_{i=1}^n w_i x_i \ge \theta \\ 0 & \text{if } \sum_{i=1}^n w_i x_i < \theta \end{cases}$$
    * *Limitation:* The hard step function has zero derivative everywhere (and is non-differentiable at $\theta$), making gradient-based backpropagation mathematically impossible.
  * **Sigmoid Neuron:**
    * Operates on continuous real inputs $x_i \in \mathbb{R}$.
    * Computes affine transformation $z = w^T x + b$ and applies smooth logistic activation:
      $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    * *Advantage:* Differentiable everywhere with an elegant analytical derivative:
      $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
      which forms the mathematical foundation for gradient descent.

---

### Q3: "Derive the backpropagation gradient for a single dense layer with Categorical Cross-Entropy loss."
* **Answer:**
  * Let the output layer produce logits $z = W a^{[l-1]} + b$, followed by Softmax $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.
  * The loss is $\mathcal{L}_{CE} = - \sum_k y_k \ln \hat{y}_k$.
  * By the chain rule:
    $$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_i}$$
  * Since $\frac{\partial \mathcal{L}}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$, and the derivative of Softmax is:
    $$\frac{\partial \hat{y}_k}{\partial z_i} = \begin{cases} \hat{y}_i (1 - \hat{y}_i) & \text{if } k = i \\ -\hat{y}_k \hat{y}_i & \text{if } k \ne i \end{cases}$$
  * Substituting back:
    $$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i(1 - \hat{y}_i) - \sum_{k \ne i} \frac{y_k}{\hat{y}_k} (-\hat{y}_k \hat{y}_i) = -y_i + y_i \hat{y}_i + \sum_{k \ne i} y_k \hat{y}_i = \hat{y}_i \left(\sum_{\text{all } k} y_k\right) - y_i$$
  * Since ground-truth one-hot vector sums to 1 ($\sum y_k = 1$):
    $$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$
  * Thus, the error vector at the output is simply **predictions minus true targets**!
  * Then, gradients for weights and biases:
    $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$
    $$\delta^{[l-1]} = (W^{[l]})^T \delta^{[l]} \odot \sigma'(z^{[l-1]})$$

---

### Q4: "What is the mathematical difference between SGD, Momentum, and Nesterov Accelerated Gradient (NAG)?"
* **Answer:**
  1. **Standard SGD:** Updates directly along the instantaneous gradient:
     $$W_{t+1} = W_t - \eta \nabla \mathcal{L}(W_t)$$
     *Suffers from oscillations in ravines and slow progress in flat plateaus.*
  2. **Momentum:** Accumulates an exponentially decaying velocity vector $v_t$:
     $$v_{t+1} = \gamma v_t + \eta \nabla \mathcal{L}(W_t)$$
     $$W_{t+1} = W_t - v_{t+1}$$
     *Dampens oscillations in perpendicular directions and accelerates down steady slopes.*
  3. **Nesterov Accelerated Gradient (NAG):** Computes gradient at a **look-ahead position** $W_t - \gamma v_t$:
     $$v_{t+1} = \gamma v_t + \eta \nabla \mathcal{L}(W_t - \gamma v_t)$$
     $$W_{t+1} = W_t - v_{t+1}$$
     *Acts as an adaptive brake: if velocity is carrying the parameters uphill, the look-ahead gradient detects this in advance and reduces momentum.*

---

## 3. Data Integrity & Computer Vision

### Q5: "What is the GTSRB track leakage trap, and how did you prevent it?"
* **Answer:**
  * GTSRB collected signs from vehicle-mounted video recordings where each sign is filmed across 30 consecutive video frames (track).
  * If images are partitioned into train/test using naive random sampling, frames 1 through 20 of Track 5 enter the training set, while frame 21 enters the test set. Because the background scenery, sunlight angle, and vehicle hood are identical, the model simply memorizes the background rather than learning the sign morphology, yielding artificially inflated test scores ($99\%+$).
  * **Our Solution:** We implemented track-aware splitting in `SplitManager.java`: all 30 frames belonging to a single track ID are routed exclusively into either the train, validation, or test split.

---

### Q6: "Why is Macro-F1 a better metric than raw Accuracy for GTSRB?"
* **Answer:**
  * GTSRB suffers from severe class imbalance: Class 1 ("Speed limit 30") has over 2,000 images, while Class 0 ("Speed limit 20") has only 180.
  * A naive model that correctly identifies frequent signs but fails completely on rare signs could still achieve $90\%$ accuracy.
  * In road safety, missing a rare sign (e.g., "Dangerous curve ahead" or "Road work") is hazardous.
  * **Macro-F1** calculates the F1-score for each class independently and averages them unweighted:
    $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C F1_c$$
    treating a rare class with equal importance to a frequent class.

---

### Q7: "How did you verify that your backpropagation code has no mathematical bugs?"
* **Answer:**
  * We implemented numerical finite-difference gradient checking in `GradientChecker.java`.
  * We compute the central difference:
    $$\nabla_{\text{num}} \mathcal{L}(\theta) = \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$
  * We compute the relative Euclidean difference against the analytical backpropagation gradient:
    $$\text{Error} = \frac{\|\nabla_{\text{anal}} - \nabla_{\text{num}}\|_2}{\|\nabla_{\text{anal}}\|_2 + \|\nabla_{\text{num}}\|_2}$$
  * Automated JUnit unit tests assert that this error is $< 10^{-6}$. If an index or sign error exists in backpropagation, this test fails immediately.
