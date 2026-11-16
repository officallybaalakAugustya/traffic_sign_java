# Neural Network Fundamentals Specification

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Package:** `com.trafficsign.neural.*`, `com.trafficsign.math.*`  
**Language:** Java 21 LTS  
**Verification:** 100% Automated Unit Tests & Finite-Difference Gradient Checking  

---

## 1. Linear Algebra & Tensor Representation

All linear algebra is implemented in pure Java to preserve pedagogical clarity and eliminate opaque library calls.

### Matrix Operations (`com.trafficsign.math.Matrix`)
* **Representation:** 2D row-major array `double[rows][cols]` with defensive copying on construction and extraction.
* **Shape Validation:** All operations (`add`, `subtract`, `multiply`, `elementMultiply`) enforce strict dimension assertions, throwing descriptive `IllegalArgumentException` on mismatch.
* **Matrix Multiplication:**
  $$C_{i, j} = \sum_{k=1}^K A_{i, k} B_{k, j}, \quad A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N} \implies C \in \mathbb{R}^{M \times N}$$

### Tensor Abstraction (`com.trafficsign.math.Tensor`)
* **Representation:** Arbitrary rank tensor stored in contiguous 1D flat storage `double[size]` with computed dimension strides:
  $$\text{stride}[i] = \prod_{j=i+1}^{\text{rank}-1} \text{shape}[j], \quad \text{stride}[\text{rank}-1] = 1$$
* **Broadcasting Invariants:**
  * Supported: Elementwise operations with scalars.
  * Supported: Row bias broadcasting across batch: $[N, D] + [D] \rightarrow [N, D]$ or $[N, D] + [1, D] \rightarrow [N, D]$.
  * Supported: Channel bias broadcasting: $[C, H, W] + [C] \rightarrow [C, H, W]$.
  * *Limitation:* Arbitrary multi-dimensional generalized broadcasting (NumPy-style) is intentionally not claimed or supported.

---

## 2. Artificial Neuron Models

### McCulloch-Pitts Neuron (1943) (`com.trafficsign.neural.neuron.McCullochPittsNeuron`)
* **Inputs:** Strictly binary $x_i \in \{0, 1\}$.
* **Mathematical Function:**
  $$y = \begin{cases} 1 & \text{if } \sum_{i=1}^n w_i x_i \ge \theta \\ 0 & \text{if } \sum_{i=1}^n w_i x_i < \theta \end{cases}$$
* **Threshold Logic Gates Verified:**
  * AND Gate: $w = [1.0, 1.0], \theta = 2.0$
  * OR Gate: $w = [1.0, 1.0], \theta = 1.0$
  * NOT Gate: $w = [-1.0], \theta = 0.0$

### Sigmoid Neuron (`com.trafficsign.neural.neuron.SigmoidNeuron`)
* **Inputs:** Real continuous values $x_i \in \mathbb{R}$.
* **Numerically Stable Logistic Activation:**
  $$\sigma(z) = \begin{cases} 0.0 & \text{if } z < -45.0 \\ 1.0 & \text{if } z > 45.0 \\ \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$
  *Clamping outside $[-45.0, 45.0]$ avoids IEEE 754 subnormal underflows and floating-point overflow.*
* **Analytical Derivative:**
  $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

---

## 3. Feed-Forward Multilayer Perceptron (`com.trafficsign.neural.mlp.MLPNetwork`)

### Layer Dimensions
* **Full Benchmark Configuration:** $3072 \rightarrow 256 \rightarrow 128 \rightarrow 43$
  * Input: $32 \times 32 \times 3 = 3072$ flattened pixels.
  * Hidden 1: 256 neurons (ReLU).
  * Hidden 2: 128 neurons (ReLU).
  * Output: 43 neurons (Softmax).
* **Test Architecture:** $D_{in} \rightarrow H_1 \dots H_k \rightarrow C_{out}$ configurable dynamically.

### Parameter Initialization
Weights are initialized using **Xavier / Glorot Uniform** initialization:
$$W^{[l]} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
Biases $b^{[l]}$ are initialized to zero vectors.

### Forward Propagation
For batch of $N$ samples, $A^{[0]} = X \in \mathbb{R}^{N \times n_0}$:
For layers $l = 1 \dots L$:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = \begin{cases} \text{ReLU}(Z^{[l]}) & \text{for hidden layers} \\ \text{Softmax}(Z^{[L]}) & \text{for output layer} \end{cases}$$

### Loss Function: Categorical Cross-Entropy
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \ln(p_{i, y_i} + \epsilon), \quad \epsilon = 10^{-15}$$

### Analytical Backpropagation Derivation
$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{1}{N} (P - Y) \in \mathbb{R}^{N \times C}$$
For layers $l = L$ down to 1:
$$\nabla_{W^{[l]}} \mathcal{L} = (A^{[l-1]})^T \delta^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}$$
$$\nabla_{b^{[l]}} \mathcal{L} = \sum_{i=1}^N \delta^{[l]}_{i, :} \in \mathbb{R}^{n_l}$$
If $l > 1$:
$$\delta^{[l-1]} = (\delta^{[l]} (W^{[l]})^T) \odot \sigma'(Z^{[l-1]})$$
All gradient steps are verified against central finite differences in `GradientChecker`.
