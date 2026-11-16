# Neural Learning Rules & Associative Memory Models

**Package:** `com.trafficsign.neural.rules`  
**Language:** Java 21 LTS  

This document explains the four alternative neural learning paradigms implemented as standalone modules for viva voce demonstrations and theoretical comparison against backpropagation.

---

## 1. Hebbian Learning & Oja's Rule (`HebbianLearning.java`)

### Theory (Donald Hebb, 1949)
Biological synapses strengthen when presynaptic and postsynaptic neurons fire synchronously:
$$\Delta w_i = \eta \cdot x_i \cdot y$$
* **The Divergence Problem:** Unconstrained Hebbian learning produces positive feedback, causing weights to grow exponentially towards infinity.
* **Oja's Normalization Rule (Erkki Oja, 1982):**
  Introduces a weight decay term proportional to output variance:
  $$\Delta w_i = \eta \cdot y \cdot (x_i - y \cdot w_i)$$
  Under Oja's rule, weight vectors asymptotically converge to unit length ($\|w\|_2 \rightarrow 1.0$), extracting the first Principal Component (PCA) of the input data without supervision.

---

## 2. Competitive Learning (`CompetitiveLearning.java`)

### Theory (Winner-Take-All / Vector Quantization)
Unlike backpropagation where all neurons receive error signals, competitive learning models local cluster specialization.
* Given input vector $x \in \mathbb{R}^D$ and $K$ prototype vectors $w_1 \dots w_K$:
* **Competition:** Find the nearest prototype:
  $$k^* = \arg\min_k \|x - w_k\|_2$$
* **Adaptation:** Only the winning prototype is updated:
  $$w_{k^*} \leftarrow w_{k^*} + \eta (x - w_{k^*})$$
  *Losing units remain unchanged.*
* **Road Safety Application:** Unsupervised clustering of visual sign characteristics (grouping circular signs vs. triangular signs by shape/color signatures).

---

## 3. Error-Correction Learning (`ErrorCorrectionLearning.java`)

### Theory (Rosenblatt Perceptron & Widrow-Hoff Delta Rule)
Calculates an explicit scalar error between target and actual response:
$$e = d - y$$
$$\Delta w = \eta \cdot e \cdot x, \quad \Delta b = \eta \cdot e$$
* If $y = d$ (correct prediction), $e = 0$ and no weight update occurs.
* Verified on linearly separable logic operations (OR gate converges in $< 10$ epochs).

---

## 4. Hopfield Associative Memory & Energy Minimization (`HopfieldEnergyModel.java`)

### Theory (John Hopfield, 1982)
A recurrent network of $N$ symmetrically coupled bipolar neurons $s_i \in \{-1, +1\}$ with zero self-connections ($W_{ii} = 0$).
* **Hebbian Storage:**
  $$W_{ij} = \frac{1}{N} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu \quad (i \ne j)$$
* **Lyapunov Energy Function:**
  $$E(s) = -\frac{1}{2} \sum_{i \ne j} W_{ij} s_i s_j$$
* **Asynchronous Dynamics:**
  $$s_i \leftarrow \text{sgn}\left(\sum_j W_{ij} s_j\right)$$
* **Guaranteed Convergence:** The energy function $E(s)$ is monotonically non-increasing ($\Delta E \le 0$) upon each state flip, guaranteeing convergence to a stable local minimum (attractor basin).
* **Pattern Restoration Test:** When probed with a 1-bit corrupted pattern, the network reconstructs the stored original pattern while reducing energy from $E_{\text{corrupted}} = +0.75$ down to $E_{\text{stable}} = -0.75$.
