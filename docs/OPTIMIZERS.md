# Neural Network Optimizers Specification

**Package:** `com.trafficsign.neural.optimizer`  
**Language:** Java 21 LTS  

This document details the mathematical formulation, update equations, and implementation details for the four optimizers implemented directly in pure Java.

---

## 1. Optimizer Interface (`Optimizer.java`)

All parameter update strategies implement a unified interface:
```java
public interface Optimizer {
    void initialize(int numLayers, int[][] weightShapes, int[] biasLengths);
    void update(int layerIdx, Matrix weights, double[] biases, Matrix gradWeights, double[] gradBiases);
    double getLearningRate();
}
```
This interface guarantees that the MLP training loop remains decoupled from optimizer-specific velocity buffers or learning rate schedules.

---

## 2. Implemented Optimizers

### A. Batch Gradient Descent (`BatchGradientDescent.java`)
* **Equation:**
  $$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}$$
  $$b \leftarrow b - \eta \cdot \nabla_b \mathcal{L}$$
* **Behavior:** Updates parameters using the exact average gradient computed over the entire batch.
* **State:** Stateless (requires no persistent buffers).

### B. Stochastic Gradient Descent (`StochasticGradientDescent.java`)
* **Equation:**
  $$\eta_t = \frac{\eta_0}{1 + \lambda \cdot t}$$
  $$W \leftarrow W - \eta_t \cdot \nabla_W \mathcal{L}$$
  $$b \leftarrow b - \eta_t \cdot \nabla_b \mathcal{L}$$
* **Behavior:** Processes small mini-batches (e.g., $N=32$) with an optional epoch-based inverse time decay schedule.

### C. Polyak Momentum (`MomentumOptimizer.java`)
* **Equations:**
  $$v_W \leftarrow \gamma v_W + \eta \cdot \nabla_W \mathcal{L}$$
  $$W \leftarrow W - v_W$$
  $$v_b \leftarrow \gamma v_b + \eta \cdot \nabla_b \mathcal{L}$$
  $$b \leftarrow b - v_b$$
* **Hyperparameters:** Learning rate $\eta > 0$, momentum factor $\gamma \in [0.0, 1.0)$ (typically $0.9$).
* **Behavior:** Accumulates velocity vectors along consistent gradient directions, dampening oscillations in narrow ravines.
* **State:** Independent per-layer velocity matrices $v_W^{[l]}$ and bias vectors $v_b^{[l]}$.

### D. Nesterov Accelerated Gradient (`NesterovOptimizer.java`)
* **Equations:**
  $$v_{\text{new}} \leftarrow \gamma v_{\text{prev}} + \eta \cdot \nabla_W \mathcal{L}$$
  $$W \leftarrow W - (\gamma v_{\text{new}} + \eta \cdot \nabla_W \mathcal{L})$$
  $$v \leftarrow v_{\text{new}}$$
* **Behavior:** Computes a look-ahead step. If the accumulated velocity is pushing parameters up an opposing slope, the look-ahead gradient detects the increase and applies braking resistance before overshooting.

---

## 3. Empirical Verification Summary

* Verified with automated unit tests in `OptimizerTest.java`:
  * Updates move parameters along the negative gradient direction.
  * Momentum correctly accumulates velocity across consecutive steps ($v_1 = 0.20 \rightarrow v_2 = 0.38$).
  * Look-ahead step in Nesterov produces larger initial braking response.
  * Two separate optimizer instances maintain independent state buffers without cross-model contamination.
