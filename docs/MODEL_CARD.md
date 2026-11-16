# Model Card: Traffic Sign Recognition Models

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Target Systems:** Pure Java MLP Engine & Java ONNX CNN Classifier  
**Status:** Architecture Design (Stage 0)  

---

## 1. Model Details

* **Model Developers:** Individual Student (Course CSAI3027P - Deep Learning)
* **Model Date:** October 2026
* **Model Types:**
  1. **Baseline Pedagogical Classifier:** Pure Java Multi-Layer Perceptron (MLP) with configurable hidden layers (e.g., $3072 \rightarrow 256 \rightarrow 128 \rightarrow 43$).
  2. **Main Convolutional Classifier:** Deep CNN (e.g., modified LeNet / SqueezeNet / ResNet-18) accepting $32 \times 32 \times 3$ image inputs and outputting logits over 43 traffic sign classes.
  3. **Object Detector (Stage 5 Extension):** YOLO / SSD architecture generating multi-scale bounding boxes on full-resolution road scenes.

---

## 2. Intended Use

* **Primary Intended Use:** Academic research, educational demonstration of neural network fundamentals, and offline analysis of road imagery within course CSAI3027P.
* **Intended Users:** University evaluators, students, and hackathon judges.
* **Out-of-Scope / Prohibited Uses:**
  * Direct integration into physical vehicle actuation or braking systems.
  * Real-time safety-critical driver replacement.
  * Any commercial deployment without independent formal verification and functional safety certification (ISO 26262).

---

## 3. Factors and Operational Domains

* **Geographic Domain:** Standard European traffic signage conformant with the Vienna Convention on Road Signs and Signals.
* **Lighting Conditions:** Operational under daytime, overcast, and well-lit dusk conditions.
* **Degraded Conditions:** Performance degrades under extreme adverse weather (heavy snowfall, blinding sun glare, dense fog) or physical sign obstruction ($> 40\%$ occlusion).

---

## 4. Evaluation Metrics & Reporting Plan

* **Classification Metrics:**
  * Top-1 Accuracy: Proportion of test images where highest probability matches ground truth.
  * Per-Class Precision & Recall: Critical for identifying whether rare hazard signs are missed.
  * Macro-Averaged F1-Score: Balanced score across all 43 classes irrespective of frequency.
  * Confusion Matrix: Pinpoints pairwise category confusion (e.g., 60 km/h vs. 80 km/h).
* **Calibration Metric:** Expected Calibration Error (ECE) assessing reliability of output confidence scores.
* **Latency:** CPU execution time per inference pass (in milliseconds).

---

## 5. Quantitative Performance (Pending Empirical Execution)

> [!NOTE]
> All quantitative benchmark metrics remain unpopulated until empirical training and evaluation are conducted in Stages 1 through 4. No synthetic or invented numbers are recorded.
