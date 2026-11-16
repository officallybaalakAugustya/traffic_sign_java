# Limitations and Responsible Use Specification

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Classification:** Academic PBL Software Prototype (CSAI3027P)  

---

## 1. Safety and Operational Scope

### Explicit Non-Certification Notice
This software is an educational prototype developed for learning and research purposes.
* **NOT Certified for Automotive Use:** This software is **not** certified under ISO 26262 (Road Vehicles — Functional Safety) or ISO 21448 (Safety of the Intended Functionality - SOTIF).
* **No Actuation Linkage:** Under no circumstances should this code or its inference outputs be wired to vehicle control units, throttle, steering, or braking systems.
* **Driver Responsibility:** In any demonstration involving a real vehicle or driving simulator, the human driver retains 100% legal and operational responsibility for vehicle control.

---

## 2. Identified Technical Failure Modes

Real-world driving exhibits stochastic conditions that deep learning models frequently struggle to handle reliably:

1. **Occlusion & Obstruction:**
   * Traffic signs obscured by overhanging tree branches, construction barriers, or parked trucks may cause partial pattern loss, resulting in false negatives or severe misclassifications.
2. **Adverse Environmental Phenomena:**
   * **Dense Fog / Rain Streaks:** Reduces high-frequency edge information required for distinguishing numerical glyphs (e.g., distinguishing 60 vs. 80 km/h).
   * **Direct Lens Glare:** Low-angle morning/evening sun washing out sign color contrast.
   * **Night Driving / Poor Illumination:** Non-reflective or dirty signs dropping below the dynamic range of digital optical sensors.
3. **Out-of-Distribution (OOD) & Adversarial Vulnerability:**
   * Street art, graffiti, stickers, or billboards placed near roadways can trigger false positive sign detections.
   * Standard Softmax classifiers often predict high confidence ($> 90\%$) on completely novel objects because the final layer forces a probability distribution over fixed known classes.
4. **Regional & Regulatory Invariance Limits:**
   * Models trained on European datasets (GTSRB) do not understand North American MUTCD signs (e.g., yellow diamond warning signs) or region-specific Asian scripts without retraining.

---

## 3. Responsible Machine Learning Practices Implemented

To mitigate these risks transparently within our academic project:
* **Explicit Uncertainty Thresholds:** We reject predictions where maximum confidence is below $\tau = 0.60$, labeling them `"Uncertain / Inspection Required"`.
* **Honest Evaluation Reporting:** We report Macro-F1 alongside Accuracy to highlight performance on rare hazard classes.
* **Zero Mocking of Metrics:** All reported test results reflect real empirical executions over isolated test splits.
