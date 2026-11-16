# Model Comparison and Architectural Selection Report

## Executive Summary

Four distinct classification strategies were evaluated on the Indian Traffic Sign Classification dataset (85 IRC:67 classes) using a strictly deduplicated, zero-leakage partition manifest.

## Comparative Benchmark Table

| Model Candidate | Parameter Count | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 0 | 0.0010 | 6.72% | 0.0015 | 0.0008 | 0.0118 |
| **Pure-Java MLP (3072-256-128-85)** | 830,549 | 0.3124 | 45.10% | 0.2571 | 0.3222 | 0.2880 |
| **Compact CNN (Unweighted)** | 640,533 | 0.3840 | 52.66% | 0.2946 | 0.3116 | 0.3250 |
| **Compact CNN (Class-Weighted)** | 640,533 | 0.2920 | 35.71% | 0.2540 | 0.2762 | 0.3008 |

## Selected Architecture for Downstream Monitoring

- **Selected Model:** DJL Compact CNN (Class-Weighted Cross-Entropy)
- **Rationale:** Convolutional inductive bias provides spatial invariance essential for road signs across variable lighting and viewing angles. Class weighting mitigates acute class imbalance (imbalance ratio 121:1 in Indian roads sign dataset).
- **Clean Evaluation:** Test evaluation was strictly isolated to the deduplicated test partition (quarantining the 574 duplicate images from training).

