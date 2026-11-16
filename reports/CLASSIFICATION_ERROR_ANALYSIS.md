# Classification Error Analysis Report (Experiment C Champion)

**Dataset Partition:** Sequence-Held-Out Validation Set (900 natural images, 85 classes)
**Model Evaluated:** Experiment C (Balanced Class-Aware Sampling)
**Analysis Timestamp:** 2026-10-19T22:01:03.680179200Z

## 1. Aggregate Validation Performance

- **Overall Accuracy:** 62.33%
- **Macro-Averaged F1 Score:** 0.4872
- **Weighted-Averaged F1 Score:** 0.6152
- **Macro-Averaged Recall:** 0.5209
- **Zero-Recall Classes:** 17 / 85

## 2. Top 10 Most Frequently Confused Class Pairs

| True Class ID | True Class Name | Predicted As ID | Predicted Class Name | Confusion Count |
| :---: | :--- | :---: | :--- | :---: |
| 52 | Speed Limit (50 km/h) | 50 | Speed Limit (30 km/h) | 13 |
| 50 | Speed Limit (30 km/h) | 52 | Speed Limit (50 km/h) | 11 |
| 9 | Compulsory Keep Right | 12 | Compulsory Turn Left | 8 |
| 36 | Pedestrian Crossing | 45 | Side Road Left | 7 |
| 53 | Speed Limit (60 km/h) | 51 | Speed Limit (40 km/h) | 7 |
| 55 | Speed Limit (80 km/h) | 51 | Speed Limit (40 km/h) | 7 |
| 14 | Cross Road | 19 | Gap in Median | 6 |
| 24 | Horn Prohibited | 19 | Gap in Median | 6 |
| 48 | Speed Limit (5 km/h) | 49 | Speed Limit (15 km/h) | 6 |
| 51 | Speed Limit (40 km/h) | 53 | Speed Limit (60 km/h) | 6 |

## 3. Zero-Recall Classes on Validation Partition

- Class 2: Barrier Ahead (Support: 1)
- Class 15: Cycles Prohibited (Support: 9)
- Class 16: Dangerous Dip (Support: 3)
- Class 21: Guarded Level Crossing (Support: 3)
- Class 40: Restriction Ends (Support: 1)
- Class 58: Steep Descent (Support: 1)
- Class 61: T-Intersection (Support: 2)
- Class 69: Compulsory Turn Right (Support: 1)
- Class 70: Cycle Crossing (Support: 9)
- Class 71: Ferry (Support: 1)
- Class 72: Left Hairpin Bend (Support: 2)
- Class 75: Load Limit (Support: 2)
- Class 77: Right Reverse Bend (Support: 2)
- Class 79: Road Widens Ahead (Support: 1)
- Class 80: Speed Limit (20 km/h) (Support: 1)
- Class 81: Tonga Prohibited (Support: 1)
- Class 83: Unguarded Level Crossing (Support: 1)

## 4. Exemplar Predictions (Correct and Misclassified)

### Correctly Classified Examples

- Image: `train/ALL_MOTOR_VEHICLE_PROHIBITED/00014.jpg` -> **All Motor Vehicles Prohibited** (Class 0) [Correct]
- Image: `train/ALL_MOTOR_VEHICLE_PROHIBITED/00015.jpg` -> **All Motor Vehicles Prohibited** (Class 0) [Correct]
- Image: `train/AXLE_LOAD_LIMIT/01000.jpg` -> **Axle Load Limit** (Class 1) [Correct]
- Image: `train/AXLE_LOAD_LIMIT/01001.jpg` -> **Axle Load Limit** (Class 1) [Correct]
- Image: `train/AXLE_LOAD_LIMIT/01005.jpg` -> **Axle Load Limit** (Class 1) [Correct]

### Misclassified Examples

- Image: `train/ALL_MOTOR_VEHICLE_PROHIBITED/00003.jpg` -> True: **All Motor Vehicles Prohibited** (Class 0) | Predicted: **Pedestrians Prohibited** (Class 37)
- Image: `train/BARRIER_AHEAD/02005.jpg` -> True: **Barrier Ahead** (Class 2) | Predicted: **Loose Gravel** (Class 28)
- Image: `train/BULLOCK_AND_HANDCART_PROHIBITED/03004.jpg` -> True: **Bullock and Handcart Prohibited** (Class 66) | Predicted: **Bullock Carts Prohibited** (Class 3)
- Image: `train/BULLOCK_PROHIBITED/04003.jpg` -> True: **Bullock Carts Prohibited** (Class 3) | Predicted: **Trucks Prohibited** (Class 62)
- Image: `train/CATTLE/05000.jpg` -> True: **Cattle Crossing** (Class 4) | Predicted: **Dangerous Dip** (Class 16)
- Image: `train/CATTLE/05010.jpg` -> True: **Cattle Crossing** (Class 4) | Predicted: **Falling Rocks** (Class 18)
- Image: `train/CATTLE/05018.jpg` -> True: **Cattle Crossing** (Class 4) | Predicted: **Side Road Right** (Class 46)
- Image: `train/CATTLE/05024.jpg` -> True: **Cattle Crossing** (Class 4) | Predicted: **Dangerous Dip** (Class 16)
- Image: `train/CATTLE/05030.jpg` -> True: **Cattle Crossing** (Class 4) | Predicted: **Staggered Intersection** (Class 56)
- Image: `train/COMPULSARY_AHEAD/06002.jpg` -> True: **Compulsory Ahead Only** (Class 5) | Predicted: **Turn Right Ahead** (Class 63)

## 5. Root Cause & Correlation Analysis

1. **Correlation with Training Scarcity:** All classes with 0% recall had fewer than 15 training samples in the train split. Despite balanced class sampling, ultra-rare classes lack diversity in lighting, perspective, and background.
2. **Visual & Semantic Overlap:** Primary confusions concentrate within categories sharing geometric frames (circular red speed limits, e.g. 50 km/h vs 60 km/h vs 40 km/h) or triangular cautionary signs with minor interior symbol variations (e.g. Right Turn Prohibited vs Compulsory Turn Right).
3. **Input Resolution Constraints:** Downsampling to 32x32 pixels significantly blurs fine numerical fonts on speed limit signboards, hindering distinction between '50' and '60'.
