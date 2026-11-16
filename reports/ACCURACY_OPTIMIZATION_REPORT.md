# Accuracy & Macro-F1 Optimization Report

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Target Taxonomy:** 85 IRC:67 Traffic Sign Classes  
**Evaluation Device:** CPU (Intel Core i7-13550H / DJL PyTorch 2.2.2)  
**Notice:** *Academic decision-support prototype under IRC:67; not certified for autonomous vehicular control.*  

## Selected Winning Candidate

- **Selected Model:** **Experiment C (Balanced Sampling)**
- **Validation Macro-F1 (Selection Criterion):** 0.4872
- **Validation Accuracy:** 62.33%
- **Clean Held-Out Test Accuracy:** 59.66%
- **Clean Held-Out Test Macro-F1:** 0.5046
- **Clean Held-Out Test Weighted-F1:** 0.6006
- **Classes with Zero Recall:** 12 / 85
- **Best Checkpoint Path:** `models/checkpoints/exp_c_balanced_sampling/exp_c_balanced_sampling_best`

## Per-Class Evaluation on Held-Out Test Data

| ID | Class Label | Precision | Recall | F1 Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | All Motor Vehicles Prohibited | 15.00% | 60.00% | 0.2400 | 5 |
| 1 | Axle Load Limit | 50.00% | 40.00% | 0.4444 | 5 |
| 2 | Barrier Ahead | 0.00% | 0.00% | 0.0000 | 2 |
| 3 | Bullock Carts Prohibited | 0.00% | 0.00% | 0.0000 | 5 |
| 4 | Cattle Crossing | 30.00% | 27.27% | 0.2857 | 11 |
| 5 | Compulsory Ahead Only | 80.00% | 88.89% | 0.8421 | 9 |
| 6 | Compulsory Ahead or Turn Right | 100.00% | 45.45% | 0.6250 | 11 |
| 7 | Compulsory Cycle Track | 100.00% | 75.00% | 0.8571 | 4 |
| 8 | Compulsory Keep Left | 100.00% | 6.25% | 0.1176 | 16 |
| 9 | Compulsory Keep Right | 50.00% | 100.00% | 0.6667 | 4 |
| 10 | Compulsory Minimum Speed | 100.00% | 100.00% | 1.0000 | 8 |
| 11 | Compulsory Sound Horn | 20.00% | 100.00% | 0.3333 | 1 |
| 12 | Compulsory Turn Left | 0.00% | 0.00% | 0.0000 | 1 |
| 13 | Compulsory Turn Right Ahead | 25.00% | 50.00% | 0.3333 | 4 |
| 14 | Cross Road | 31.82% | 21.88% | 0.2593 | 32 |
| 15 | Cycles Prohibited | 100.00% | 62.50% | 0.7692 | 8 |
| 16 | Dangerous Dip | 60.00% | 75.00% | 0.6667 | 4 |
| 17 | Direction / Side Indicator | 100.00% | 100.00% | 1.0000 | 1 |
| 18 | Falling Rocks | 25.00% | 12.50% | 0.1667 | 8 |
| 19 | Gap in Median | 39.13% | 37.50% | 0.3830 | 24 |
| 20 | Give Way | 94.12% | 100.00% | 0.9697 | 16 |
| 21 | Guarded Level Crossing | 100.00% | 25.00% | 0.4000 | 4 |
| 22 | Handcarts Prohibited | 0.00% | 0.00% | 0.0000 | 2 |
| 23 | Height Limit | 100.00% | 25.00% | 0.4000 | 8 |
| 24 | Horn Prohibited | 85.71% | 66.67% | 0.7500 | 9 |
| 25 | Hump or Rough Road (Speed Breaker) | 0.00% | 0.00% | 0.0000 | 5 |
| 26 | Left Reverse Bend | 40.00% | 28.57% | 0.3333 | 7 |
| 28 | Loose Gravel | 60.00% | 37.50% | 0.4615 | 8 |
| 29 | Men at Work | 0.00% | 0.00% | 0.0000 | 2 |
| 30 | Narrow Road Ahead | 100.00% | 50.00% | 0.6667 | 8 |
| 31 | No Entry | 100.00% | 100.00% | 1.0000 | 25 |
| 32 | No Parking | 69.23% | 64.29% | 0.6667 | 14 |
| 33 | No Stopping or Standing | 100.00% | 100.00% | 1.0000 | 48 |
| 34 | Overtaking Prohibited | 45.45% | 55.56% | 0.5000 | 9 |
| 35 | Pass Either Side | 100.00% | 75.00% | 0.8571 | 8 |
| 36 | Pedestrian Crossing | 56.52% | 68.42% | 0.6190 | 19 |
| 37 | Pedestrians Prohibited | 50.00% | 72.73% | 0.5926 | 11 |
| 38 | Priority for Oncoming Vehicles | 100.00% | 60.00% | 0.7500 | 5 |
| 39 | Quay Side or River Bank | 70.00% | 46.67% | 0.5600 | 15 |
| 40 | Restriction Ends | 100.00% | 100.00% | 1.0000 | 3 |
| 41 | Right Hairpin Bend | 10.00% | 20.00% | 0.1333 | 5 |
| 42 | Right Hand Curve | 42.86% | 50.00% | 0.4615 | 6 |
| 43 | Roundabout | 71.43% | 45.45% | 0.5556 | 11 |
| 44 | School Ahead | 50.00% | 50.00% | 0.5000 | 14 |
| 45 | Side Road Left | 42.86% | 54.55% | 0.4800 | 11 |
| 46 | Side Road Right | 13.33% | 50.00% | 0.2105 | 4 |
| 47 | Slippery Road | 50.00% | 50.00% | 0.5000 | 12 |
| 48 | Speed Limit (5 km/h) | 100.00% | 83.33% | 0.9091 | 12 |
| 49 | Speed Limit (15 km/h) | 85.71% | 100.00% | 0.9231 | 6 |
| 50 | Speed Limit (30 km/h) | 71.43% | 55.56% | 0.6250 | 18 |
| 51 | Speed Limit (40 km/h) | 80.00% | 81.82% | 0.8090 | 44 |
| 52 | Speed Limit (50 km/h) | 57.14% | 22.22% | 0.3200 | 18 |
| 53 | Speed Limit (60 km/h) | 65.85% | 71.05% | 0.6835 | 38 |
| 54 | Speed Limit (70 km/h) | 62.50% | 71.43% | 0.6667 | 7 |
| 55 | Speed Limit (80 km/h) | 57.14% | 66.67% | 0.6154 | 6 |
| 56 | Staggered Intersection | 0.00% | 0.00% | 0.0000 | 6 |
| 57 | Steep Ascent | 27.27% | 60.00% | 0.3750 | 5 |
| 58 | Steep Descent | 100.00% | 66.67% | 0.8000 | 3 |
| 59 | Stop | 100.00% | 100.00% | 1.0000 | 2 |
| 61 | T-Intersection | 66.67% | 50.00% | 0.5714 | 4 |
| 62 | Trucks Prohibited | 57.14% | 66.67% | 0.6154 | 6 |
| 63 | Turn Right Ahead | 70.59% | 100.00% | 0.8276 | 12 |
| 64 | U-Turn Prohibited | 100.00% | 50.00% | 0.6667 | 6 |
| 65 | Y-Intersection | 70.00% | 77.78% | 0.7368 | 9 |
| 66 | Bullock and Handcart Prohibited | 100.00% | 50.00% | 0.6667 | 2 |
| 67 | Compulsory Ahead or Turn Left | 100.00% | 75.00% | 0.8571 | 4 |
| 68 | Compulsory Turn Left Ahead | 50.00% | 100.00% | 0.6667 | 4 |
| 69 | Compulsory Turn Right | 100.00% | 50.00% | 0.6667 | 2 |
| 70 | Cycle Crossing | 0.00% | 0.00% | 0.0000 | 4 |
| 71 | Ferry | 0.00% | 0.00% | 0.0000 | 1 |
| 72 | Left Hairpin Bend | 0.00% | 0.00% | 0.0000 | 3 |
| 73 | Left Hand Curve | 41.67% | 83.33% | 0.5556 | 6 |
| 74 | Length Limit | 100.00% | 100.00% | 1.0000 | 4 |
| 75 | Load Limit | 20.00% | 33.33% | 0.2500 | 3 |
| 76 | Narrow Bridge | 50.00% | 50.00% | 0.5000 | 4 |
| 77 | Right Reverse Bend | 100.00% | 25.00% | 0.4000 | 4 |
| 78 | Right Turn Prohibited | 40.00% | 100.00% | 0.5714 | 2 |
| 79 | Road Widens Ahead | 100.00% | 33.33% | 0.5000 | 3 |
| 80 | Speed Limit (20 km/h) | 0.00% | 0.00% | 0.0000 | 1 |
| 81 | Tonga Prohibited | 20.00% | 50.00% | 0.2857 | 2 |
| 82 | Traffic Signal | 100.00% | 50.00% | 0.6667 | 4 |
| 83 | Unguarded Level Crossing | 0.00% | 0.00% | 0.0000 | 1 |
| 84 | Width Limit | 100.00% | 66.67% | 0.8000 | 6 |
