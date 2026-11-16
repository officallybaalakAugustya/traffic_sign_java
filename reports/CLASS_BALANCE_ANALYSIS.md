# Indian Traffic Sign Class Balance & Imbalance Analysis

**Dataset:** Indian Traffic Sign Classification (IRC:67 Taxonomy, 85 Classes)  
**Split Hygiene:** SHA-256 Deduplicated Grouping (0% Train/Val/Test Leakage)  

## Imbalance Metrics Overview

- **Total Recognized Classes:** 85
- **Minimum Class Support (Train):** 1 samples
- **Maximum Class Support (Train):** 200 samples
- **Severe Imbalance Ratio:** 200.00 : 1

## Detailed Per-Class Partition Representation

| ID | Canonical Class Name | Category | Train Support | Val Support | Clean Test Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | All Motor Vehicles Prohibited | MANDATORY | 13 | 3 | 5 |
| 1 | Axle Load Limit | MANDATORY | 14 | 4 | 5 |
| 2 | Barrier Ahead | CAUTIONARY | 6 | 1 | 2 |
| 3 | Bullock Carts Prohibited | MANDATORY | 8 | 2 | 5 |
| 4 | Cattle Crossing | CAUTIONARY | 29 | 7 | 11 |
| 5 | Compulsory Ahead Only | MANDATORY | 33 | 9 | 9 |
| 6 | Compulsory Ahead or Turn Right | MANDATORY | 27 | 8 | 11 |
| 7 | Compulsory Cycle Track | MANDATORY | 12 | 3 | 4 |
| 8 | Compulsory Keep Left | MANDATORY | 46 | 12 | 16 |
| 9 | Compulsory Keep Right | MANDATORY | 176 | 47 | 4 |
| 10 | Compulsory Minimum Speed | MANDATORY | 19 | 4 | 8 |
| 11 | Compulsory Sound Horn | MANDATORY | 36 | 11 | 1 |
| 12 | Compulsory Turn Left | MANDATORY | 33 | 9 | 1 |
| 13 | Compulsory Turn Right Ahead | MANDATORY | 23 | 6 | 4 |
| 14 | Cross Road | CAUTIONARY | 112 | 28 | 32 |
| 15 | Cycles Prohibited | MANDATORY | 35 | 9 | 8 |
| 16 | Dangerous Dip | CAUTIONARY | 10 | 3 | 4 |
| 17 | Direction / Side Indicator | INFORMATORY | 5 | 1 | 1 |
| 18 | Falling Rocks | CAUTIONARY | 23 | 6 | 8 |
| 19 | Gap in Median | CAUTIONARY | 144 | 36 | 24 |
| 20 | Give Way | MANDATORY | 54 | 13 | 16 |
| 21 | Guarded Level Crossing | CAUTIONARY | 15 | 3 | 4 |
| 22 | Handcarts Prohibited | MANDATORY | 20 | 7 | 2 |
| 23 | Height Limit | MANDATORY | 18 | 4 | 8 |
| 24 | Horn Prohibited | MANDATORY | 129 | 31 | 9 |
| 25 | Hump or Rough Road (Speed Breaker) | CAUTIONARY | 81 | 20 | 5 |
| 26 | Left Reverse Bend | CAUTIONARY | 15 | 5 | 7 |
| 27 | Left Turn Prohibited | MANDATORY | 97 | 29 | 0 |
| 28 | Loose Gravel | CAUTIONARY | 18 | 4 | 8 |
| 29 | Men at Work | CAUTIONARY | 45 | 8 | 2 |
| 30 | Narrow Road Ahead | CAUTIONARY | 18 | 5 | 8 |
| 31 | No Entry | MANDATORY | 130 | 44 | 25 |
| 32 | No Parking | MANDATORY | 33 | 11 | 14 |
| 33 | No Stopping or Standing | MANDATORY | 200 | 42 | 48 |
| 34 | Overtaking Prohibited | MANDATORY | 19 | 5 | 9 |
| 35 | Pass Either Side | INFORMATORY | 18 | 5 | 8 |
| 36 | Pedestrian Crossing | CAUTIONARY | 99 | 22 | 19 |
| 37 | Pedestrians Prohibited | MANDATORY | 25 | 6 | 11 |
| 38 | Priority for Oncoming Vehicles | MANDATORY | 11 | 3 | 5 |
| 39 | Quay Side or River Bank | CAUTIONARY | 34 | 9 | 15 |
| 40 | Restriction Ends | MANDATORY | 5 | 1 | 3 |
| 41 | Right Hairpin Bend | CAUTIONARY | 12 | 3 | 5 |
| 42 | Right Hand Curve | CAUTIONARY | 13 | 3 | 6 |
| 43 | Roundabout | CAUTIONARY | 40 | 10 | 11 |
| 44 | School Ahead | CAUTIONARY | 31 | 8 | 14 |
| 45 | Side Road Left | CAUTIONARY | 28 | 8 | 11 |
| 46 | Side Road Right | CAUTIONARY | 8 | 2 | 4 |
| 47 | Slippery Road | CAUTIONARY | 29 | 7 | 12 |
| 48 | Speed Limit (5 km/h) | MANDATORY | 78 | 21 | 12 |
| 49 | Speed Limit (15 km/h) | MANDATORY | 31 | 8 | 6 |
| 50 | Speed Limit (30 km/h) | MANDATORY | 192 | 46 | 18 |
| 51 | Speed Limit (40 km/h) | MANDATORY | 135 | 35 | 44 |
| 52 | Speed Limit (50 km/h) | MANDATORY | 157 | 43 | 18 |
| 53 | Speed Limit (60 km/h) | MANDATORY | 153 | 37 | 38 |
| 54 | Speed Limit (70 km/h) | MANDATORY | 129 | 31 | 7 |
| 55 | Speed Limit (80 km/h) | MANDATORY | 153 | 39 | 6 |
| 56 | Staggered Intersection | CAUTIONARY | 12 | 3 | 6 |
| 57 | Steep Ascent | CAUTIONARY | 11 | 3 | 5 |
| 58 | Steep Descent | CAUTIONARY | 6 | 1 | 3 |
| 59 | Stop | MANDATORY | 40 | 10 | 2 |
| 60 | Straight Prohibited | MANDATORY | 41 | 11 | 0 |
| 61 | T-Intersection | CAUTIONARY | 10 | 2 | 4 |
| 62 | Trucks Prohibited | MANDATORY | 13 | 3 | 6 |
| 63 | Turn Right Ahead | MANDATORY | 78 | 21 | 12 |
| 64 | U-Turn Prohibited | MANDATORY | 28 | 6 | 6 |
| 65 | Y-Intersection | CAUTIONARY | 21 | 5 | 9 |
| 66 | Bullock and Handcart Prohibited | MANDATORY | 6 | 2 | 2 |
| 67 | Compulsory Ahead or Turn Left | MANDATORY | 10 | 2 | 4 |
| 68 | Compulsory Turn Left Ahead | MANDATORY | 14 | 4 | 4 |
| 69 | Compulsory Turn Right | MANDATORY | 5 | 1 | 2 |
| 70 | Cycle Crossing | CAUTIONARY | 39 | 9 | 4 |
| 71 | Ferry | CAUTIONARY | 1 | 1 | 1 |
| 72 | Left Hairpin Bend | CAUTIONARY | 9 | 2 | 3 |
| 73 | Left Hand Curve | CAUTIONARY | 14 | 4 | 6 |
| 74 | Length Limit | MANDATORY | 12 | 3 | 4 |
| 75 | Load Limit | MANDATORY | 7 | 2 | 3 |
| 76 | Narrow Bridge | CAUTIONARY | 10 | 2 | 4 |
| 77 | Right Reverse Bend | CAUTIONARY | 9 | 2 | 4 |
| 78 | Right Turn Prohibited | MANDATORY | 22 | 5 | 2 |
| 79 | Road Widens Ahead | CAUTIONARY | 7 | 1 | 3 |
| 80 | Speed Limit (20 km/h) | MANDATORY | 6 | 1 | 1 |
| 81 | Tonga Prohibited | MANDATORY | 5 | 1 | 2 |
| 82 | Traffic Signal | CAUTIONARY | 8 | 2 | 4 |
| 83 | Unguarded Level Crossing | CAUTIONARY | 4 | 1 | 1 |
| 84 | Width Limit | MANDATORY | 13 | 3 | 6 |
