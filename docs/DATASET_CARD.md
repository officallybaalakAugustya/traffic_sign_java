# Dataset Card: Traffic Sign Recognition Datasets

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** Deep Learning (CSAI3027P)  
**Status:** Verified Ingestion Architecture (Stage 1 Complete)  

---

## 1. Verified Candidate Dataset Audit

| Dataset | Official Source / Publisher | Verified URL / Repository | License & Usage | Samples & Classes | Format & Resolution | Critical Audit & Leakage Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GTSRB** (German Traffic Sign Recognition Benchmark) | Institut für Neuroinformatik (INI), Ruhr-Universität Bochum (Stallkamp et al., IJCNN 2011 / Neural Networks 2012) | Official Archive: `https://benchmark.ini.rub.de/gtsrb_dataset.html` (Mirrored on Kaggle & Academic Repositories) | Free for academic, educational, and research evaluation | **51,839 total images** across **43 classes** (~39,209 train, ~12,630 test) | Binary PPM (P6) & CSV annotations; variable $15 \times 15$ to $250 \times 250$ px | **Track/Sequence Leakage Risk:** Signs are recorded in 30-frame video clips. Random frame-level split creates massive data leakage ($99\%+$ artificial accuracy). Splitting **must group by track ID**. |
| **GTSDB** (German Traffic Sign Detection Benchmark) | INI, Ruhr-Universität Bochum (Houben et al., IJCNN 2013) | `https://benchmark.ini.rub.de/gtsdb_dataset.html` | Free for academic research | **900 full road scenes** (600 train, 300 test) | PPM images ($1360 \times 800$ px) + TSV bounding box annotations | Designed for sign localization and object detection (Stage 5). Complex natural backgrounds with 0 to 6 signs per image. |
| **TSRD** (Traffic Sign Recognition Database - China) | National Laboratory of Pattern Recognition (NLPR), CASIA | `http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html` | Academic / Non-commercial | **6,164 images** across **58 classes** | PNG/JPG; variable resolutions ($16 \times 16$ to $164 \times 152$) | Severe class imbalance; valuable for cross-national generalization tests. |
| **BelgiumTS** (Belgium Traffic Signs) | KU Leuven (Timofte et al., 2011) | `https://btsd.ethz.ch/` | Academic research | **7,095 sign instances** across 62 classes | PNG/PPM; 3D camera tracks | Useful for domain-adaptation experiments across different European member states. |

---

## 2. Primary Selected Benchmark: GTSRB

GTSRB is the definitive standard benchmark for European traffic-sign classification.

### Official Class Taxonomy (43 Classes)
The classes are grouped into four functional categories defined in `com.trafficsign.data.TrafficSignClass`:
1. **Prohibitory Signs (Circular, Red Border):**
   * Class 0: Speed limit (20km/h)
   * Class 1: Speed limit (30km/h)
   * Class 2: Speed limit (50km/h)
   * Class 3: Speed limit (60km/h)
   * Class 4: Speed limit (70km/h)
   * Class 5: Speed limit (80km/h)
   * Class 7: Speed limit (100km/h)
   * Class 8: Speed limit (120km/h)
   * Class 9: No passing
   * Class 10: No passing vehicles over 3.5 metric tons
   * Class 15: No vehicles
   * Class 16: Vehicles over 3.5 metric tons prohibited
   * Class 17: No entry
2. **Danger / Warning Signs (Triangular, Red Border):**
   * Class 11: Right-of-way at intersection
   * Class 13: Yield
   * Class 18: General caution
   * Class 19: Dangerous curve to the left
   * Class 20: Dangerous curve to the right
   * Class 21: Double curve
   * Class 22: Bumpy road
   * Class 23: Slippery road
   * Class 24: Road narrows on the right
   * Class 25: Road work
   * Class 26: Traffic signals
   * Class 27: Pedestrians
   * Class 28: Children crossing
   * Class 29: Bicycles crossing
   * Class 30: Beware of ice/snow
   * Class 31: Wild animals crossing
3. **Mandatory Signs (Circular, Blue Background):**
   * Class 33: Turn right ahead
   * Class 34: Turn left ahead
   * Class 35: Ahead only
   * Class 36: Go straight or right
   * Class 37: Go straight or left
   * Class 38: Keep right
   * Class 39: Keep left
   * Class 40: Roundabout mandatory
4. **Other Signs (Priority, End Restrictions):**
   * Class 6: End of speed limit (80km/h)
   * Class 12: Priority road
   * Class 14: Stop (Octagonal)
   * Class 32: End of all speed and passing limits
   * Class 41: End of no passing
   * Class 42: End of no passing vehicles over 3.5 metric tons

---

## 3. Physical Directory Structure & Annotation Specification

When uncompressed, raw GTSRB training data adheres to this layout:
```
data/raw/GTSRB/
└── Final_Training/
    └── Images/
        ├── 00000/
        │   ├── 00000_00000.ppm
        │   ├── 00000_00001.ppm
        │   ├── ...
        │   ├── 00000_00029.ppm  <-- 30 frames per track (Track 00000)
        │   ├── 00001_00000.ppm  <-- Track 00001
        │   └── GT-00000.csv
        ├── 00001/
        │   └── ...
        └── 00042/
```

### Raw CSV Annotation Header Format
```csv
Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
00000_00000.ppm;29;30;5;6;24;25;0
00000_00001.ppm;30;30;5;5;25;25;0
```
* **Filename:** The image file name.
* **Width, Height:** Dimensions in pixels of the uncropped image.
* **Roi.X1, Roi.Y1, Roi.X2, Roi.Y2:** Bounding box coordinates enclosing the sign inside the frame.
* **ClassId:** Integer identifier $(0 \le \text{ClassId} \le 42)$.

---

## 4. Track-Aware Splitting & Leakage Audit Engine

The Java data pipeline in `com.trafficsign.data` implements strict leakage prevention:

1. **`DatasetSplitter`:**
   * Extracts `trackId` from the filename (e.g. `00000` from `00000_00014.ppm`).
   * Groups all 30 consecutive approach frames of track $T$ as a single atomic unit.
   * Shuffles tracks deterministically with fixed seed $42$.
   * Partitions into **Train ($70\%$)**, **Validation ($15\%$)**, and **Test ($15\%$)**.
2. **`LeakageDetector`:**
   * Mathematically proves that:
     $$\text{Tracks}_{\text{train}} \cap \text{Tracks}_{\text{val}} = \emptyset$$
     $$\text{Tracks}_{\text{train}} \cap \text{Tracks}_{\text{test}} = \emptyset$$
     $$\text{Tracks}_{\text{val}} \cap \text{Tracks}_{\text{test}} = \emptyset$$
   * Checks for duplicate file paths and checks class representation across partitions.
3. **`SplitManifest` (`data/splits/manifest.json`):**
   * Stores complete JSON-serialized partition records with seed, timestamp, and distribution counts for complete reproducibility.

---

## 5. Pure Java Image Preprocessing Specification

1. **PPM Decoding:** Implemented in `PpmImageReader.java` (pure Java binary P6 & ASCII P3 reader, zero external native libraries).
2. **ROI Cropping:** Extracts region of interest using annotated bounding box `(Roi.X1, Roi.Y1, Roi.X2, Roi.Y2)`.
3. **Bilinear Resizing:** Smoothly scales from variable dimensions to standard $32 \times 32$ pixels.
4. **Channel Normalization:**
   * Min-Max scaling: $x \in [0, 255] \rightarrow x / 255.0f \in [0.0, 1.0]$.
   * Channel standardization: $(x - \mu) / \sigma$ using computed dataset statistics:
     $$\mu = [0.340, 0.312, 0.321], \quad \sigma = [0.272, 0.260, 0.266]$$
5. **Tensor Output:** Generates 3D float arrays in both $[C, H, W]$ and $[H, W, C]$ formats.
