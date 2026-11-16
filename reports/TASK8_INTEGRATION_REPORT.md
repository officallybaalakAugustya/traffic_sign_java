# Task 8 Integration Report: Road-Scene Detection & 85-Class Classification

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor
**Execution Timestamp:** 2026-10-20T04:47:39.360419800Z
**Academic Notice:** *Decision-support research prototype under IRC:67; not certified for autonomous vehicular control or safety-critical actuation.*

## 1. System Integration Architecture

The integrated end-to-end traffic-sign perception pipeline cascades two neural networks:
1. **Stage 1 (Single-Stage Road-Scene Detector):**
   - Model: `sign_detector` (`CompactSignDetectorBlock`, 242,885 parameters)
   - Input: $128 \times 128 \times 3$ RGB full road frame (downscaled from $1920 \times 994$)
   - Output: $[5 \times 8 \times 8]$ grid tensor with bounding box coordinates and objectness scores
   - Post-Processing: Non-Maximum Suppression (IoU threshold = 0.40, detection threshold = 0.30)
2. **Stage 2 (Boundary-Clipped Region Cropper & Preprocessor):**
   - Crops candidate bounding box from the native-resolution original image
   - Clips box coordinates to image dimensions $[0, W] \times [0, H]$
   - Safely discards degenerate or empty crops ($w \le 2$ or $h \le 2$)
   - Converts patch to standard 3-channel RGB (`BufferedImage.TYPE_INT_RGB`)
   - Resizes patch to $48 \times 48 \times 3$ RGB using verified bilinear interpolation
3. **Stage 3 (Task 7 Champion 85-Class Classifier):**
   - Model: `exp_b_48x48` (`IndianTrafficSignCnnBlock`, 1,295,893 parameters)
   - Output: 85 softmax probabilities mapped to IRC:67 canonical names and risk alerts
4. **Stage 4 (Uncertainty-Gated Visual Renderer):**
   - Categorizes prediction into `CONFIDENT`, `LOW_CLASSIFICATION_CONFIDENCE`, `LOW_DETECTION_CONFIDENCE`, or `UNKNOWN_CLASS`
   - Renders color-coded bounding boxes and semi-transparent label pills on a separate output frame

## 2. Quantitative Performance & Latency Benchmark

### Latency and Throughput Profile (CPU Execution)

| Pipeline Stage | Latency | Share of Budget |
| :--- | :---: | :---: |
| Stage 1: Road-Scene Detector (128x128) | 34.18 ms | 5.0% |
| Stage 2 & 3: Crop + Classifier (48x48) | 610.30 ms | 89.8% |
| Stage 4: Rendering & Overlay Drawing | 35.52 ms | 5.2% |
| **Total End-to-End Latency** | **680.00 ms** | **100.0%** |
| **End-to-End Throughput** | **1.5 FPS** | *Real-time video budget: 30 FPS (33.3 ms)* |

### Detection Performance on Ground-Truth Road Driving Sample

| Evaluation Metric | Threshold IoU = 0.50 | Threshold IoU = 0.30 |
| :--- | :---: | :---: |
| **Precision** | 11.88% | 18.81% |
| **Recall** | 9.84% | 15.57% |
| **F1 Score** | 0.1076 | 0.1704 |
| **True Positives (TP)** | 12 | 19 |
| **False Positives (FP)** | 89 | 82 |
| **False Negatives / Missed (FN)** | 110 | 103 |

## 3. Demonstration Artifacts

- Annotated Demonstration Image: `reports/detection_demo/annotated_driving_frame.png`
- Machine-Readable Metrics: `reports/task8_integration_metrics.json`
- Classifier Checkpoint: `models/checkpoints/task7_champion`
- Detector Checkpoint: `models/checkpoints/detector/detector_best`

## 4. Run Instructions for Demonstration

To execute the demonstration pipeline on any arbitrary road driving image:
```powershell
Set-Location S:\TrafficSignRecognitionRoadSafetyMonitor
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"
& 'C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd' exec:java "-Dexec.mainClass=com.trafficsign.app.RoadSceneDemonstrationCli"
```
