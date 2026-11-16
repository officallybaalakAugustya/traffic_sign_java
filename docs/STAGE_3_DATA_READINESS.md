# Stage 3 Data Readiness & GTSRB Acquisition Audit

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** CSAI3027P — Deep Learning PBL  
**Date:** 2026-10-19  
**Status:** Pre-Flight Complete & Integration-Tested (74/74 Tests Passing). Download paused awaiting explicit user approval.  

---

## 1. Dataset Source, Licensing & Citation

| Field | Authoritative Specification |
| :--- | :--- |
| **Dataset Title** | German Traffic Sign Recognition Benchmark (GTSRB) |
| **Authors & Publishers** | Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel (Institut für Neuroinformatik, Ruhr-Universität Bochum & Department of Computer Science, University of Copenhagen) |
| **Primary Academic Citations** | 1. *Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition*, Neural Networks 32, pp. 323–332, 2012.<br>2. *The German Traffic Sign Recognition Benchmark: A Multi-class Classification Competition*, IJCNN 2011, pp. 1453–1460, IEEE Press. |
| **Official Archive Portal** | `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html` |
| **License Terms** | Freely available for non-commercial academic research and university educational benchmarking. |

---

## 2. Archive Metadata & Verified Byte Counts

Exact file sizes were verified against the official `published-files.json` manifest and confirmed via remote HTTP HEAD requests:

| Archive Name | Official Remote URL | Exact Published Size | Contents & Purpose |
| :--- | :--- | :--- | :--- |
| **`GTSRB_Final_Training_Images.zip`** | `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip` | **276,294,756 bytes** (~263.5 MB) | 39,209 images across 43 classes (`00000`–`00042`) in 30-frame track sequences, plus `GT-[class].csv` annotations. |
| **`GTSRB_Final_Test_Images.zip`** | `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip` | **88,978,620 bytes** (~84.8 MB) | 12,630 official unlabelled test images (`00000.ppm`–`12629.ppm`). |
| **`GTSRB_Final_Test_GT.zip`** | `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip` | **99,620 bytes** (~97.3 KB) | Ground-truth labels and bounding boxes (`GT-final_test.csv`) for the official test set. |

* **Total Compressed Download Size:** $276.3 + 84.8 + 0.1 \approx \mathbf{365.4\text{ MB}}$
* **Estimated Uncompressed Disk Footprint:** $\approx \mathbf{550\text{ MB}}$
* **Intended Download Destination:** `S:\TrafficSignRecognitionRoadSafetyMonitor\data\raw\GTSRB\`

---

## 3. Data Integrity & Security Protections Implemented

### 3.1 Zip Slip Path Traversal Protection
In [`DatasetAcquisitionManager.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/data/DatasetAcquisitionManager.java), archive extraction enforces strict canonical target directory prefix checking:
```java
Path resolvedPath = canonicalTarget.resolve(entry.getName()).normalize();
if (!resolvedPath.startsWith(canonicalTarget)) {
    throw new SecurityException("Malicious Zip Slip path traversal detected in entry: " + entry.getName());
}
```
*Unit-tested and verified with synthetic malicious archives in `DatasetAcquisitionManagerTest`.*

### 3.2 Incomplete Download Guard
Both the Java `DatasetAcquisitionManager` and the PowerShell script `scripts/download_dataset.ps1` stream incoming data to temporary files (`filename.zip.tmp`). Atomic renaming to `filename.zip` occurs **only** when the downloaded byte count exactly matches the authoritative published length.

### 3.3 Non-Destructive Dry-Run
Executing the acquisition script with `-DryRun` probes all three remote HTTP endpoints, checks headers (HTTP 200 OK, Content-Length match), tests disk space, and exits cleanly without downloading bytes.

---

## 4. GTSRB Directory Structure & Split Strategy

### 4.1 Uncompressed Directory Layout
```
data/raw/GTSRB/
└── Final_Training/
    └── Images/
        ├── 00000/
        │   ├── 00000_00000.ppm
        │   ├── 00000_00001.ppm
        │   ├── ...
        │   ├── 00000_00029.ppm   <-- 30 frames from single sign approach (Track 00000)
        │   ├── 00001_00000.ppm   <-- Track 00001
        │   └── GT-00000.csv
        ├── ...
        └── 00042/
```

### 4.2 Track Leakage Mitigation
- **Empirical Rule:** In GTSRB, filenames strictly follow `[TrackId]_[FrameId].ppm`. The 5-digit prefix corresponds to a single physical traffic sign captured across 30 consecutive video frames as the test car approaches.
- **Leakage Danger:** A random train/validation split would place frames 1–25 in training and 26–30 in validation, yielding near-100% artificial accuracy because the background, vehicle hood, and weather are identical.
- **Our Defense:** [`DatasetSplitter.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/main/java/com/trafficsign/data/DatasetSplitter.java) groups all 30 frames of each `trackId` into an atomic block, ensuring zero track overlap across Train, Validation, and Test partitions.
- **Normalization Isolation:** Channel mean and standard deviation are calculated **exclusively on the Train partition** and applied frozen to Validation and Test images.

---

## 5. Execution Device & GPU Status

- **Host Machine:** Intel Core i7-13550H (12 cores, 16 threads), 16 GB RAM, Windows 11 Home 64-bit.
- **GPU Profile:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, 90W TGP, Driver 595.97, CUDA 13.2 support via `nvidia-smi`).
- **DJL Runtime Status:**
  ```
  DEBUG ai.djl.util.cuda.CudaUtils -- No cudart library found in path.
  INFO  com.trafficsign.neural.djl.GpuVerifier -- Using CPU fallback: cpu()
  INFO  ai.djl.pytorch.engine.PtEngine -- Number of inter-op threads is 12
  INFO  ai.djl.pytorch.engine.PtEngine -- Number of intra-op threads is 8
  ```
- **Honest Diagnostic Finding:** PyTorch 2.2.2 CPU engine is fully operational with 8 intra-op threads and 12 inter-op threads. The GPU is recognized by Windows, but because `cudart64_*.dll` is not on the system PATH, DJL uses safe CPU fallback. We **do not claim** GPU execution until CUDA runtime libraries are registered.

---

## 6. Real-Data Pipeline Integration Test Results

Implemented in [`RealDataPipelineIntegrationTest.java`](file:///s:/TrafficSignRecognitionRoadSafetyMonitor/src/test/java/com/trafficsign/data/RealDataPipelineIntegrationTest.java):
1. **PPM Decoding:** Pure Java P6 binary reader decodes raw RGB samples.
2. **ROI Cropping & Bilinear Resizing:** Crops annotated bounding box and smoothly resizes to $32 \times 32 \times 3$.
3. **Channel Standardization:** Normalizes to zero-mean unit-variance.
4. **Track Isolation:** `DatasetSplitter` partitions tracks with zero leakage verified by `LeakageDetector.inspect()`.
5. **CNN Forward Pass:** Batch forwarded through `GtsrbCnnBlock` producing logits shape $[B, 43]$ and finite Softmax Cross-Entropy loss ($L > 0$).

---

## 7. Automated Test Suite Status

```
Results: Tests run: 74, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```
All 67 baseline tests + 7 new acquisition and integration tests pass cleanly.

---

## 8. Proposed Acquisition Execution Command

To download and extract the dataset, the user can run:

```powershell
# Pre-flight check (Dry-run mode, 0 bytes downloaded)
powershell -ExecutionPolicy Bypass -File "S:\TrafficSignRecognitionRoadSafetyMonitor\scripts\download_dataset.ps1" -DryRun

# Full download execution (Requires ~365 MB download / ~550 MB free disk space)
powershell -ExecutionPolicy Bypass -File "S:\TrafficSignRecognitionRoadSafetyMonitor\scripts\download_dataset.ps1"
```

*Acquisition has stopped at this boundary to request explicit user approval.*
