# Indian Traffic Sign Dataset Readiness & Pre-Flight Acquisition Plan

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Target Dataset:** `kannanwisen/Indian-Traffic-Sign-Classification` (Hugging Face)  
**Status:** Pre-flight complete. Research and validation finished. **AWAITING USER APPROVAL TO DOWNLOAD.**  
**Date:** October 19, 2026  

---

## 1. Compliance and Safeguards Declaration

In accordance with university course ethics and non-negotiable project guidelines:
- **No dataset files have been downloaded in this stage.**
- No external accounts, tokens, or licenses have been accepted without permission.
- The pre-flight verification establishes all necessary integrity checks, storage allocations, and destination paths prior to receiving download approval.

---

## 2. Target Dataset Profile

| Parameter | Specification |
| :--- | :--- |
| **Dataset Name** | Indian Traffic Sign Classification (`kannanwisen/Indian-Traffic-Sign-Classification`) |
| **Hosting Platform** | Hugging Face Dataset Hub |
| **Repository URL** | `https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification` |
| **Direct Archive URL** | `https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification/resolve/main/data/train.zip` (or parquet/image chunks) |
| **Licence** | **Creative Commons Attribution 4.0 International (CC-BY-4.0)** |
| **Attribution Requirement** | Must cite repository owner: Kannan Wisen / Hugging Face Community |
| **Image Count** | Approximately 5,726 to 5,735 images |
| **Number of Classes** | 85 standard Indian classes (IRC:67 aligned) |
| **Storage Footprint** | Compressed download: **~120 MB**; Extracted footprint: **~185 MB** |
| **Image Formats** | Real photographic JPEG / PNG files |
| **Proposed Destination Path** | `S:\TrafficSignRecognitionRoadSafetyMonitor\data\raw\IndianTrafficSigns\` |

---

## 3. Proposed Directory Structure After Extraction

```
data/raw/IndianTrafficSigns/
├── ALL_MOTOR_VEHICLE_PROHIBITED/
│   ├── img_001.jpg
│   └── ...
├── AXLE_LOAD_LIMIT/
│   ├── img_001.jpg
│   └── ...
├── BARRIER_AHEAD/
├── BULLOCK_PROHIBITED/
├── CATTLE/
├── COMPULSARY_AHEAD/
├── COMPULSARY_KEEP_LEFT/
├── COMPULSARY_SOUND_HORN/
├── GIVE_WAY/
├── HORN_PROHIBITED/
├── HUMP_OR_ROUGH_ROAD/
├── NO_ENTRY/
├── NO_PARKING/
├── OVERTAKING_PROHIBITED/
├── PEDESTRIAN_CROSSING/
├── SCHOOL_AHEAD/
├── SPEED_LIMIT_40/
├── SPEED_LIMIT_80/
├── STOP/
└── ... (up to 85 class directories)
```

---

## 4. Integrity and Extraction Safeguards

When acquisition is approved by the user, the acquisition workflow will execute with the following automated safety measures:

1. **Atomic Streaming Download:**
   - Files are downloaded with a `.tmp` extension (`IndianTrafficSigns.zip.tmp`).
   - Only renamed to `.zip` after full byte count and checksum validation.
   - Resumable download supported via HTTP Range headers.

2. **Zip-Slip Path Traversal Guard:**
   - The extraction method verifies that every zip entry's canonical destination strictly starts with the resolved target directory path:
   ```java
   Path destination = targetDir.resolve(entry.getName()).normalize();
   if (!destination.startsWith(targetDir)) {
       throw new SecurityException("Zip Slip detected: " + entry.getName());
   }
   ```

3. **Bit-for-Bit Duplicate & Corruption Inspection:**
   - `DatasetIntegrityChecker` scans all extracted images.
   - Computes SHA-256 hashes for every file to identify duplicate instances.
   - Verifies that zero-byte files are quarantined.

4. **Class Frequency and Leakage Validation:**
   - `ImageFolderDatasetLoader` maps subdirectories to `IndianTrafficSignTaxonomy`.
   - Generates `ClassFrequencyReport` flagging class imbalances or missing classes.
   - `GroupedDatasetSplitter` creates deterministic 70/15/15 partitions.
   - `DatasetIntegrityChecker.checkCrossSplitHashLeakage()` ensures zero image overlap between train, val, and test partitions.

---

## 5. Dry-Run Verification Script

The dry-run check script `scripts/verify_indian_dataset_dryrun.ps1` allows inspection of local storage and pre-flight connectivity without downloading files:

```powershell
# scripts/verify_indian_dataset_dryrun.ps1
$TargetDir = "S:\TrafficSignRecognitionRoadSafetyMonitor\data\raw\IndianTrafficSigns"
Write-Host "=== Indian Dataset Acquisition Pre-Flight ===" -ForegroundColor Cyan
Write-Host "Target Directory: $TargetDir"

# Check available disk space
$drive = Get-PSDrive -Name S
$freeSpaceMb = [math]::Round($drive.Free / 1MB, 2)
Write-Host "Available Disk Space on S: $freeSpaceMb MB"

if ($freeSpaceMb -lt 500) {
    Write-Warning "Insufficient space. At least 500 MB required for archive and extracted data."
} else {
    Write-Host "Disk space check PASSED (> 500 MB available)." -ForegroundColor Green
}

# Test network endpoint reachability
$url = "https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification"
try {
    $response = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 10 -UseBasicParsing
    Write-Host "Hugging Face endpoint reachability: OK (Status: $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Warning "Endpoint check warning: $($_.Exception.Message)"
}
```

---

## 6. Exact Files to be Downloaded Upon Approval

1. `data/raw/IndianTrafficSigns/dataset_archive.zip` (~120 MB)
   - Containing ~5,726 JPEG/PNG images organized in 85 class subdirectories.
2. `data/raw/IndianTrafficSigns/manifest.json`
   - Manifest documenting source provenance, commit hash, license citation, download timestamp, and SHA-256 archive checksum.

**No files will be downloaded until the user gives explicit confirmation.**
