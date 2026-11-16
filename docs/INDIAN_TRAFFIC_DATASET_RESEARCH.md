# Research and Comparative Evaluation of Indian Traffic Sign Datasets

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Course:** CSAI3027P — Deep Learning PBL  
**Target:** Java 21 LTS + Apache Maven  
**Date:** October 19, 2026  

---

## 1. Research Scope and Methodology

Traffic sign recognition in India presents distinct challenges that foreign benchmarks such as GTSRB, TT100K, or Belgian TS do not address:
1. **IRC:67 Sign Standards:** Indian Roads Congress specifies unique sign geometries, pictograms, and bilingual/trilingual texts (e.g. speed bumps, cattle crossings, mandatory horn honking at ghat roads, horn prohibited zones).
2. **Extreme Environmental Variations:** Tropical monsoons, high-intensity sunlight causing glare/reflection, dust accumulation, heavy particulate matter/fog, nocturnal driving under non-standard street illumination.
3. **Severe Sign Degradation:** Physical weathering, rust, fading, occlusion by roadside foliage, billboards, overhangs, and commercial banners.
4. **Chaotic Background Clutter:** Dense heterogeneous traffic (autorickshaws, two-wheelers, trucks, pedestrians, stray animals) competing in the background.

This investigation evaluates publicly accessible, verified datasets containing genuine Indian road imagery to select the primary and supplementary datasets for our Java-only deep learning pipeline.

---

## 2. Evaluation of Candidate Datasets

### Candidate 1: Indian Traffic Sign Classification Dataset (Kannan Wisen / Hugging Face) — **RECOMMENDED PRIMARY**

- **Canonical URL:** `https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification`
- **Imagery Source:** Real-world photographic images of traffic signs captured across Indian roads (urban streets, state highways, national highways).
- **Dataset Scale:** Approximately **5,726 to 5,735 images** across **85 classes**.
- **Task Scope:** Multi-class Image Classification.
- **Annotation & Layout:** Standard `ImageFolder` directory layout where each subdirectory corresponds to a normalized sign class name (e.g., `SPEED_LIMIT_40`, `HORN_PROHIBITED`, `CATTLE`, `HUMP_OR_ROUGH_ROAD`).
- **Indian-Specific Categories:** Fully aligned with Indian Roads Congress (IRC:67) standards across Mandatory (Regulatory), Cautionary (Warning), and Informatory signs. Includes Indian specific signs: `CATTLE`, `HUMP_OR_ROUGH_ROAD` (speed breaker), `BARRIER_AHEAD`, `FALLING_ROCKS`, `GAP_IN_MEDIAN`, `COMPULSARY_SOUND_HORN`.
- **License & Legal Terms:** **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Explicitly allows academic, educational, and commercial research with standard attribution.
- **Storage Footprint:** ~120 MB compressed, ~180 MB uncompressed. Highly suitable for rapid prototyping and local disk constraints.
- **Known Quality Limitations:** Natural class imbalance (common signs like `STOP` and `SPEED_LIMIT` have higher representation than rare signs like `QUAY_SIDE_OR_RIVER_BANK`). Minor resolution variances across captures.
- **Java 21 Usability:** **Excellent**. Standard image formats (JPEG/PNG) decoded cleanly via Java's native `javax.imageio.ImageIO`. No custom C/C++ decoders, Python packages, or OpenCV bindings required.

---

### Candidate 2: Indian Traffic Signs Prediction (85 Classes) (Sarang Dilip Jodh / Darshan PK / Kaggle)

- **Canonical URL:** `https://www.kaggle.com/datasets/sarangdilipjodh/indian-traffic-signs-prediction85-classes`
- **Imagery Source:** Real road sign photographs collected in Indian urban/rural settings.
- **Dataset Scale:** ~4,438 total images (4,031 train, 407 validation) across 85 classes.
- **Task Scope:** Image Classification.
- **Annotation Format:** Directory-based class organization.
- **Indian-Specific Categories:** Same 85-class taxonomy as Candidate 1.
- **License & Legal Terms:** Kaggle open-access / community research terms.
- **Storage Footprint:** ~150 MB.
- **Known Limitations:** Pre-split into train/val without documented track or capture session grouping, creating potential visual leakage between identical physical signposts.
- **Java 21 Usability:** Good, but downloading requires Kaggle credentials/CLI authentication, making automated headless acquisition more cumbersome than Hugging Face direct endpoints.

---

### Candidate 3: Datacluster Labs Indian Traffic Sign Image Dataset (Kaggle / GitHub) — **RECOMMENDED SUPPLEMENTARY FOR DETECTION**

- **Canonical URL:** `https://www.kaggle.com/datasets/datacluster/indian-traffic-sign-image-dataset` / `https://github.com/datacluster-labs/Indian-Traffic-Sign-Image-Dataset`
- **Imagery Source:** High-definition real-world driving captures in over 20 Indian cities, covering daytime, twilight, nighttime, and adverse weather.
- **Dataset Scale:** ~4,000+ full-scene road images with thousands of annotated traffic sign bounding boxes.
- **Task Scope:** Object Detection and Localization (Full scene $\to$ Sign location + Class).
- **Annotation Format:** Bounding boxes in YOLO (`.txt`), PASCAL VOC (`.xml`), and COCO (`.json`) formats.
- **Indian-Specific Categories:** Covers mandatory, warning, and informational signs under challenging Indian lighting and perspective distortions.
- **License & Legal Terms:** Non-commercial educational / student research access on public tier. Commercial deployment requires enterprise licensing from Datacluster Labs.
- **Storage Footprint:** ~1.2 GB.
- **Known Limitations:** Full scenes require localization before classification. Bounding box XML/JSON parsing is necessary in Java.
- **Java 21 Usability:** Moderate. Requires XML (DOM/SAX) or JSON (Jackson) bounding-box parsing in pure Java, both of which are readily supported without Python.

---

### Candidate 4: India Driving Dataset (IDD) & IDD-MTSVD (AutoNue / CVIT, IIIT Hyderabad) — **BENCHMARK REFERENCE**

- **Canonical URL:** `https://idd.insaan.iiit.ac.in/` (INSAAN Portal)
- **Paper Citation:** Varma et al., *"IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments"*, WACV 2019; and Reddy et al., *"Missing Traffic Signs Video Dataset (IDD-MTSVD)"*.
- **Imagery Source:** Over 1,590 real dashcam driving videos recorded on Indian roads (Hyderabad, Bangalore, national highways) in unstructured traffic.
- **Dataset Scale:** 10,000+ unique track IDs, 70+ sign categories, dense frame-level context annotations.
- **Task Scope:** Context-driven missing traffic sign detection and full autonomous driving scene segmentation.
- **Annotation Format:** Sequence XML annotations, polygonal masks, video bounding boxes.
- **License & Legal Terms:** Strict Academic Research Agreement. **Mandatory institutional email registration and individual approval via the IIIT-H portal**. Direct automated unauthenticated scraping or headless downloading is strictly prohibited by terms of service.
- **Storage Footprint:** > 40 GB.
- **Known Limitations:** Extremely large footprint; requires video frame extraction and institutional credential management. Cannot be acquired autonomously in a classroom environment without manual user credential approval.
- **Java 21 Usability:** High complexity for an individual student semester PBL, but provides authoritative scientific citations for our Indian road safety monitor design.

---

### Candidate 5: Mapillary Traffic Sign Dataset (MTSD) — Indian Subset Analysis

- **Canonical URL:** `https://www.mapillary.com/dataset/trafficsign`
- **Imagery Source:** Street-level imagery crowdsourced globally.
- **Indian Subset:** While MTSD contains over 100,000 images globally across 300+ classes, the Indian subset is sparse (< 3% of the total dataset), geographically concentrated, and not pre-filtered into an independent Indian distribution.
- **License:** Non-commercial research license (CC BY-NC-SA 4.0 for academic use). Full dataset is > 200 GB.
- **Usability:** Impractical to filter gigabytes of global data just to extract a few hundred Indian signs when dedicated Indian datasets exist.

---

## 3. Comparative Summary Matrix

| Metric / Feature | Candidate 1: Kannan Wisen (Hugging Face) | Candidate 2: Sarang Jodh (Kaggle) | Candidate 3: Datacluster Labs | Candidate 4: IDD-MTSVD (IIIT-H) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Task** | Sign Classification | Sign Classification | Object Detection | Context / Missing Signs |
| **Images** | ~5,726 | ~4,438 | ~4,000+ | 1,590 Videos |
| **Classes** | 85 (IRC:67) | 85 (IRC:67) | ~40-50 | 70+ |
| **Authentic Indian Data** | Yes (100%) | Yes (100%) | Yes (100%) | Yes (100%) |
| **License** | **CC-BY-4.0 (Open)** | Community Open | Non-commercial | Academic Agreement |
| **Storage Size** | **~120 MB** | ~150 MB | ~1.2 GB | > 40 GB |
| **Java Compatibility** | **Native ImageIO** | Native ImageIO | Java XML/JSON parser | Video decoding needed |
| **Headless Acquisition** | **Supported** | Needs Kaggle Token | Needs Kaggle/Token | Restricted behind portal |
| **Status** | **RECOMMENDED PRIMARY** | Secondary Backup | **RECOMMENDED SUPPLEMENTARY** | Benchmark Citation |

---

## 4. Explicit Recommendation and Decision Rationale

### Primary Dataset: `kannanwisen/Indian-Traffic-Sign-Classification` (Hugging Face)
1. **Legal & Compliance:** Explicit **CC-BY-4.0** license guarantees lawful university research and publication without proprietary entanglements.
2. **Alignment with Indian Standards:** The 85 classes map directly to the **Indian Roads Congress (IRC:67)** classification system, including Indian-specific signs (Cattle, Speed Breakers, Horn Prohibitions, Median Gaps).
3. **Pure Java Suitability:** Standard image folder structure allows pure Java 21 decoding via `javax.imageio.ImageIO`, fully honoring the zero-Python requirement.
4. **Feasibility:** ~120 MB compressed footprint guarantees reproducible acquisition and rapid training iterations on student hardware (RTX 4050 6GB / Core i7-13550H).

### Supplementary Dataset: `Datacluster Labs Indian Traffic Sign Dataset`
- Retained as the target supplementary dataset for future object detection / localization evaluation (evaluating bounding boxes in Indian road scenes).

### Theoretical & Safety Reference: `IDD-MTSVD` (IIIT Hyderabad)
- Cited in our research framework for Indian driving risk modeling and missing traffic sign contextual monitoring.
