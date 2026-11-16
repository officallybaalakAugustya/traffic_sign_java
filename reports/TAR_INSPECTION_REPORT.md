# Indian Road Driving TAR Archives Inspection Report

**Source Directory:** `data\raw\indian-road-driving-sample`

## Archive Processing Summary

| Metric | Observed Value |
| :--- | :---: |
| **Archives Inspected** | 10 |
| **Total Frame Stems** | 10000 |
| **Matched JPG + JSON Pairs** | 10000 |
| **JPG Only (Missing Annotation)** | 0 |
| **JSON Only (Missing Image Frame)** | 0 |
| **PNG Files Ignored** | 9958 (Semantic purpose unverified; excluded) |
| **Total Annotated Objects** | 55589 |
| **Generic Traffic Sign Annotations** | 1032 |

## Detection vs Classification Distinction Notice

> [!IMPORTANT]
> The TAR archive annotations contain generic `traffic sign` bounding boxes ($x_1, y_1, x_2, y_2$). These boxes indicate spatial locations of traffic signs in driving scenes but **DO NOT** provide fine-grained sign classification categories (such as Stop, Speed Limit 40, or Cattle). They must **NEVER** be conflated with the 85-class IRC:67 classification ground truth.

## Object Category Distribution Across Inspected Archives

| Category | Object Count | Percentage of Total |
| :--- | :---: | :---: |
| `animal` | 640 | 1.15% |
| `autorickshaw` | 2966 | 5.34% |
| `bicycle` | 1245 | 2.24% |
| `bus` | 838 | 1.51% |
| `car` | 26276 | 47.27% |
| `motorcycle` | 7438 | 13.38% |
| `person` | 4812 | 8.66% |
| `rider` | 6615 | 11.90% |
| `traffic light` | 471 | 0.85% |
| `traffic sign` | 1032 | 1.86% |
| `truck` | 1554 | 2.80% |
| `vehicle fallback` | 1702 | 3.06% |

## Data Quality & Error Log

- Zero annotation errors or coordinate anomalies detected across inspected frames.

