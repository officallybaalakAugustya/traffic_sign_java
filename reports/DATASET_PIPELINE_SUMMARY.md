# Indian Traffic Sign & Road Safety Pipeline: Master Data Summary

**Generated At:** 2026-10-19T20:20:20.082840400Z

## 1. Indian Sign Classification Pipeline (IRC:67 85 Classes)
- Detailed report: [`CLASSIFICATION_DATASET_REPORT.md`](CLASSIFICATION_DATASET_REPORT.md)
- Machine-readable report: [`classification_validation_report.json`](classification_validation_report.json)

## 2. Indian Road Driving Detection Pipeline (Generic Signs in Road Scenes)
- Detailed report: [`TAR_INSPECTION_REPORT.md`](TAR_INSPECTION_REPORT.md)
- Machine-readable report: [`tar_inspection_report.json`](tar_inspection_report.json)

## 3. Separation of Responsibilities
- **Classification Target:** 85-class IRC:67 sign recognition trained on pure cropped signboards.
- **Detection Target:** Full-frame bounding box localization for `traffic sign` category.
- **Road Safety Monitor:** Decision-support alerts combining detection localization with fine-grained classification.
