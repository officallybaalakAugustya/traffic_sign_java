# Data Directory

This directory stores datasets across different stages of ingestion and processing.

## Subdirectories:
- `raw/`: Unmodified raw archives and image files (e.g., GTSRB PPM files). Excluded from Git.
- `processed/`: Standardized, preprocessed images (e.g., resized $32 \times 32$ PNGs or binary tensors). Excluded from Git.
- `splits/`: Deterministic JSON manifests specifying exact train, validation, and test splits with zero track leakage. Manifests are versioned in Git.

See `docs/DATASET_CARD.md` for complete dataset metadata, licensing, and splitting rules.
