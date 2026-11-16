# Models Directory

This directory stores neural network model artifacts.

## Subdirectories:
- `checkpoints/`: Serialized Java model weights, training checkpoints, and optimizer states saved during training runs.
- `exported/`: Optimized production model formats (e.g., `.onnx` files) for low-latency Java inference via ONNX Runtime.

All checkpoint and exported binaries are excluded from version control via `.gitignore`.
See `docs/MODEL_CARD.md` for architecture details.
