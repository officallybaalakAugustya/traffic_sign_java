from pathlib import Path
import torch
from trafficsign.neural.models import HighResSignDetector, CompactSignDetector, IndianTrafficSignCNN


def test_models_forward_and_params():
    # Detector
    det = HighResSignDetector()
    dummy_img = torch.randn(1, 3, 128, 128)
    det_out = det(dummy_img)
    assert det_out.shape == (1, 5, 16, 16)

    # Classifier
    cls_model = IndianTrafficSignCNN(num_classes=85, input_resolution=48)
    dummy_crop = torch.randn(2, 3, 48, 48)
    cls_out = cls_model(dummy_crop)
    assert cls_out.shape == (2, 85)

    # Checkpoint loading verification if checkpoint file exists
    det_ckpt = Path("S:/TrafficSignRecognitionRoadSafetyMonitor/models/checkpoints/detector_task11_champion/sign_detector-0005.params")
    if det_ckpt.exists():
        det.load_djl_checkpoint(det_ckpt)
        assert float(det.head.weight.detach().norm()) > 0.0

    cls_ckpt = Path("S:/TrafficSignRecognitionRoadSafetyMonitor/models/checkpoints/task7_champion/exp_b_48x48-0008.params")
    if cls_ckpt.exists():
        cls_model.load_djl_checkpoint(cls_ckpt)
        assert float(cls_model.fc2.weight.detach().norm()) > 0.0
