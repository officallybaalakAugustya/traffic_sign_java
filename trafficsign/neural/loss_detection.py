"""
Objectness Calibration and Detection Grid Losses in PyTorch
mirroring com.trafficsign.neural.djl.DetectionTrainer losses:
- DetectionGridLoss
- BalancedDetectionLoss
- CalibratedDetectionLoss
- FocalDetectionLoss
- BalancedCalibratedLoss
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionGridLoss(nn.Module):
    """
    Mean Squared Error detection grid loss.
    """

    def __init__(self, box_weight: float = 5.0):
        super().__init__()
        self.box_weight = box_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_conf = pred[:, 0:1, :, :]
        target_conf = target[:, 0:1, :, :]

        conf_loss = (pred_conf - target_conf) ** 2
        mask = target_conf

        pred_box = pred[:, 1:5, :, :]
        target_box = target[:, 1:5, :, :]
        box_loss = ((pred_box - target_box) ** 2) * mask

        return torch.mean(conf_loss + self.box_weight * box_loss)


class BalancedDetectionLoss(nn.Module):
    """
    Balanced Detection Loss dividing positive BCE by norm_pos and negative BCE by norm_neg:
    conf_loss = (pos_bce / norm_pos) + bg_weight * (neg_bce / norm_neg)
    box_loss  = (xy_loss / norm_pos) + (wh_loss / norm_pos)
    total     = conf_loss + box_weight * box_loss
    """

    def __init__(self, bg_weight: float = 0.5, box_weight: float = 5.0):
        super().__init__()
        self.bg_weight = bg_weight
        self.box_weight = box_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_conf = pred[:, 0:1, :, :]
        target_conf = target[:, 0:1, :, :]

        # Numerically stable BCE with logits
        # bce = relu(x) - x * z + log(1 + exp(-|x|))
        bce = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")

        pos_mask = target_conf
        neg_mask = 1.0 - pos_mask

        n_pos = float(torch.sum(pos_mask).item())
        norm_pos = max(1.0, n_pos)

        n_neg = float(torch.sum(neg_mask).item())
        norm_neg = max(1.0, n_neg)

        pos_conf_loss = torch.sum(bce * pos_mask) / norm_pos
        neg_conf_loss = torch.sum(bce * neg_mask) / norm_neg
        conf_loss = pos_conf_loss + self.bg_weight * neg_conf_loss

        pred_xy = torch.sigmoid(pred[:, 1:3, :, :])
        target_xy = target[:, 1:3, :, :]
        xy_loss = torch.sum(((pred_xy - target_xy) ** 2) * pos_mask) / norm_pos

        pred_wh = pred[:, 3:5, :, :]
        target_wh = target[:, 3:5, :, :]
        wh_loss = torch.sum(((pred_wh - target_wh) ** 2) * pos_mask) / norm_pos

        box_loss = xy_loss + wh_loss
        return conf_loss + self.box_weight * box_loss


class CalibratedDetectionLoss(nn.Module):
    """
    Calibrated Detection Loss where negative BCE is normalized by norm_pos.
    """

    def __init__(self, bg_weight: float = 0.25, box_weight: float = 8.0):
        super().__init__()
        self.bg_weight = bg_weight
        self.box_weight = box_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_conf = pred[:, 0:1, :, :]
        target_conf = target[:, 0:1, :, :]

        bce = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")

        pos_mask = target_conf
        neg_mask = 1.0 - pos_mask

        n_pos = float(torch.sum(pos_mask).item())
        norm_pos = max(1.0, n_pos)

        pos_conf_loss = torch.sum(bce * pos_mask) / norm_pos
        neg_conf_loss = torch.sum(bce * neg_mask) / norm_pos
        conf_loss = pos_conf_loss + self.bg_weight * neg_conf_loss

        pred_xy = torch.sigmoid(pred[:, 1:3, :, :])
        target_xy = target[:, 1:3, :, :]
        xy_loss = torch.sum(((pred_xy - target_xy) ** 2) * pos_mask) / norm_pos

        pred_wh = pred[:, 3:5, :, :]
        target_wh = target[:, 3:5, :, :]
        wh_loss = torch.sum(((pred_wh - target_wh) ** 2) * pos_mask) / norm_pos

        box_loss = xy_loss + wh_loss
        return conf_loss + self.box_weight * box_loss


class FocalDetectionLoss(nn.Module):
    """
    Lin et al. Focal Loss adapted for detection grid cell objectness.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, box_weight: float = 8.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_conf = pred[:, 0:1, :, :]
        target_conf = target[:, 0:1, :, :]

        p = torch.sigmoid(pred_conf)
        bce = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")

        pos_mask = target_conf
        neg_mask = 1.0 - pos_mask

        pos_modulator = (1.0 - p) ** self.gamma
        neg_modulator = p ** self.gamma

        n_pos = float(torch.sum(pos_mask).item())
        norm_pos = max(1.0, n_pos)

        pos_conf_loss = torch.sum(bce * pos_modulator * self.alpha * pos_mask) / norm_pos
        neg_conf_loss = torch.sum(bce * neg_modulator * (1.0 - self.alpha) * neg_mask) / norm_pos
        conf_loss = pos_conf_loss + neg_conf_loss

        pred_xy = torch.sigmoid(pred[:, 1:3, :, :])
        target_xy = target[:, 1:3, :, :]
        xy_loss = torch.sum(((pred_xy - target_xy) ** 2) * pos_mask) / norm_pos

        pred_wh = pred[:, 3:5, :, :]
        target_wh = target[:, 3:5, :, :]
        wh_loss = torch.sum(((pred_wh - target_wh) ** 2) * pos_mask) / norm_pos

        box_loss = xy_loss + wh_loss
        return conf_loss + self.box_weight * box_loss


class BalancedCalibratedLoss(nn.Module):
    """
    Task 15/16 Balanced Calibrated Loss with bg_weight=12.0 and box_weight=8.0.
    """

    def __init__(self, bg_weight: float = 12.0, box_weight: float = 8.0):
        super().__init__()
        self.bg_weight = bg_weight
        self.box_weight = box_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_conf = pred[:, 0:1, :, :]
        target_conf = target[:, 0:1, :, :]

        bce = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")

        pos_mask = target_conf
        neg_mask = 1.0 - pos_mask

        n_pos = float(torch.sum(pos_mask).item())
        norm_pos = max(1.0, n_pos)

        n_neg = float(torch.sum(neg_mask).item())
        norm_neg = max(1.0, n_neg)

        pos_conf_loss = torch.sum(bce * pos_mask) / norm_pos
        neg_conf_loss = torch.sum(bce * neg_mask) / norm_neg
        conf_loss = pos_conf_loss + self.bg_weight * neg_conf_loss

        pred_xy = torch.sigmoid(pred[:, 1:3, :, :])
        target_xy = target[:, 1:3, :, :]
        xy_loss = torch.sum(((pred_xy - target_xy) ** 2) * pos_mask) / norm_pos

        pred_wh = pred[:, 3:5, :, :]
        target_wh = target[:, 3:5, :, :]
        wh_loss = torch.sum(((pred_wh - target_wh) ** 2) * pos_mask) / norm_pos

        box_loss = xy_loss + wh_loss
        return conf_loss + self.box_weight * box_loss
