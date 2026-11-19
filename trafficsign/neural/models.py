"""
PyTorch Deep Learning architectures mirroring com.trafficsign.neural.djl:
- HighResSignDetector
- CompactSignDetector
- IndianTrafficSignCNN
And the DJL binary `.params` parser and checkpoint loader.
"""

from __future__ import annotations
import io
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_djl_params(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse a Deep Java Library (DJL) binary .params checkpoint file into a dictionary of numpy arrays.
    """
    with open(file_path, "rb") as f:
        content = f.read()

    arrays: Dict[str, np.ndarray] = {}
    idx = 0
    while True:
        pos = content.find(b"NDAR", idx)
        if pos == -1:
            break
        p = pos + 4
        # Version bytes (e.g. \x00\x00\x00\x03\x01)
        p += 5
        # 2-byte name length
        nlen = struct.unpack(">H", content[p : p + 2])[0]
        p += 2
        name = content[p : p + nlen].decode("utf-8", errors="ignore")
        p += nlen
        # encoding (e.g. DENSE)
        elen = struct.unpack(">H", content[p : p + 2])[0]
        p += 2
        p += elen
        # dtype (e.g. FLOAT32)
        dlen = struct.unpack(">H", content[p : p + 2])[0]
        p += 2
        p += dlen
        # ndim (4-byte int)
        ndim = struct.unpack(">I", content[p : p + 4])[0]
        p += 4
        shape = [struct.unpack(">q", content[p + i * 8 : p + (i + 1) * 8])[0] for i in range(ndim)]
        p += ndim * 8
        # layout (layout_len * 2 bytes)
        layout_len = struct.unpack(">I", content[p : p + 4])[0]
        p += 4 + layout_len * 2

        # Byte order marker: '<' (0x3c) or '>' (0x3e)
        byte_order = chr(content[p])
        p += 1

        # 4-byte data length
        data_len = struct.unpack(">I", content[p : p + 4])[0]
        p += 4

        dtype = "<f4" if byte_order == "<" else ">f4"
        raw = content[p : p + data_len]
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if len(shape) > 0:
            arr = arr.reshape(shape)
        key = f"{name}_{len(arrays)}"
        arrays[key] = arr
        idx = p + data_len

    return arrays


class HighResSignDetector(nn.Module):
    """
    16x16 grid cell Traffic Sign Detector mirroring HighResSignDetectorBlock.
    Outputs: [B, 5, 16, 16] (objectness, x, y, w, h).
    """

    GRID_ROWS = 16
    GRID_COLS = 16
    CHANNELS_PER_CELL = 5

    def __init__(self):
        super().__init__()
        # Stage 1: [B, 3, 128, 128] -> [B, 32, 64, 64]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Stage 2: [B, 32, 64, 64] -> [B, 64, 32, 32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 3: [B, 64, 32, 32] -> [B, 128, 16, 16]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Stage 4: [B, 128, 16, 16] -> [B, 128, 16, 16]
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Head: [B, 128, 16, 16] -> [B, 5, 16, 16]
        self.head = nn.Conv2d(128, self.CHANNELS_PER_CELL, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Head
        x = self.head(x)
        return x

    def load_djl_checkpoint(self, params_path: Union[str, Path]) -> None:
        arrays = parse_djl_params(params_path)
        # Check mapping
        self.conv1.weight.data.copy_(torch.from_numpy(arrays["weight_0"]))
        self.conv1.bias.data.copy_(torch.from_numpy(arrays["bias_1"]))
        self.bn1.weight.data.copy_(torch.from_numpy(arrays["gamma_2"]))
        self.bn1.bias.data.copy_(torch.from_numpy(arrays["beta_3"]))
        self.bn1.running_mean.copy_(torch.from_numpy(arrays["runningMean_4"]))
        self.bn1.running_var.copy_(torch.from_numpy(arrays["runningVar_5"]))

        self.conv2.weight.data.copy_(torch.from_numpy(arrays["weight_6"]))
        self.conv2.bias.data.copy_(torch.from_numpy(arrays["bias_7"]))
        self.bn2.weight.data.copy_(torch.from_numpy(arrays["gamma_8"]))
        self.bn2.bias.data.copy_(torch.from_numpy(arrays["beta_9"]))
        self.bn2.running_mean.copy_(torch.from_numpy(arrays["runningMean_10"]))
        self.bn2.running_var.copy_(torch.from_numpy(arrays["runningVar_11"]))

        self.conv3.weight.data.copy_(torch.from_numpy(arrays["weight_12"]))
        self.conv3.bias.data.copy_(torch.from_numpy(arrays["bias_13"]))
        self.bn3.weight.data.copy_(torch.from_numpy(arrays["gamma_14"]))
        self.bn3.bias.data.copy_(torch.from_numpy(arrays["beta_15"]))
        self.bn3.running_mean.copy_(torch.from_numpy(arrays["runningMean_16"]))
        self.bn3.running_var.copy_(torch.from_numpy(arrays["runningVar_17"]))

        self.conv4.weight.data.copy_(torch.from_numpy(arrays["weight_18"]))
        self.conv4.bias.data.copy_(torch.from_numpy(arrays["bias_19"]))
        self.bn4.weight.data.copy_(torch.from_numpy(arrays["gamma_20"]))
        self.bn4.bias.data.copy_(torch.from_numpy(arrays["beta_21"]))
        self.bn4.running_mean.copy_(torch.from_numpy(arrays["runningMean_22"]))
        self.bn4.running_var.copy_(torch.from_numpy(arrays["runningVar_23"]))

        self.head.weight.data.copy_(torch.from_numpy(arrays["weight_24"]))
        self.head.bias.data.copy_(torch.from_numpy(arrays["bias_25"]))


class CompactSignDetector(nn.Module):
    """
    8x8 grid cell Traffic Sign Detector mirroring CompactSignDetectorBlock.
    Outputs: [B, 5, 8, 8].
    """

    GRID_ROWS = 8
    GRID_COLS = 8
    CHANNELS_PER_CELL = 5

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.head = nn.Conv2d(64, self.CHANNELS_PER_CELL, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.head(x)
        return x

    def load_djl_checkpoint(self, params_path: Union[str, Path]) -> None:
        arrays = parse_djl_params(params_path)
        self.conv1.weight.data.copy_(torch.from_numpy(arrays["weight_0"]))
        self.conv1.bias.data.copy_(torch.from_numpy(arrays["bias_1"]))
        self.bn1.weight.data.copy_(torch.from_numpy(arrays["gamma_2"]))
        self.bn1.bias.data.copy_(torch.from_numpy(arrays["beta_3"]))
        self.bn1.running_mean.copy_(torch.from_numpy(arrays["runningMean_4"]))
        self.bn1.running_var.copy_(torch.from_numpy(arrays["runningVar_5"]))

        self.conv2.weight.data.copy_(torch.from_numpy(arrays["weight_6"]))
        self.conv2.bias.data.copy_(torch.from_numpy(arrays["bias_7"]))
        self.bn2.weight.data.copy_(torch.from_numpy(arrays["gamma_8"]))
        self.bn2.bias.data.copy_(torch.from_numpy(arrays["beta_9"]))
        self.bn2.running_mean.copy_(torch.from_numpy(arrays["runningMean_10"]))
        self.bn2.running_var.copy_(torch.from_numpy(arrays["runningVar_11"]))

        self.conv3.weight.data.copy_(torch.from_numpy(arrays["weight_12"]))
        self.conv3.bias.data.copy_(torch.from_numpy(arrays["bias_13"]))
        self.bn3.weight.data.copy_(torch.from_numpy(arrays["gamma_14"]))
        self.bn3.bias.data.copy_(torch.from_numpy(arrays["beta_15"]))
        self.bn3.running_mean.copy_(torch.from_numpy(arrays["runningMean_16"]))
        self.bn3.running_var.copy_(torch.from_numpy(arrays["runningVar_17"]))

        self.conv4.weight.data.copy_(torch.from_numpy(arrays["weight_18"]))
        self.conv4.bias.data.copy_(torch.from_numpy(arrays["bias_19"]))
        self.bn4.weight.data.copy_(torch.from_numpy(arrays["gamma_20"]))
        self.bn4.bias.data.copy_(torch.from_numpy(arrays["beta_21"]))
        self.bn4.running_mean.copy_(torch.from_numpy(arrays["runningMean_22"]))
        self.bn4.running_var.copy_(torch.from_numpy(arrays["runningVar_23"]))

        self.head.weight.data.copy_(torch.from_numpy(arrays["weight_24"]))
        self.head.bias.data.copy_(torch.from_numpy(arrays["bias_25"]))


class IndianTrafficSignCNN(nn.Module):
    """
    85-Class Indian Traffic Sign Classifier mirroring IndianTrafficSignCnnBlock.
    Supports 48x48 resolution (flatten=4608) or 32x32 resolution (flatten=2048).
    """

    def __init__(self, num_classes: int = 85, input_resolution: int = 48):
        super().__init__()
        self.num_classes = num_classes
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        flat_dim = 128 * (input_resolution // 8) * (input_resolution // 8)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(flat_dim, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def load_djl_checkpoint(self, params_path: Union[str, Path]) -> None:
        arrays = parse_djl_params(params_path)
        self.conv1.weight.data.copy_(torch.from_numpy(arrays["weight_0"]))
        self.conv1.bias.data.copy_(torch.from_numpy(arrays["bias_1"]))
        self.bn1.weight.data.copy_(torch.from_numpy(arrays["gamma_2"]))
        self.bn1.bias.data.copy_(torch.from_numpy(arrays["beta_3"]))
        self.bn1.running_mean.copy_(torch.from_numpy(arrays["runningMean_4"]))
        self.bn1.running_var.copy_(torch.from_numpy(arrays["runningVar_5"]))

        self.conv2.weight.data.copy_(torch.from_numpy(arrays["weight_6"]))
        self.conv2.bias.data.copy_(torch.from_numpy(arrays["bias_7"]))
        self.bn2.weight.data.copy_(torch.from_numpy(arrays["gamma_8"]))
        self.bn2.bias.data.copy_(torch.from_numpy(arrays["beta_9"]))
        self.bn2.running_mean.copy_(torch.from_numpy(arrays["runningMean_10"]))
        self.bn2.running_var.copy_(torch.from_numpy(arrays["runningVar_11"]))

        self.conv3.weight.data.copy_(torch.from_numpy(arrays["weight_12"]))
        self.conv3.bias.data.copy_(torch.from_numpy(arrays["bias_13"]))
        self.bn3.weight.data.copy_(torch.from_numpy(arrays["gamma_14"]))
        self.bn3.bias.data.copy_(torch.from_numpy(arrays["beta_15"]))
        self.bn3.running_mean.copy_(torch.from_numpy(arrays["runningMean_16"]))
        self.bn3.running_var.copy_(torch.from_numpy(arrays["runningVar_17"]))

        # Linear weights in DJL: [fan_out, fan_in] or [fan_in, fan_out]
        # DJL Linear weights are stored as [fan_out, fan_in] matching PyTorch!
        fc1_w = arrays["weight_18"]
        if fc1_w.shape != self.fc1.weight.shape:
            fc1_w = fc1_w.T
        self.fc1.weight.data.copy_(torch.from_numpy(fc1_w))
        self.fc1.bias.data.copy_(torch.from_numpy(arrays["bias_19"]))

        fc2_w = arrays["weight_20"]
        if fc2_w.shape != self.fc2.weight.shape:
            fc2_w = fc2_w.T
        self.fc2.weight.data.copy_(torch.from_numpy(fc2_w))
        self.fc2.bias.data.copy_(torch.from_numpy(arrays["bias_21"]))
