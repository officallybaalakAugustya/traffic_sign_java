"""
Image preprocessing engine for traffic-sign recognition
mirroring com.trafficsign.data.ImagePreprocessor.
Handles format decoding, ROI cropping, bilinear resizing,
min-max scaling to [0.0, 1.0], and optional channel standardization.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Image preprocessing engine for traffic-sign recognition and detection.
    """

    DEFAULT_TARGET_WIDTH = 32
    DEFAULT_TARGET_HEIGHT = 32

    GTSRB_CHANNEL_MEANS = (0.340, 0.312, 0.321)
    GTSRB_CHANNEL_STDS = (0.272, 0.260, 0.266)

    def __init__(
        self,
        target_width: int = DEFAULT_TARGET_WIDTH,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        standardize: bool = False,
        means: Sequence[float] = GTSRB_CHANNEL_MEANS,
        stds: Sequence[float] = GTSRB_CHANNEL_STDS,
    ):
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target dimensions must be strictly positive")
        self.target_width = target_width
        self.target_height = target_height
        self.standardize = standardize
        self.channel_means = np.array(means, dtype=np.float32)
        self.channel_stds = np.array(stds, dtype=np.float32)

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        img = Image.open(p).convert("RGB")
        return img

    def crop_roi(
        self,
        image: Union[Image.Image, np.ndarray],
        roi_x1: int,
        roi_y1: int,
        roi_x2: int,
        roi_y2: int
    ) -> Union[Image.Image, np.ndarray]:
        if isinstance(image, Image.Image):
            w, h = image.size
            x1 = max(0, min(roi_x1, w - 1))
            y1 = max(0, min(roi_y1, h - 1))
            x2 = max(x1 + 1, min(roi_x2, w))
            y2 = max(y1 + 1, min(roi_y2, h))
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return image
            return image.crop((x1, y1, x2, y2))
        else:
            h, w = image.shape[:2]
            x1 = max(0, min(roi_x1, w - 1))
            y1 = max(0, min(roi_y1, h - 1))
            x2 = max(x1 + 1, min(roi_x2, w))
            y2 = max(y1 + 1, min(roi_y2, h))
            return image[y1:y2, x1:x2]

    def resize_bilinear(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if width <= 0 or height <= 0:
            raise ValueError("Resize dimensions must be positive")
        if image.size == (width, height):
            return image
        return image.resize((width, height), Image.Resampling.BILINEAR)

    def to_tensor_chw(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """
        Converts an image into a 3D float32 numpy array with shape [3, Height, Width] (CHW).
        """
        if isinstance(image, (str, Path)):
            image = self.load_image(image)

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            resized = self.resize_bilinear(image, self.target_width, self.target_height)
            arr = np.array(resized, dtype=np.float32) / 255.0  # [H, W, 3]
        else:
            # OpenCV or numpy array [H, W, 3]
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.shape[0] != self.target_height or arr.shape[1] != self.target_width:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray((arr * 255).astype(np.uint8))
                resized = self.resize_bilinear(pil_img, self.target_width, self.target_height)
                arr = np.array(resized, dtype=np.float32) / 255.0

        if self.standardize:
            arr = (arr - self.channel_means) / self.channel_stds

        # Transpose from HWC to CHW
        chw = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=np.float32)

    def to_tensor_hwc(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        chw = self.to_tensor_chw(image)
        hwc = np.transpose(chw, (1, 2, 0))
        return np.ascontiguousarray(hwc, dtype=np.float32)

    def process(self, path: Union[str, Path]) -> np.ndarray:
        return self.to_tensor_chw(path)
