"""
Image augmentation pipeline for traffic signs
mirroring com.trafficsign.data.augmentation.ImageAugmentor.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Optional, Union
import numpy as np
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    max_rotation_degrees: float
    max_translation_fraction: float
    min_scale: float
    max_scale: float
    brightness_factor_range: float
    contrast_factor_range: float
    seed: int

    @classmethod
    def moderate_default(cls, seed: int = 42) -> AugmentationConfig:
        return cls(
            enabled=True,
            max_rotation_degrees=10.0,
            max_translation_fraction=0.05,
            min_scale=0.90,
            max_scale=1.10,
            brightness_factor_range=0.15,
            contrast_factor_range=0.15,
            seed=seed
        )

    @classmethod
    def photometric_only(cls, seed: int = 42) -> AugmentationConfig:
        return cls(
            enabled=True,
            max_rotation_degrees=0.0,
            max_translation_fraction=0.0,
            min_scale=1.0,
            max_scale=1.0,
            brightness_factor_range=0.10,
            contrast_factor_range=0.10,
            seed=seed
        )

    @classmethod
    def disabled(cls) -> AugmentationConfig:
        return cls(
            enabled=False,
            max_rotation_degrees=0.0,
            max_translation_fraction=0.0,
            min_scale=1.0,
            max_scale=1.0,
            brightness_factor_range=0.0,
            contrast_factor_range=0.0,
            seed=0
        )


class ImageAugmentor:
    """
    Image augmentor applying photometric and geometric transformations.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig.moderate_default()
        self.rng = random.Random(self.config.seed)

    def augment(self, image: Image.Image) -> Image.Image:
        if not self.config.enabled:
            return image

        img = image.copy()

        # 1. Rotation
        if self.config.max_rotation_degrees > 0.0:
            angle = self.rng.uniform(-self.config.max_rotation_degrees, self.config.max_rotation_degrees)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        # 2. Scaling
        if self.config.min_scale != 1.0 or self.config.max_scale != 1.0:
            scale = self.rng.uniform(self.config.min_scale, self.config.max_scale)
            orig_w, orig_h = img.size
            new_w = max(4, int(orig_w * scale))
            new_h = max(4, int(orig_h * scale))
            scaled = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            img = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
            offset_x = (orig_w - new_w) // 2
            offset_y = (orig_h - new_h) // 2
            img.paste(scaled, (offset_x, offset_y))

        # 3. Brightness
        if self.config.brightness_factor_range > 0.0:
            b_factor = 1.0 + self.rng.uniform(-self.config.brightness_factor_range, self.config.brightness_factor_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(max(0.1, b_factor))

        # 4. Contrast
        if self.config.contrast_factor_range > 0.0:
            c_factor = 1.0 + self.rng.uniform(-self.config.contrast_factor_range, self.config.contrast_factor_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(max(0.1, c_factor))

        return img
