from __future__ import annotations
"""
Pattern image generators producing polar-grid images (height = radial_bins / tracks,
width = angular_bins / theta). Values are grayscale in [0,1], where 0 is dark (more
transitions in silkscreen) and 1 is light.
"""
from typing import Literal
import numpy as np
from PIL import Image

PatternName = Literal["checker", "wedges", "bars_theta", "bars_radial"]


def generate_polar_pattern(
    name: PatternName,
    angular_bins: int,
    radial_bins: int,
    *,
    # wedges
    k: int = 12,
    duty: float = 0.5,
    # checker / bars periods (in bins)
    theta_period: int = 36,
    radial_period: int = 8,
) -> np.ndarray:
    W = int(max(1, angular_bins))
    H = int(max(1, radial_bins))
    duty = float(np.clip(duty, 0.01, 0.99))
    theta_period = max(1, int(theta_period))
    radial_period = max(1, int(radial_period))

    img = np.ones((H, W), dtype=np.float32)

    if name == "wedges":
        k = max(2, int(k))
        wedge_w = W / float(k)
        # Dark wedges occupy a fraction 'duty' of each wedge
        xs = np.arange(W)
        pos_in_wedge = xs % wedge_w
        mask_theta = pos_in_wedge < (duty * wedge_w)
        img[:, mask_theta] = 0.0
        return img

    if name == "bars_theta":
        xs = np.arange(W)
        pos = xs % theta_period
        mask = pos < (duty * theta_period)
        img[:, mask] = 0.0
        return img

    if name == "bars_radial":
        ys = np.arange(H)
        pos = ys % radial_period
        mask = pos < (duty * radial_period)
        img[mask, :] = 0.0
        return img

    # default: checker
    xs = (np.arange(W) // theta_period)
    ys = (np.arange(H) // radial_period)
    grid = (ys[:, None] + xs[None, :]) % 2
    img = grid.astype(np.float32)
    # 0→dark, 1→light already
    return img


def save_polar_png(arr01: np.ndarray, path: str) -> None:
    a = np.clip(arr01, 0.0, 1.0)
    img8 = (a * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode='L').save(path)
