from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from exceptions import InvalidInputError


def get_tight_bbox_from_mask(mask: Image.Image) -> Tuple[int, int, int, int]:
    """
    Tightest (x1,y1,x2,y2) bbox around all non-zero pixels of an "L" mask.

    ControlCom requires the foreground crop to fill edge-to-edge, so this bbox must
    be tight (with a tiny padding to avoid clipping 1px details).
    """

    mask_np = np.array(mask.convert("L"))
    rows = np.any(mask_np > 10, axis=1)
    cols = np.any(mask_np > 10, axis=0)
    if not rows.any() or not cols.any():
        raise InvalidInputError("car", "Mask is empty — segmentation found no foreground.")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = mask_np.shape

    return (max(0, cmin - 2), max(0, rmin - 2), min(w, cmax + 3), min(h, rmax + 3))


def restore_high_freq_details(
    original_composite: Image.Image,
    harmonized: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    blend_alpha: float = 0.25,
) -> Image.Image:
    original_rgb = original_composite.convert("RGB")
    harmonized_rgb = harmonized.convert("RGB")
    if harmonized_rgb.size != original_rgb.size:
        harmonized_rgb = harmonized_rgb.resize(original_rgb.size, Image.Resampling.LANCZOS)

    orig_np = np.array(original_rgb).astype(np.float32)
    harm_np = np.array(harmonized_rgb).astype(np.float32)

    height, width = orig_np.shape[:2]
    x1, y1, x2, y2 = foreground_bbox
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return harmonized_rgb

    orig_region = orig_np[y1:y2, x1:x2]
    if orig_region.size == 0:
        return harmonized_rgb

    orig_gray = cv2.cvtColor(orig_region, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(orig_gray, (15, 15), 0)
    hf = orig_gray - blurred

    result_np = harm_np.copy()
    target_h = max(1, y2 - y1)
    target_w = max(1, x2 - x1)
    if hf.shape != (target_h, target_w):
        hf = cv2.resize(hf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    for c in range(3):
        result_np[y1:y2, x1:x2, c] = np.clip(
            harm_np[y1:y2, x1:x2, c] + hf * blend_alpha, 0, 255
        )

    return Image.fromarray(result_np.astype(np.uint8), mode="RGB")


def detect_ground_plane(image: Image.Image) -> Image.Image:
    """
    MVP heuristic ground mask: bottom 40% of the image.

    libcom.ReflectionGenerationModel uses this mask to restrict reflection to ground.
    """

    w, h = image.size
    ground = np.zeros((h, w), dtype=np.uint8)
    ground[int(h * 0.60) :, :] = 255
    return Image.fromarray(ground, mode="L")


def paste_mask_into_background(
    bg_size: Tuple[int, int],
    placement_bbox: Tuple[int, int, int, int],
    mask_crop_resized: Image.Image,
) -> Image.Image:
    x1, y1, x2, y2 = placement_bbox
    bg_w, bg_h = bg_size
    if x1 < 0 or y1 < 0 or x2 > bg_w or y2 > bg_h:
        raise InvalidInputError("background", "Placement bbox is out of bounds.")

    canvas = Image.new("L", bg_size, 0)
    canvas.paste(mask_crop_resized.convert("L"), (x1, y1))
    return canvas
