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


def _clip_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))
    return (x1, y1, x2, y2)


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


def apply_low_frequency_harmonization(
    original_composite: Image.Image,
    harmonized_guidance: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    delta_l_limit: float = 25.0,
    delta_ab_limit: float = 8.0,
    sigma_scale: float = 0.03,
) -> Image.Image:
    base_rgb = original_composite.convert("RGB")
    guidance_rgb = harmonized_guidance.convert("RGB")
    mask_l = foreground_mask.convert("L")

    if guidance_rgb.size != base_rgb.size:
        guidance_rgb = guidance_rgb.resize(base_rgb.size, Image.Resampling.LANCZOS)
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    base_np = np.array(base_rgb, dtype=np.float32)
    guidance_np = np.array(guidance_rgb, dtype=np.float32)
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0

    height, width = base_np.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return base_rgb

    base_region = base_np[y1:y2, x1:x2]
    guidance_region = guidance_np[y1:y2, x1:x2]
    alpha_region = mask_np[y1:y2, x1:x2]
    if alpha_region.max() <= 0.0:
        return base_rgb

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    sigma = max(12, int(round(max(bbox_w, bbox_h) * sigma_scale)))

    base_lab = cv2.cvtColor(base_region.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    guidance_lab = cv2.cvtColor(guidance_region.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    base_low = cv2.GaussianBlur(base_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    guidance_low = cv2.GaussianBlur(guidance_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    delta = guidance_low - base_low
    delta[:, :, 0] = np.clip(delta[:, :, 0], -float(delta_l_limit), float(delta_l_limit))
    delta[:, :, 1] = np.clip(delta[:, :, 1], -float(delta_ab_limit), float(delta_ab_limit))
    delta[:, :, 2] = np.clip(delta[:, :, 2], -float(delta_ab_limit), float(delta_ab_limit))

    alpha_soft = cv2.GaussianBlur(alpha_region, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha_soft = np.clip(alpha_soft, 0.0, 1.0)

    out_lab = np.clip(base_lab + delta * alpha_soft[..., None], 0.0, 255.0)
    out_rgb = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    result_np = base_np.copy()
    result_np[y1:y2, x1:x2] = out_rgb
    return Image.fromarray(np.clip(result_np, 0.0, 255.0).astype(np.uint8), mode="RGB")


def compute_detail_preservation_ratio(
    baseline_image: Image.Image,
    candidate_image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> float:
    baseline_rgb = baseline_image.convert("RGB")
    candidate_rgb = candidate_image.convert("RGB")
    mask_l = foreground_mask.convert("L")

    if candidate_rgb.size != baseline_rgb.size:
        candidate_rgb = candidate_rgb.resize(baseline_rgb.size, Image.Resampling.LANCZOS)
    if mask_l.size != baseline_rgb.size:
        mask_l = mask_l.resize(baseline_rgb.size, Image.Resampling.LANCZOS)

    baseline_np = np.array(baseline_rgb, dtype=np.float32)
    candidate_np = np.array(candidate_rgb, dtype=np.float32)
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0

    height, width = baseline_np.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return 1.0

    base_region = baseline_np[y1:y2, x1:x2]
    cand_region = candidate_np[y1:y2, x1:x2]
    alpha_region = mask_np[y1:y2, x1:x2]

    active_mask = (alpha_region > 0.1).astype(np.float32)
    if active_mask.sum() <= 1:
        return 1.0

    base_gray = cv2.cvtColor(base_region.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    cand_gray = cv2.cvtColor(cand_region.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_base = np.abs(cv2.Laplacian(base_gray, cv2.CV_32F, ksize=3))
    lap_cand = np.abs(cv2.Laplacian(cand_gray, cv2.CV_32F, ksize=3))

    denom = float(np.sum(active_mask)) + 1e-6
    energy_base = float(np.sum(lap_base * active_mask) / denom)
    energy_cand = float(np.sum(lap_cand * active_mask) / denom)
    return float(energy_cand / max(energy_base, 1e-6))


def apply_luminance_transfer_fallback(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> Image.Image:
    rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != rgb.size:
        mask_l = mask_l.resize(rgb.size, Image.Resampling.LANCZOS)

    rgb_np = np.array(rgb, dtype=np.float32)
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    height, width = rgb_np.shape[:2]

    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return rgb

    fg_mask = mask_np > 0.2

    pad_x = max(8, int(round((x2 - x1) * 0.10)))
    pad_y = max(8, int(round((y2 - y1) * 0.10)))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(width, x2 + pad_x)
    ey2 = min(height, y2 + pad_y)

    ring_mask = np.zeros((height, width), dtype=bool)
    ring_mask[ey1:ey2, ex1:ex2] = True
    ring_mask[y1:y2, x1:x2] = False
    ring_mask = np.logical_and(ring_mask, np.logical_not(fg_mask))
    if ring_mask.sum() < 50:
        ring_mask = np.logical_not(fg_mask)

    lab = cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    fg_l = l_channel[fg_mask]
    bg_l = l_channel[ring_mask]
    if fg_l.size == 0 or bg_l.size == 0:
        return rgb

    fg_mean = float(fg_l.mean())
    fg_std = float(fg_l.std() + 1e-6)
    bg_mean = float(bg_l.mean())
    bg_std = float(bg_l.std() + 1e-6)

    adjusted_l = ((l_channel - fg_mean) * (bg_std / fg_std)) + bg_mean
    adjusted_l = np.clip(adjusted_l, 0.0, 255.0)

    alpha_soft = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha_soft = np.clip(alpha_soft, 0.0, 1.0)
    lab[:, :, 0] = l_channel * (1.0 - alpha_soft) + adjusted_l * alpha_soft

    out_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out_rgb.astype(np.uint8), mode="RGB")


def blend_background_only(
    original_image: Image.Image,
    generated_image: Image.Image,
    foreground_mask: Image.Image,
    alpha: float,
) -> Image.Image:
    base_rgb = original_image.convert("RGB")
    generated_rgb = generated_image.convert("RGB")
    mask_l = foreground_mask.convert("L")

    if generated_rgb.size != base_rgb.size:
        generated_rgb = generated_rgb.resize(base_rgb.size, Image.Resampling.LANCZOS)
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    base_np = np.array(base_rgb, dtype=np.float32)
    generated_np = np.array(generated_rgb, dtype=np.float32)
    fg_alpha = np.array(mask_l, dtype=np.float32) / 255.0
    bg_alpha = (1.0 - fg_alpha) * max(0.0, min(1.0, float(alpha)))
    bg_alpha = bg_alpha[..., None]

    blended = (base_np * (1.0 - bg_alpha)) + (generated_np * bg_alpha)
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def detect_ground_plane(image: Image.Image) -> Image.Image:
    """
    MVP heuristic ground mask: bottom 40% of the image.

    libcom.ReflectionGenerationModel uses this mask to restrict reflection to ground.
    """

    w, h = image.size
    ground = np.zeros((h, w), dtype=np.uint8)
    ground[int(h * 0.60) :, :] = 255
    return Image.fromarray(ground, mode="L")
