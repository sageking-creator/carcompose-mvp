from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from exceptions import InvalidInputError


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


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


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    scaled = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - (2.0 * scaled))


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    rgb_linear = np.clip(rgb_linear.astype(np.float32), 0.0, 1.0)
    return np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        (1.055 * np.power(rgb_linear, 1.0 / 2.4)) - 0.055,
    )


def resize_rgba_premultiplied(
    rgba: Image.Image,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4,
) -> tuple[Image.Image, Image.Image]:
    target_w = max(1, int(size[0]))
    target_h = max(1, int(size[1]))

    rgba_np = np.array(rgba.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = np.clip(rgba_np[:, :, 3], 0.0, 1.0)
    rgb_linear = _srgb_to_linear(rgba_np[:, :, :3])

    premult = rgb_linear * alpha[..., None]
    premult_resized = cv2.resize(premult, (target_w, target_h), interpolation=interpolation)
    alpha_resized = cv2.resize(alpha, (target_w, target_h), interpolation=interpolation)
    alpha_resized = np.clip(alpha_resized, 0.0, 1.0).astype(np.float32)

    alpha_safe = np.maximum(alpha_resized, 1e-4)
    rgb_linear_resized = premult_resized / alpha_safe[..., None]
    rgb_linear_resized[alpha_resized <= 1e-3] = 0.0
    rgb_linear_resized = np.clip(rgb_linear_resized, 0.0, 1.0)

    rgb_srgb = _linear_to_srgb(rgb_linear_resized)
    rgb_u8 = np.clip(rgb_srgb * 255.0, 0.0, 255.0).astype(np.uint8)
    alpha_u8 = np.clip(alpha_resized * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(rgb_u8, mode="RGB"), Image.fromarray(alpha_u8, mode="L")


def get_tight_bbox_from_mask(mask: Image.Image, min_area_ratio: float = 0.005) -> Tuple[int, int, int, int]:
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    # Use a threshold consistent with alpha>=0.5 (128/255 ≈ 0.502) to avoid including
    # low-alpha halos in the bbox.
    hard = mask_np >= 128
    area_ratio = float(hard.mean())
    if area_ratio < float(min_area_ratio):
        raise InvalidInputError(
            "car",
            f"Foreground mask area is too small ({area_ratio:.4f} < {min_area_ratio:.4f}).",
        )

    rows = np.any(hard, axis=1)
    cols = np.any(hard, axis=0)
    if not rows.any() or not cols.any():
        raise InvalidInputError("car", "Mask is empty — segmentation found no foreground.")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    height, width = mask_np.shape
    return (
        max(0, int(cmin) - 2),
        max(0, int(rmin) - 2),
        min(width, int(cmax) + 3),
        min(height, int(rmax) + 3),
    )


def reharden_resized_alpha(
    mask_crop_resized: Image.Image,
    edge_px: int | None = None,
    studio_mode: bool = False,
) -> Image.Image:
    alpha = np.array(mask_crop_resized.convert("L"), dtype=np.float32) / 255.0
    hard = (alpha >= 0.5).astype(np.uint8)
    if hard.max() == 0:
        return mask_crop_resized.convert("L")

    long_edge = max(alpha.shape)
    band_px = int(edge_px if edge_px is not None else np.clip(round(long_edge * 0.0025), 2, 5))
    if studio_mode:
        band_px = min(band_px, 3)
    kernel_size = _odd_kernel_size(max(3, band_px * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    solid = cv2.erode(hard, kernel)
    dilated = cv2.dilate(hard, kernel)
    edge_band = np.logical_and(dilated > 0, solid == 0)

    alpha_soft = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, band_px / 2.0), sigmaY=max(1.0, band_px / 2.0))
    out = np.zeros_like(alpha, dtype=np.float32)
    out[solid > 0] = 1.0
    edge_inside = np.logical_and(edge_band, hard > 0)
    edge_outside = np.logical_and(edge_band, hard == 0)
    out[edge_inside] = 1.0

    if edge_outside.any():
        edge_vals = np.clip(alpha_soft[edge_outside], 0.0, 1.0)
        # Leak-safe AA: keep only a thin, low-alpha outside band so the car does not
        # “fog” the background. This is intentionally capped at <= 0.12 so the near-leak
        # p95 gate can pass and ControlCom can run when otherwise safe.
        edge_cap = 0.035 if studio_mode else 0.12
        edge_vals = np.minimum(edge_vals, 0.5)
        gamma = 2.2
        edge_vals = edge_cap * np.power(edge_vals / 0.5, gamma)
        out[edge_outside] = np.clip(edge_vals, 0.0, edge_cap)
    out_u8 = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out_u8, mode="L")


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


def compute_mask_artifact_checks(
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> dict[str, float]:
    mask_l = foreground_mask.convert("L")
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    hard = (mask_np >= 0.5).astype(np.uint8)
    mask_area_ratio = float(hard.mean())

    height, width = mask_np.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    region_alpha = mask_np[y1:y2, x1:x2]
    region_hard = hard[y1:y2, x1:x2]
    interior_kernel_size = _odd_kernel_size(max(7, int(round(max(bbox_w, bbox_h) * 0.06))))
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (interior_kernel_size, interior_kernel_size)
    )
    interior = cv2.erode(region_hard, interior_kernel)

    if interior.sum() > 0:
        interior_opaque_ratio = float(np.mean(region_alpha[interior > 0] >= 0.98))
    else:
        interior_opaque_ratio = float(np.mean(region_alpha >= 0.98)) if region_alpha.size else 0.0

    leak_kernel_size = _odd_kernel_size(max(9, int(round(max(bbox_w, bbox_h) * 0.02))))
    leak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (leak_kernel_size, leak_kernel_size))
    dilated = cv2.dilate(hard, leak_kernel)
    outside = dilated == 0
    outside_leak_mean_alpha = float(np.mean(mask_np[outside])) if outside.any() else 0.0

    near_kernel_size = _odd_kernel_size(max(9, int(round(max(bbox_w, bbox_h) * 0.04))))
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_kernel_size, near_kernel_size))
    outer_ring = np.logical_and(cv2.dilate(hard, near_kernel) > 0, hard == 0)
    if outer_ring.any():
        near_leak_mean_alpha = float(np.mean(mask_np[outer_ring]))
        near_leak_p95_alpha = float(np.percentile(mask_np[outer_ring], 95))
    else:
        near_leak_mean_alpha = 0.0
        near_leak_p95_alpha = 0.0

    return {
        "interiorOpaqueRatio": interior_opaque_ratio,
        "outsideLeakMeanAlpha": outside_leak_mean_alpha,
        "nearLeakMeanAlpha": near_leak_mean_alpha,
        "nearLeakP95Alpha": near_leak_p95_alpha,
        "maskAreaRatio": mask_area_ratio,
    }


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


def apply_multiband_harmonization(
    original_composite: Image.Image,
    harmonized_guidance: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> tuple[Image.Image, dict[str, float]]:
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
        return base_rgb, {"protectCoverageRatio": 0.0}

    base_region = base_np[y1:y2, x1:x2]
    guidance_region = guidance_np[y1:y2, x1:x2]
    alpha_region = mask_np[y1:y2, x1:x2]
    active = alpha_region > 0.2
    if active.sum() < 16:
        return base_rgb, {"protectCoverageRatio": 0.0}

    base_u8 = np.clip(base_region, 0.0, 255.0).astype(np.uint8)
    guidance_u8 = np.clip(guidance_region, 0.0, 255.0).astype(np.uint8)

    gray = cv2.cvtColor(base_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edge_energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge_threshold = float(np.percentile(edge_energy[active], 90)) if np.any(active) else 0.0
    protect = edge_energy > max(edge_threshold, 2.0)

    hsv = cv2.cvtColor(base_u8, cv2.COLOR_RGB2HSV)
    yellow_plate = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] >= 60)
    )
    yellow_plate = cv2.dilate(yellow_plate.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0

    protect = np.logical_or(protect, yellow_plate)
    protect = np.logical_and(protect, active)
    protect = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    base_low = cv2.bilateralFilter(base_u8, d=9, sigmaColor=50, sigmaSpace=9).astype(np.float32)
    guidance_low = cv2.bilateralFilter(guidance_u8, d=9, sigmaColor=50, sigmaSpace=9).astype(np.float32)
    detail = base_region - base_low
    candidate = np.clip(guidance_low + detail, 0.0, 255.0)
    base_lab = cv2.cvtColor(base_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    cand_lab = cv2.cvtColor(np.clip(candidate, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    delta_lab = cand_lab - base_lab
    delta_lab[:, :, 0] = np.clip(delta_lab[:, :, 0], -18.0, 18.0)
    delta_lab[:, :, 1] = np.clip(delta_lab[:, :, 1], -7.0, 7.0)
    delta_lab[:, :, 2] = np.clip(delta_lab[:, :, 2], -7.0, 7.0)
    candidate = cv2.cvtColor(np.clip(base_lab + delta_lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(
        np.float32
    )

    alpha_soft = cv2.GaussianBlur(alpha_region, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha_soft = np.clip(alpha_soft, 0.0, 1.0)
    weight = alpha_soft * (1.0 - protect.astype(np.float32))

    out_region = (base_region * (1.0 - weight[..., None])) + (candidate * weight[..., None])

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    smooth_threshold = float(np.percentile(texture[active], 45)) if np.any(active) else 0.0
    smooth_surfaces = np.logical_and(texture <= smooth_threshold, np.logical_not(protect))
    smooth_surfaces = np.logical_and(smooth_surfaces, active)

    smooth_weight = alpha_soft * smooth_surfaces.astype(np.float32) * 0.35
    out_region = (out_region * (1.0 - smooth_weight[..., None])) + (candidate * smooth_weight[..., None])

    out_np = base_np.copy()
    out_np[y1:y2, x1:x2] = np.clip(out_region, 0.0, 255.0)
    protect_coverage = float(protect.sum()) / float(max(active.sum(), 1))
    return Image.fromarray(np.clip(out_np, 0.0, 255.0).astype(np.uint8), mode="RGB"), {
        "protectCoverageRatio": protect_coverage,
    }


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


def defringe_to_target_background(
    composite_image: Image.Image,
    background_image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    edge_alpha_max: float = 0.85,
) -> Image.Image:
    composite_rgb = composite_image.convert("RGB")
    background_rgb = background_image.convert("RGB")
    mask_l = foreground_mask.convert("L")

    if background_rgb.size != composite_rgb.size:
        background_rgb = background_rgb.resize(composite_rgb.size, Image.Resampling.LANCZOS)
    if mask_l.size != composite_rgb.size:
        mask_l = mask_l.resize(composite_rgb.size, Image.Resampling.LANCZOS)

    composite_np = np.array(composite_rgb, dtype=np.float32)
    background_np = np.array(background_rgb, dtype=np.float32)
    alpha_np = np.array(mask_l, dtype=np.float32) / 255.0

    height, width = alpha_np.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return composite_rgb

    pad = max(4, int(round(max(x2 - x1, y2 - y1) * 0.03)))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)

    edge_band = np.logical_and(alpha_np > 0.005, alpha_np < max(0.05, float(edge_alpha_max)))
    scope = np.zeros_like(edge_band, dtype=bool)
    scope[ry1:ry2, rx1:rx2] = True
    edge_band = np.logical_and(edge_band, scope)
    if not edge_band.any():
        return composite_rgb

    hard_core_seed = np.logical_and(alpha_np >= 0.98, scope)
    hard_core = cv2.erode(hard_core_seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if hard_core.max() == 0:
        hard_core = hard_core_seed.astype(np.uint8)
    hard_core = hard_core.astype(bool)
    core_pixels = np.column_stack(np.where(hard_core))
    if core_pixels.size == 0:
        return composite_rgb

    # Nearest core color propagation: in a studio setup halos are usually edge RGB
    # contamination, not alpha leakage. Pull edge pixels toward a recomposited color
    # derived from nearby opaque foreground and target background.
    distance_input = np.where(hard_core, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    label_idx = np.clip(labels.astype(np.int32) - 1, 0, len(core_pixels) - 1)
    nearest_core = core_pixels[label_idx]
    core_rgb = composite_np[nearest_core[:, :, 0], nearest_core[:, :, 1]]

    alpha3 = alpha_np[..., None]
    target = (core_rgb * alpha3) + (background_np * (1.0 - alpha3))

    edge_upper = float(np.clip(edge_alpha_max, 0.18, 0.95))
    replace_weight = _smoothstep(0.03, edge_upper, 1.0 - alpha_np)
    replace_weight = np.clip(replace_weight, 0.0, 1.0) * edge_band.astype(np.float32)
    # Keep low-alpha boundary fully background-consistent.
    replace_weight[np.logical_and(edge_band, alpha_np <= 0.08)] = 1.0
    # Preserve identity inside strong-alpha edge pixels while still removing visible halos.
    mid_alpha = np.logical_and(alpha_np >= 0.70, alpha_np < 0.90)
    high_alpha = alpha_np >= 0.90
    replace_weight[mid_alpha] *= 0.80
    replace_weight[high_alpha] *= 0.35

    hard_fg = np.logical_and(alpha_np >= 0.55, scope)
    inner_ring = np.logical_and(
        hard_fg,
        cv2.erode(hard_fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) == 0,
    )
    if inner_ring.any():
        replace_weight[inner_ring] = np.maximum(replace_weight[inner_ring], 0.55)

    weight3 = replace_weight[..., None]
    output = (composite_np * (1.0 - weight3)) + (target * weight3)
    return Image.fromarray(np.clip(output, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _apply_contact_shadow_v1(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float,
) -> tuple[Image.Image, bool, Image.Image]:
    shadow_strength = max(0.0, min(1.0, float(strength)))
    if shadow_strength <= 0.0:
        return image.convert("RGB"), False, Image.new("L", image.size, 0)

    base_rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    base_np = np.array(base_rgb, dtype=np.float32) / 255.0
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    height, width = mask_np.shape

    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    if bbox_w <= 1 or bbox_h <= 1:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    region_mask = mask_np[y1:y2, x1:x2]
    if region_mask.max() <= 0.0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    seed_start = int(round(bbox_h * (1.0 - 0.22)))
    seed_start = max(0, min(seed_start, bbox_h - 1))
    seed = region_mask[seed_start:, :]
    if seed.size == 0 or seed.max() <= 0.0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    squashed_h = max(3, int(round(seed.shape[0] * 0.18)))
    seed_squashed = cv2.resize(seed, (bbox_w, squashed_h), interpolation=cv2.INTER_LINEAR)

    sigma = max(6, int(round(bbox_w * 0.012)))
    seed_blurred = cv2.GaussianBlur(seed_squashed, (0, 0), sigmaX=sigma, sigmaY=sigma)

    y_offset = max(2, int(round(bbox_h * 0.01)))
    top = int(y2 + y_offset)
    top = max(0, min(top, height - squashed_h))

    shadow_canvas = np.zeros((height, width), dtype=np.float32)
    shadow_canvas[top : top + squashed_h, x1:x2] = np.maximum(
        shadow_canvas[top : top + squashed_h, x1:x2], seed_blurred
    )

    hard_fg = (mask_np >= 0.5).astype(np.uint8)
    protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protect = cv2.dilate(hard_fg, protect_kernel, iterations=1)
    bg_only = (protect == 0).astype(np.float32)

    shadow_alpha = shadow_canvas * shadow_strength * bg_only
    output_np = base_np * (1.0 - shadow_alpha[..., None])
    output = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    shadow_mask = Image.fromarray(np.clip(shadow_alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return Image.fromarray(output, mode="RGB"), True, shadow_mask


def _apply_contact_shadow_v2(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float,
) -> tuple[Image.Image, bool, Image.Image]:
    shadow_strength = max(0.0, min(1.0, float(strength)))
    if shadow_strength <= 0.0:
        return image.convert("RGB"), False, Image.new("L", image.size, 0)

    base_rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    base_np = np.array(base_rgb, dtype=np.float32) / 255.0
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    height, width = mask_np.shape

    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    if bbox_w <= 1 or bbox_h <= 1:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    y_search2 = min(height, y2 + max(4, int(round(bbox_h * 0.15))))
    region_hard = (mask_np[y1:y_search2, x1:x2] >= 0.5).astype(np.uint8)
    if region_hard.max() == 0:
        region_hard = (mask_np[y1:y2, x1:x2] >= 0.5).astype(np.uint8)
    if region_hard.max() == 0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    columns = np.where(region_hard.sum(axis=0) > 0)[0]
    if columns.size == 0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    contour_seed = np.zeros((height, width), dtype=np.float32)
    contact_seed = np.zeros((height, width), dtype=np.float32)
    floor_mask = np.zeros((height, width), dtype=np.float32)
    y_offset = max(1, int(round(bbox_h * 0.012)))

    points = []
    for col in columns:
        rows = np.where(region_hard[:, col] > 0)[0]
        if rows.size == 0:
            continue
        bottom_local = int(rows.max())
        gx = x1 + int(col)
        gy = y1 + bottom_local
        points.append((gx, gy))
        floor_start = min(height, gy + y_offset)
        if floor_start < height:
            floor_mask[floor_start:, gx] = 1.0

    if len(points) < 3:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    thickness = max(1, int(round(bbox_h * 0.01)))
    contour_points = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(contour_seed, [contour_points], isClosed=False, color=1.0, thickness=thickness)

    if y_offset > 0:
        contour_seed = np.roll(contour_seed, shift=y_offset, axis=0)
        contour_seed[:y_offset, :] = 0.0

    # Reduce unrealistic “bumper shadow” by emphasizing wheel contact zones over the centerline.
    if bbox_w >= 64:
        local_bottoms: list[int] = []
        local_cols: list[int] = []
        for col in columns:
            rows = np.where(region_hard[:, col] > 0)[0]
            if rows.size == 0:
                continue
            local_cols.append(int(col))
            local_bottoms.append(int(rows.max()))

        x = np.linspace(0.0, 1.0, bbox_w, dtype=np.float32)
        sigma = 0.13
        left_mu = 0.28
        right_mu = 0.72
        if local_cols:
            cols_np = np.array(local_cols, dtype=np.int32)
            bottoms_np = np.array(local_bottoms, dtype=np.float32)
            norm_cols = cols_np.astype(np.float32) / max(1.0, float(bbox_w - 1))
            left_mask = np.logical_and(norm_cols >= 0.12, norm_cols <= 0.46)
            right_mask = np.logical_and(norm_cols >= 0.54, norm_cols <= 0.90)
            if not left_mask.any() or not right_mask.any():
                mid_col = float(np.median(cols_np))
                left_mask = cols_np <= mid_col
                right_mask = cols_np > mid_col
            if left_mask.any() and right_mask.any():
                left_cols = cols_np[left_mask]
                right_cols = cols_np[right_mask]
                left_bottoms = bottoms_np[left_mask]
                right_bottoms = bottoms_np[right_mask]

                def _pick_contact(cols: np.ndarray, bottoms: np.ndarray, target_mu: float) -> tuple[int, int]:
                    if cols.size == 1:
                        return int(cols[0]), int(bottoms[0])
                    norm = cols.astype(np.float32) / max(1.0, float(bbox_w - 1))
                    b_min = float(bottoms.min())
                    b_span = float(bottoms.max() - b_min)
                    if b_span > 1e-6:
                        b_norm = (bottoms - b_min) / b_span
                    else:
                        b_norm = np.zeros_like(bottoms, dtype=np.float32)
                    proximity = np.exp(-0.5 * ((norm - float(target_mu)) / 0.11) ** 2)
                    score = (0.70 * b_norm) + (0.30 * proximity)
                    idx = int(np.argmax(score))
                    return int(cols[idx]), int(bottoms[idx])

                left_contact_col, left_contact_bottom = _pick_contact(left_cols, left_bottoms, left_mu)
                right_contact_col, right_contact_bottom = _pick_contact(right_cols, right_bottoms, right_mu)

                left_mu = float(np.clip(left_contact_col / max(1, bbox_w - 1), 0.12, 0.48))
                right_mu = float(np.clip(right_contact_col / max(1, bbox_w - 1), 0.52, 0.90))
                sigma = 0.10

                cv2.circle(
                    contact_seed,
                    (x1 + left_contact_col, y1 + left_contact_bottom + y_offset),
                    radius=max(2, int(round(bbox_h * 0.012))),
                    color=1.0,
                    thickness=-1,
                )
                cv2.circle(
                    contact_seed,
                    (x1 + right_contact_col, y1 + right_contact_bottom + y_offset),
                    radius=max(2, int(round(bbox_h * 0.012))),
                    color=1.0,
                    thickness=-1,
                )

        w_left = np.exp(-0.5 * ((x - left_mu) / sigma) ** 2)
        w_right = np.exp(-0.5 * ((x - right_mu) / sigma) ** 2)
        weights = w_left + w_right
        weights = weights / max(float(weights.max()), 1e-6)
        contour_seed[:, x1:x2] = contour_seed[:, x1:x2] * weights[None, :]

    floor_mask = cv2.dilate(floor_mask, np.ones((1, 9), np.uint8), iterations=1)
    sigma_x = max(4.0, float(bbox_w) * 0.02)
    sigma_y = max(2.0, float(bbox_w) * 0.008)
    contour_blur = cv2.GaussianBlur(contour_seed, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    contact_blur = cv2.GaussianBlur(
        contact_seed,
        (0, 0),
        sigmaX=max(3.0, float(bbox_w) * 0.012),
        sigmaY=max(1.8, float(bbox_h) * 0.010),
    )

    hard_fg = (mask_np >= 0.5).astype(np.uint8)
    protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protect = cv2.dilate(hard_fg, protect_kernel, iterations=1)
    bg_only = (protect == 0).astype(np.float32)

    shadow_alpha = np.clip((contour_blur * 0.58) + (contact_blur * 0.95), 0.0, 1.0)
    shadow_alpha = shadow_alpha * floor_mask * shadow_strength * bg_only
    output_np = base_np * (1.0 - shadow_alpha[..., None])
    output = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    shadow_mask = Image.fromarray(np.clip(shadow_alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return Image.fromarray(output, mode="RGB"), True, shadow_mask


def _apply_contact_shadow_v3(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float,
) -> tuple[Image.Image, bool, Image.Image]:
    shadow_strength = max(0.0, min(1.0, float(strength)))
    if shadow_strength <= 0.0:
        return image.convert("RGB"), False, Image.new("L", image.size, 0)

    base_rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    base_np = np.array(base_rgb, dtype=np.float32) / 255.0
    alpha = np.array(mask_l, dtype=np.float32) / 255.0
    hard = (alpha >= 0.5).astype(np.uint8)
    if hard.max() == 0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    height, width = hard.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    region = hard[y1:y2, x1:x2]
    if region.max() == 0:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    # 1) Near-contact AO band from distance to lower silhouette edge.
    lower_band_h = max(6, int(round(bbox_h * 0.18)))
    region_lower = np.zeros_like(region, dtype=np.uint8)
    region_lower[max(0, region.shape[0] - lower_band_h) :, :] = region[max(0, region.shape[0] - lower_band_h) :, :]
    dist = cv2.distanceTransform(1 - region_lower, cv2.DIST_L2, 3)
    ao_local = np.exp(-(dist / max(2.0, bbox_w * 0.01)) ** 2).astype(np.float32)
    ao_local *= (region_lower == 0).astype(np.float32)

    ao_canvas = np.zeros((height, width), dtype=np.float32)
    ao_canvas[y1:y2, x1:x2] = ao_local

    # 2) Wider soft-floor shadow from projected contour.
    contour_seed = np.zeros((height, width), dtype=np.float32)
    cols = np.where(region.sum(axis=0) > 0)[0]
    points = []
    for col in cols:
        rows = np.where(region[:, col] > 0)[0]
        if rows.size == 0:
            continue
        points.append((x1 + int(col), y1 + int(rows.max())))
    if len(points) < 3:
        return base_rgb, False, Image.new("L", base_rgb.size, 0)

    poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(contour_seed, [poly], isClosed=False, color=1.0, thickness=max(1, int(round(bbox_h * 0.01))))
    contour_seed = np.roll(contour_seed, shift=max(2, int(round(bbox_h * 0.012))), axis=0)
    contour_seed[: max(2, int(round(bbox_h * 0.012))), :] = 0.0
    soft = cv2.GaussianBlur(
        contour_seed,
        (0, 0),
        sigmaX=max(5.0, float(bbox_w) * 0.025),
        sigmaY=max(3.0, float(bbox_w) * 0.010),
    )

    shadow_alpha = np.clip((ao_canvas * 0.65) + (soft * 0.55), 0.0, 1.0)

    # Keep shadow only on background.
    protect = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    shadow_alpha *= (protect == 0).astype(np.float32)
    shadow_alpha *= shadow_strength

    output_np = base_np * (1.0 - shadow_alpha[..., None])
    output = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    shadow_mask = Image.fromarray(np.clip(shadow_alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return Image.fromarray(output, mode="RGB"), True, shadow_mask


def apply_contact_shadow(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float = 0.32,
    mode: Literal["v1", "v2", "v3"] = "v3",
    return_shadow_mask: bool = False,
) -> tuple[Image.Image, bool] | tuple[Image.Image, bool, Image.Image]:
    if mode == "v1":
        output, applied, shadow_mask = _apply_contact_shadow_v1(image, foreground_mask, foreground_bbox, strength)
    elif mode == "v2":
        output, applied, shadow_mask = _apply_contact_shadow_v2(image, foreground_mask, foreground_bbox, strength)
    else:
        output, applied, shadow_mask = _apply_contact_shadow_v3(image, foreground_mask, foreground_bbox, strength)

    if return_shadow_mask:
        return output, applied, shadow_mask
    return output, applied


def is_studio_background(image: Image.Image) -> bool:
    rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    if height < 16 or width < 16:
        return False

    # Focus on the upper wall region. This avoids the floor rings/edges and the
    # bottom-right watermark that can distort “texture” heuristics.
    y1 = int(round(height * 0.05))
    y2 = int(round(height * 0.60))
    x1 = int(round(width * 0.08))
    x2 = int(round(width * 0.92))
    y2 = max(y1 + 8, min(y2, height))
    x2 = max(x1 + 8, min(x2, width))
    crop = rgb_u8[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_mean = float(np.mean(hsv[:, :, 1] / 255.0))
    value_std = float(np.std(hsv[:, :, 2] / 255.0))

    # High-pass residual is more robust than raw Laplacian percentiles for “smooth studios”.
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0)
    highpass = np.abs(gray - blur)
    texture_p95 = float(np.percentile(highpass, 95))

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    ab_std = float(np.mean(np.std(lab[:, :, 1:3], axis=(0, 1))))

    return sat_mean < 0.10 and value_std < 0.20 and texture_p95 < 10.0 and ab_std < 6.0


def estimate_turntable_alignment(image: Image.Image) -> dict[str, int] | None:
    """
    Best-effort heuristic for studio “turntable” backgrounds.

    Returns:
      {"centerX": int, "groundY": int, "spanW": int}
    or None when detection is unreliable.
    """
    rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    if height < 64 or width < 64:
        return None

    # Downscale for speed and robustness of morphology/contours.
    scale = 1.0
    max_width = 900
    if width > max_width:
        scale = max_width / float(width)
        new_w = max(64, int(round(width * scale)))
        new_h = max(64, int(round(height * scale)))
        rgb_u8 = cv2.resize(rgb_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        height, width = rgb_u8.shape[:2]

    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

    # Bottom ROI where the turntable rim lives (avoid top wall/ceiling).
    roi_y1 = int(round(height * 0.50))
    roi_y2 = int(round(height * 0.99))
    roi_x1 = int(round(width * 0.05))
    roi_x2 = int(round(width * 0.95))
    if roi_y2 <= roi_y1 + 16 or roi_x2 <= roi_x1 + 16:
        return None

    roi = gray_u8[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    max_kernel = _odd_kernel_size(max(5, min(roi_h, roi_w) - 1))
    if max_kernel < 5:
        return None

    # Search a few stable presets; pick the best ellipse candidate.
    kernel_sizes = [15, 21, 31]
    percentiles = [96.0, 97.0]

    best: tuple[float, float, float, float, float] | None = None
    best_score = -1.0

    for base_k in kernel_sizes:
        k_size = min(_odd_kernel_size(base_k), max_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
        blackhat = cv2.GaussianBlur(blackhat, (0, 0), sigmaX=1.2, sigmaY=1.2)

        for perc in percentiles:
            threshold = float(np.percentile(blackhat, perc))
            threshold = max(threshold, 6.0)
            binary = (blackhat >= threshold).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            for contour in contours:
                if len(contour) < 320:
                    continue
                try:
                    ellipse = cv2.fitEllipse(contour)
                except cv2.error:
                    continue
                (cx, cy), (a, b), angle = ellipse
                major = float(max(a, b))
                minor = float(min(a, b))

                cx_full = float(cx + roi_x1)
                cy_full = float(cy + roi_y1)

                # Basic sanity filters.
                if major < 0.45 * width or major > 1.50 * width:
                    continue
                if minor < 0.10 * height or minor > 0.80 * height:
                    continue
                aspect = major / max(minor, 1e-6)
                if aspect < 1.5 or aspect > 9.0:
                    continue
                if cx_full < 0.18 * width or cx_full > 0.82 * width:
                    continue
                if cy_full < 0.55 * height or cy_full > 0.98 * height:
                    continue

                score = major * minor
                if score > best_score:
                    best_score = score
                    best = (cx_full, cy_full, major, minor, float(angle))

    if best is None:
        return None

    cx_full, cy_full, major, minor, angle = best

    center_x = int(round(cx_full / scale))
    center_y = int(round(cy_full / scale))
    major_axis = int(round(major / scale))
    minor_axis = int(round(minor / scale))

    # Ground estimate: ellipse bottom is a better contact proxy for turntables.
    ground_y = int(round(center_y + (minor_axis / 2.0)))
    ground_y = max(0, min(ground_y, int(round(image.size[1])) - 1))

    return {
        "centerX": center_x,
        "groundY": ground_y,
        "spanW": major_axis,
        "centerY": center_y,
        "majorAxis": major_axis,
        "minorAxis": minor_axis,
        "angle": int(round(angle)),
    }


def render_placement_overlay(
    background: Image.Image,
    placement_bbox: Tuple[int, int, int, int],
    alignment: dict[str, int] | None,
    strict_bottom_local: int,
) -> Image.Image:
    overlay = background.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = placement_bbox
    draw.rectangle((x1, y1, x2, y2), outline=(32, 192, 255), width=3)

    strict_y = y1 + max(0, strict_bottom_local)
    draw.line((x1, strict_y, x2, strict_y), fill=(255, 64, 64), width=2)

    if alignment:
        center_x = int(alignment.get("centerX", 0))
        ground_y = int(alignment.get("groundY", 0))
        major_axis = int(alignment.get("majorAxis", alignment.get("spanW", 0)))
        minor_axis = int(alignment.get("minorAxis", 0))

        if major_axis > 0 and minor_axis > 0:
            cx = int(alignment.get("centerX", center_x))
            cy = int(alignment.get("centerY", ground_y))
            left = cx - (major_axis // 2)
            right = cx + (major_axis // 2)
            top = cy - (minor_axis // 2)
            bottom = cy + (minor_axis // 2)
            draw.ellipse((left, top, right, bottom), outline=(220, 220, 220), width=2)

        draw.line((0, ground_y, overlay.size[0], ground_y), fill=(255, 192, 0), width=2)
        draw.line((center_x, 0, center_x, overlay.size[1]), fill=(120, 200, 255), width=1)

    return overlay


def apply_glass_normalization(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    mode: Literal["off", "auto", "force"] = "off",
    return_candidate_mask: bool = False,
    candidate_mask_override: Image.Image | None = None,
    return_glass_render: bool = False,
) -> tuple[Image.Image, bool] | tuple[Image.Image, bool, Image.Image] | tuple[Image.Image, bool, Image.Image, Image.Image]:
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"off", "auto", "force"}:
        normalized_mode = "off"
    if normalized_mode == "off":
        result = image.convert("RGB")
        candidate_mask = Image.new("L", result.size, 0)
        glass_render = Image.new("RGB", result.size, (0, 0, 0))
        if return_candidate_mask and return_glass_render:
            return result, False, candidate_mask, glass_render
        if return_candidate_mask:
            return result, False, candidate_mask
        if return_glass_render:
            return result, False, glass_render
        return result, False

    base_rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    rgb_u8 = np.array(base_rgb, dtype=np.uint8)
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    height, width = mask_np.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        if return_candidate_mask and return_glass_render:
            return base_rgb, False, Image.new("L", base_rgb.size, 0), Image.new("RGB", base_rgb.size, (0, 0, 0))
        if return_candidate_mask:
            return base_rgb, False, Image.new("L", base_rgb.size, 0)
        if return_glass_render:
            return base_rgb, False, Image.new("RGB", base_rgb.size, (0, 0, 0))
        return base_rgb, False

    bbox_h = y2 - y1
    upper_limit = y1 + int(round(bbox_h * 0.55))
    upper_limit = max(y1 + 1, min(upper_limit, y2))
    lower_cut = y1 + int(round(bbox_h * 0.45))
    lower_cut = max(y1 + 1, min(lower_cut, y2))

    yy, xx = np.ogrid[:height, :width]
    geo_region = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < upper_limit)
    fg_region = mask_np > 0.25
    candidate_base = geo_region & fg_region
    candidate_override = None
    if candidate_mask_override is not None:
        override = np.array(candidate_mask_override.convert("L"), dtype=np.uint8)
        if override.shape != mask_np.shape:
            override = np.array(
                candidate_mask_override.convert("L").resize(base_rgb.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        candidate_override = override > 127

    if not candidate_base.any():
        candidate_mask = Image.new("L", base_rgb.size, 0)
        glass_render = Image.new("RGB", base_rgb.size, (0, 0, 0))
        if return_candidate_mask and return_glass_render:
            return base_rgb, False, candidate_mask, glass_render
        if return_candidate_mask:
            return base_rgb, False, candidate_mask
        if return_glass_render:
            return base_rgb, False, glass_render
        return base_rgb, False

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))

    if candidate_override is not None:
        candidate = candidate_override & candidate_base
        candidate &= (yy < lower_cut)
        if int(candidate.sum()) < 80:
            candidate_mask = Image.new("L", base_rgb.size, 0)
            glass_render = Image.new("RGB", base_rgb.size, (0, 0, 0))
            if return_candidate_mask and return_glass_render:
                return base_rgb, False, candidate_mask, glass_render
            if return_candidate_mask:
                return base_rgb, False, candidate_mask
            if return_glass_render:
                return base_rgb, False, glass_render
            return base_rgb, False
    else:
        if normalized_mode == "auto":
            texture_threshold = float(np.percentile(texture[candidate_base], 35))
            candidate = candidate_base & (texture <= texture_threshold)
            min_pixels = max(80, int(candidate_base.sum() * 0.02))
            if int(candidate.sum()) < min_pixels:
                candidate_mask = Image.new("L", base_rgb.size, 0)
                glass_render = Image.new("RGB", base_rgb.size, (0, 0, 0))
                if return_candidate_mask and return_glass_render:
                    return base_rgb, False, candidate_mask, glass_render
                if return_candidate_mask:
                    return base_rgb, False, candidate_mask
                if return_glass_render:
                    return base_rgb, False, glass_render
                return base_rgb, False
        else:
            candidate = candidate_base

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_out = hsv.copy()
    hsv_out[:, :, 1][candidate] = np.clip(hsv_out[:, :, 1][candidate] * 0.70, 0.0, 255.0)
    hsv_out[:, :, 2][candidate] = np.clip(hsv_out[:, :, 2][candidate] * 0.94, 0.0, 255.0)

    rgb_mod_u8 = cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2RGB)
    rgb_blur_u8 = cv2.bilateralFilter(rgb_mod_u8, d=9, sigmaColor=50, sigmaSpace=9)
    rgb_mod_u8 = np.clip((rgb_mod_u8.astype(np.float32) * 0.45) + (rgb_blur_u8.astype(np.float32) * 0.55), 0, 255).astype(
        np.uint8
    )

    rgb_mod = rgb_mod_u8.astype(np.float32) / 255.0
    base_np = rgb_u8.astype(np.float32) / 255.0

    candidate_alpha = _smoothstep(0.15, 0.95, mask_np) * candidate.astype(np.float32)
    output_np = (base_np * (1.0 - candidate_alpha[..., None])) + (rgb_mod * candidate_alpha[..., None])

    gradient_map = np.zeros((height, width), dtype=np.float32)
    gradient_line = np.linspace(0.12, 0.0, max(1, upper_limit - y1), dtype=np.float32)
    gradient_map[y1:upper_limit, x1:x2] = gradient_line[:, None]
    gradient_map = gradient_map * candidate_alpha
    neutral = np.ones_like(output_np) * 0.92
    output_np = (output_np * (1.0 - gradient_map[..., None])) + (neutral * gradient_map[..., None])

    output_u8 = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    candidate_mask_u8 = np.zeros((height, width), dtype=np.uint8)
    candidate_mask_u8[candidate] = 255
    candidate_mask = Image.fromarray(candidate_mask_u8, mode="L")
    glass_render_u8 = np.zeros_like(output_u8, dtype=np.uint8)
    glass_render_u8[candidate] = output_u8[candidate]
    glass_render = Image.fromarray(glass_render_u8, mode="RGB")

    result = Image.fromarray(output_u8, mode="RGB")
    if return_candidate_mask and return_glass_render:
        return result, True, candidate_mask, glass_render
    if return_candidate_mask:
        return result, True, candidate_mask
    if return_glass_render:
        return result, True, glass_render
    return result, True


def compute_edge_halo_stats(
    baseline_image: Image.Image,
    candidate_image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> dict[str, float]:
    baseline = np.array(baseline_image.convert("RGB"), dtype=np.float32)
    candidate = np.array(candidate_image.convert("RGB"), dtype=np.float32)
    alpha = np.array(foreground_mask.convert("L"), dtype=np.float32) / 255.0

    if baseline.shape != candidate.shape:
        candidate = np.array(candidate_image.convert("RGB").resize(baseline_image.size, Image.Resampling.LANCZOS), dtype=np.float32)
    if alpha.shape[:2] != baseline.shape[:2]:
        alpha = np.array(foreground_mask.convert("L").resize(baseline_image.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    height, width = alpha.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    region_alpha = alpha[y1:y2, x1:x2]
    if region_alpha.size == 0:
        return {"edgeHaloMeanDelta": 0.0, "edgeBandWidthPx": 0.0}

    edge = np.logical_and(region_alpha > 0.02, region_alpha < 0.98)
    if edge.sum() < 16:
        region_hard = (region_alpha >= 0.5).astype(np.uint8)
        edge = np.logical_and(
            cv2.dilate(region_hard, np.ones((3, 3), np.uint8)) > 0,
            cv2.erode(region_hard, np.ones((3, 3), np.uint8)) == 0,
        )

    if edge.sum() < 8:
        return {"edgeHaloMeanDelta": 0.0, "edgeBandWidthPx": 0.0}

    delta = np.abs(candidate[y1:y2, x1:x2] - baseline[y1:y2, x1:x2]).mean(axis=2)
    edge_halo_mean_delta = float(np.mean(delta[edge]))

    perimeter = cv2.Canny((region_alpha * 255.0).astype(np.uint8), 20, 80) > 0
    perimeter_count = float(max(int(perimeter.sum()), 1))
    edge_band_width_px = float(edge.sum() / perimeter_count)

    return {
        "edgeHaloMeanDelta": edge_halo_mean_delta,
        "edgeBandWidthPx": edge_band_width_px,
    }


def compute_composite_fringe_stats(
    composite_image: Image.Image,
    background_image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
) -> dict[str, float]:
    composite = np.array(composite_image.convert("RGB"), dtype=np.float32)
    background = np.array(background_image.convert("RGB"), dtype=np.float32)
    alpha = np.array(foreground_mask.convert("L"), dtype=np.float32) / 255.0

    if composite.shape != background.shape:
        background = np.array(
            background_image.convert("RGB").resize(composite_image.size, Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    if alpha.shape[:2] != composite.shape[:2]:
        alpha = np.array(
            foreground_mask.convert("L").resize(composite_image.size, Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0

    height, width = alpha.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    region_alpha = alpha[y1:y2, x1:x2]
    if region_alpha.size == 0:
        return {"fringeRgbMean": 0.0, "fringeRgbP95": 0.0}

    hard = (region_alpha >= 0.5).astype(np.uint8)
    if hard.max() == 0:
        return {"fringeRgbMean": 0.0, "fringeRgbP95": 0.0}

    outer = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    ring = np.logical_and(outer > 0, hard == 0)
    if ring.sum() < 12:
        return {"fringeRgbMean": 0.0, "fringeRgbP95": 0.0}

    diff = np.abs(composite[y1:y2, x1:x2] - background[y1:y2, x1:x2]).mean(axis=2)
    values = diff[ring]
    return {
        "fringeRgbMean": float(np.mean(values)),
        "fringeRgbP95": float(np.percentile(values, 95)),
    }


def detect_ground_plane(image: Image.Image) -> Image.Image:
    w, h = image.size
    ground = np.zeros((h, w), dtype=np.uint8)
    ground[int(h * 0.60) :, :] = 255
    return Image.fromarray(ground, mode="L")
