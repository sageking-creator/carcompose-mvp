from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np
from PIL import Image

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


def reharden_resized_alpha(mask_crop_resized: Image.Image, edge_px: int | None = None) -> Image.Image:
    alpha = np.array(mask_crop_resized.convert("L"), dtype=np.float32) / 255.0
    hard = (alpha >= 0.5).astype(np.uint8)
    if hard.max() == 0:
        return mask_crop_resized.convert("L")

    long_edge = max(alpha.shape)
    band_px = int(edge_px if edge_px is not None else np.clip(round(long_edge * 0.0025), 2, 5))
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
        edge_cap = 0.12
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


def _apply_contact_shadow_v1(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float,
) -> tuple[Image.Image, bool]:
    shadow_strength = max(0.0, min(1.0, float(strength)))
    if shadow_strength <= 0.0:
        return image.convert("RGB"), False

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
        return base_rgb, False

    region_mask = mask_np[y1:y2, x1:x2]
    if region_mask.max() <= 0.0:
        return base_rgb, False

    seed_start = int(round(bbox_h * (1.0 - 0.22)))
    seed_start = max(0, min(seed_start, bbox_h - 1))
    seed = region_mask[seed_start:, :]
    if seed.size == 0 or seed.max() <= 0.0:
        return base_rgb, False

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

    fg_alpha = mask_np
    shadow_alpha = shadow_canvas * shadow_strength * (1.0 - fg_alpha)
    output_np = base_np * (1.0 - shadow_alpha[..., None])
    output = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output, mode="RGB"), True


def _apply_contact_shadow_v2(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float,
) -> tuple[Image.Image, bool]:
    shadow_strength = max(0.0, min(1.0, float(strength)))
    if shadow_strength <= 0.0:
        return image.convert("RGB"), False

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
        return base_rgb, False

    region_hard = (mask_np[y1:y2, x1:x2] >= 0.5).astype(np.uint8)
    if region_hard.max() == 0:
        return base_rgb, False

    columns = np.where(region_hard.sum(axis=0) > 0)[0]
    if columns.size == 0:
        return base_rgb, False

    contour_seed = np.zeros((height, width), dtype=np.float32)
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
        return base_rgb, False

    thickness = max(1, int(round(bbox_h * 0.01)))
    contour_points = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(contour_seed, [contour_points], isClosed=False, color=1.0, thickness=thickness)

    if y_offset > 0:
        contour_seed = np.roll(contour_seed, shift=y_offset, axis=0)
        contour_seed[:y_offset, :] = 0.0

    # Reduce unrealistic “bumper shadow” by emphasizing wheel contact zones over the centerline.
    if bbox_w >= 64:
        x = np.linspace(0.0, 1.0, bbox_w, dtype=np.float32)
        sigma = 0.16
        w_left = np.exp(-0.5 * ((x - 0.28) / sigma) ** 2)
        w_right = np.exp(-0.5 * ((x - 0.72) / sigma) ** 2)
        weights = w_left + w_right
        weights = weights / max(float(weights.max()), 1e-6)
        contour_seed[:, x1:x2] = contour_seed[:, x1:x2] * weights[None, :]

    floor_mask = cv2.dilate(floor_mask, np.ones((1, 9), np.uint8), iterations=1)
    sigma_x = max(4.0, float(bbox_w) * 0.02)
    sigma_y = max(2.0, float(bbox_w) * 0.008)
    contour_blur = cv2.GaussianBlur(contour_seed, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

    fg_alpha = mask_np
    shadow_alpha = contour_blur * floor_mask * shadow_strength * (1.0 - fg_alpha)
    output_np = base_np * (1.0 - shadow_alpha[..., None])
    output = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output, mode="RGB"), True


def apply_contact_shadow(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    strength: float = 0.32,
    mode: Literal["v1", "v2"] = "v2",
) -> tuple[Image.Image, bool]:
    if mode == "v1":
        return _apply_contact_shadow_v1(image, foreground_mask, foreground_bbox, strength)
    return _apply_contact_shadow_v2(image, foreground_mask, foreground_bbox, strength)


def is_studio_background(image: Image.Image) -> bool:
    rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    if height < 16 or width < 16:
        return False

    # Focus on a crop that avoids the floor/turntable edge lines and bottom-right watermark.
    y1 = int(round(height * 0.05))
    y2 = int(round(height * 0.68))
    x1 = int(round(width * 0.10))
    x2 = int(round(width * 0.90))
    y2 = max(y1 + 8, min(y2, height))
    x2 = max(x1 + 8, min(x2, width))
    crop = rgb_u8[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    texture_score = float(np.percentile(lap, 80))

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_mean = float(np.mean(hsv[:, :, 1] / 255.0))
    value_std = float(np.std(hsv[:, :, 2] / 255.0))

    return texture_score < 12.0 and sat_mean < 0.22 and value_std < 0.28


def estimate_turntable_alignment(image: Image.Image) -> dict[str, int] | None:
    """
    Best-effort heuristic for studio “turntable” backgrounds.

    Returns:
      {"centerX": int, "groundY": int, "spanW": int}
    or None when detection is unreliable.
    """
    rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape[:2]
    if height < 64 or width < 64:
        return None

    y1 = int(round(height * 0.66))
    y2 = int(round(height * 0.93))
    x1 = int(round(width * 0.15))
    x2 = int(round(width * 0.85))
    if y2 <= y1 + 8 or x2 <= x1 + 8:
        return None

    roi = gray[y1:y2, x1:x2]
    grad_y = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))
    if grad_y.size == 0:
        return None

    max_vals = grad_y.max(axis=0)
    y_idxs = grad_y.argmax(axis=0)
    if max_vals.size == 0:
        return None

    threshold = float(np.percentile(max_vals, 82))
    threshold = max(threshold, 8.0)
    good = max_vals >= threshold
    if int(good.sum()) < max(40, int(0.25 * max_vals.size)):
        return None

    xs = np.where(good)[0] + x1
    ys = y_idxs[good] + y1

    span_w = int(xs.max() - xs.min() + 1)
    if span_w < int(width * 0.25):
        return None

    center_x = int(round((xs.max() + xs.min()) / 2.0))
    ground_y = int(np.median(ys))
    return {"centerX": center_x, "groundY": ground_y, "spanW": span_w}


def apply_glass_normalization(
    image: Image.Image,
    foreground_mask: Image.Image,
    foreground_bbox: Tuple[int, int, int, int],
    mode: Literal["off", "auto", "force"] = "off",
) -> tuple[Image.Image, bool]:
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"off", "auto", "force"}:
        normalized_mode = "off"
    if normalized_mode == "off":
        return image.convert("RGB"), False

    base_rgb = image.convert("RGB")
    mask_l = foreground_mask.convert("L")
    if mask_l.size != base_rgb.size:
        mask_l = mask_l.resize(base_rgb.size, Image.Resampling.LANCZOS)

    rgb_u8 = np.array(base_rgb, dtype=np.uint8)
    mask_np = np.array(mask_l, dtype=np.float32) / 255.0
    height, width = mask_np.shape
    x1, y1, x2, y2 = _clip_bbox(foreground_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return base_rgb, False

    bbox_h = y2 - y1
    upper_limit = y1 + int(round(bbox_h * 0.55))
    upper_limit = max(y1 + 1, min(upper_limit, y2))

    yy, xx = np.ogrid[:height, :width]
    geo_region = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < upper_limit)
    fg_region = mask_np > 0.25
    candidate_base = geo_region & fg_region
    if not candidate_base.any():
        return base_rgb, False

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))

    if normalized_mode == "auto":
        texture_threshold = float(np.percentile(texture[candidate_base], 35))
        candidate = candidate_base & (texture <= texture_threshold)
        min_pixels = max(80, int(candidate_base.sum() * 0.02))
        if int(candidate.sum()) < min_pixels:
            return base_rgb, False
    else:
        candidate = candidate_base

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_out = hsv.copy()
    hsv_out[:, :, 1][candidate] = np.clip(hsv_out[:, :, 1][candidate] * 0.75, 0.0, 255.0)
    hsv_out[:, :, 2][candidate] = np.clip(hsv_out[:, :, 2][candidate] * 0.92, 0.0, 255.0)

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
    gradient_line = np.linspace(0.06, 0.0, max(1, upper_limit - y1), dtype=np.float32)
    gradient_map[y1:upper_limit, x1:x2] = gradient_line[:, None]
    gradient_map = gradient_map * candidate_alpha
    neutral = np.ones_like(output_np) * 0.92
    output_np = (output_np * (1.0 - gradient_map[..., None])) + (neutral * gradient_map[..., None])

    output_u8 = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output_u8, mode="RGB"), True


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


def detect_ground_plane(image: Image.Image) -> Image.Image:
    w, h = image.size
    ground = np.zeros((h, w), dtype=np.uint8)
    ground[int(h * 0.60) :, :] = 255
    return Image.fromarray(ground, mode="L")
