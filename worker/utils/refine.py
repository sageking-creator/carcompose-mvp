from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = _odd_kernel_size(radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_connected_component(mask_bin: np.ndarray) -> np.ndarray:
    if mask_bin.dtype != np.uint8:
        mask_bin = mask_bin.astype(np.uint8)
    if mask_bin.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if component_count <= 1:
        return mask_bin

    largest_label = 1
    largest_area = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_label = label
            largest_area = area
    return (labels == largest_label).astype(np.uint8)


def _guided_filter(guide_gray: np.ndarray, alpha_init: np.ndarray, radius: int, eps: float) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(guide_gray, alpha_init, int(radius), float(eps))
    return cv2.bilateralFilter(alpha_init, d=9, sigmaColor=0.15, sigmaSpace=max(10, radius // 2))


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    scaled = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - (2.0 * scaled))


def _decontaminate_foreground_rgb(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    image_rgb = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    outside = (alpha <= 0.02).astype(np.float32)
    if outside.sum() < 10:
        return image_rgb

    sigma = 10
    den = cv2.GaussianBlur(outside, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = np.maximum(den, 1e-3)

    background_estimate = np.zeros_like(image_rgb)
    for channel in range(3):
        weighted = image_rgb[:, :, channel] * outside
        num = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background_estimate[:, :, channel] = num / den

    edge = np.logical_and(alpha > 0.02, alpha < 0.98)
    if not np.any(edge):
        return image_rgb

    alpha_safe = np.maximum(alpha, 1e-3)[..., None]
    foreground_estimate = (image_rgb - ((1.0 - alpha)[..., None] * background_estimate)) / alpha_safe
    foreground_estimate = np.clip(foreground_estimate, 0.0, 1.0)

    blend_weight = _smoothstep(0.2, 0.9, alpha)[..., None]
    output = image_rgb.copy()
    output[edge] = ((1.0 - blend_weight[edge]) * image_rgb[edge]) + (
        blend_weight[edge] * foreground_estimate[edge]
    )
    return np.clip(output, 0.0, 1.0)


def build_hardened_alpha(
    image: Image.Image,
    prob_map: np.ndarray,
    threshold: float = 0.50,
    guided_radius: int = 45,
    guided_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    if prob_map.ndim != 2:
        raise ValueError("prob_map must be a 2D array.")

    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    height, width = prob.shape
    long_edge = max(height, width)
    guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mask_bin = (prob >= float(threshold)).astype(np.uint8)
    mask_bin = _largest_connected_component(mask_bin)
    if mask_bin.max() == 0:
        raise RuntimeError("Failed to build hardened alpha mask: no foreground component found.")

    core_mask = (prob >= 0.70).astype(np.uint8)
    core_mask = _largest_connected_component(core_mask)
    if core_mask.max() > 0:
        support_radius = int(np.clip(round(long_edge * 0.02), 30, 80))
        support_region = cv2.dilate(core_mask, _ellipse_kernel(support_radius))
        constrained = (mask_bin & (support_region > 0)).astype(np.uint8)
        if constrained.max() > 0:
            mask_bin = constrained
        else:
            mask_bin = core_mask

    k_close = _odd_kernel_size(max(5, round(long_edge * 0.004)))
    k_open = _odd_kernel_size(max(3, round(long_edge * 0.002)))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open)
    mask_bin = _largest_connected_component(mask_bin)
    if mask_bin.max() == 0:
        raise RuntimeError("Failed to build hardened alpha mask: no foreground after morphology.")

    edge_px = int(np.clip(round(long_edge * 0.006), 6, 20))
    edge_kernel = _ellipse_kernel(edge_px)
    solid = cv2.erode(mask_bin, edge_kernel)
    dilated = cv2.dilate(mask_bin, edge_kernel)
    edge_band = np.logical_and(dilated > 0, solid == 0)

    alpha_init = prob.copy()
    alpha_init[solid > 0] = 1.0
    alpha_init[dilated == 0] = 0.0

    guided = _guided_filter(guide, alpha_init, radius=guided_radius, eps=guided_eps)
    guided = np.clip(guided, 0.0, 1.0)

    alpha = np.zeros_like(prob, dtype=np.float32)
    alpha[solid > 0] = 1.0
    alpha[edge_band] = guided[edge_band]
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha, mask_bin.astype(np.uint8)


def refine_foreground(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    image_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    cleaned_rgb = _decontaminate_foreground_rgb(image_rgb, alpha)
    cleaned_u8 = np.clip(cleaned_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    cleaned_pil = Image.fromarray(cleaned_u8, mode="RGB")
    r_ch, g_ch, b_ch = cleaned_pil.split()
    return Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(alpha_u8, mode="L")))
