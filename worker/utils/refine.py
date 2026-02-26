from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


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

    mask_bin = (prob >= float(threshold)).astype(np.uint8)
    mask_bin = _largest_connected_component(mask_bin)

    k_close = _odd_kernel_size(max(5, int(round(long_edge * 0.004))))
    k_open = _odd_kernel_size(max(3, int(round(long_edge * 0.002))))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)
    mask_clean = _largest_connected_component(mask_clean)

    edge_px = int(np.clip(round(long_edge * 0.006), 6, 20))
    edge_kernel_size = _odd_kernel_size(edge_px * 2 + 1)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_kernel_size, edge_kernel_size))

    solid = cv2.erode(mask_clean, edge_kernel)
    dilated = cv2.dilate(mask_clean, edge_kernel)
    edge_band = np.logical_and(dilated > 0, solid == 0)

    alpha_init = prob.copy()
    alpha_init[solid > 0] = 1.0
    alpha_init[dilated == 0] = 0.0

    guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guided = _guided_filter(guide, alpha_init, radius=guided_radius, eps=guided_eps)
    guided = np.clip(guided, 0.0, 1.0)

    alpha = alpha_init.copy()
    alpha[edge_band] = guided[edge_band]
    alpha[solid > 0] = 1.0
    alpha[dilated == 0] = 0.0
    return alpha, mask_clean.astype(np.uint8)


def refine_foreground(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    r_ch, g_ch, b_ch = image.convert("RGB").split()
    return Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(alpha_u8, mode="L")))
