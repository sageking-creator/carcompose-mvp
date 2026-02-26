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


def _compute_alpha_checks(alpha: np.ndarray, long_edge: int) -> dict[str, float]:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    hard = (alpha >= 0.5).astype(np.uint8)
    area_ratio = float(hard.mean())

    interior_kernel = _ellipse_kernel(max(3, int(round(long_edge * 0.015))))
    interior = cv2.erode(hard, interior_kernel)
    if interior.any():
        interior_opaque = float(np.mean(alpha[interior > 0] >= 0.98))
    elif hard.any():
        interior_opaque = float(np.mean(alpha[hard > 0] >= 0.98))
    else:
        interior_opaque = 0.0

    leak_kernel = _ellipse_kernel(max(6, int(round(long_edge * 0.02))))
    dilated = cv2.dilate(hard, leak_kernel)
    outside = dilated == 0
    outside_leak = float(np.mean(alpha[outside])) if outside.any() else 0.0

    return {
        "interiorOpaqueRatio": interior_opaque,
        "outsideLeakMeanAlpha": outside_leak,
        "maskAreaRatio": area_ratio,
    }


def _mask_quality_score(checks: dict[str, float]) -> float:
    area_ratio = checks["maskAreaRatio"]
    outside_leak = checks["outsideLeakMeanAlpha"]
    interior_opaque = checks["interiorOpaqueRatio"]

    score = outside_leak * 5.0
    score += max(0.0, 0.985 - interior_opaque) * 2.0
    if area_ratio < 0.01:
        score += (0.01 - area_ratio) * 20.0
    if area_ratio > 0.80:
        score += (area_ratio - 0.80) * 10.0
    return float(score)


def _build_candidate_alpha(
    prob: np.ndarray,
    guide: np.ndarray,
    long_edge: int,
    *,
    core_threshold: float,
    support_threshold: float,
    support_expand_scale: float,
    guided_radius: int,
    guided_eps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    core = (prob >= float(core_threshold)).astype(np.uint8)
    core = _largest_connected_component(core)

    k_close = _odd_kernel_size(int(np.clip(round(long_edge * 0.0015), 3, 9)))
    k_open = _odd_kernel_size(int(np.clip(round(long_edge * 0.0010), 3, 7)))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    core = _largest_connected_component(core)

    support = (prob >= float(support_threshold)).astype(np.uint8)
    support = _largest_connected_component(support)

    expand_px = int(np.clip(round(long_edge * 0.02 * float(support_expand_scale)), 30, 80))
    allowed = cv2.dilate(core, _ellipse_kernel(expand_px))
    if allowed.any():
        support = (support & (allowed > 0)).astype(np.uint8)
    support = _largest_connected_component(support)
    if not support.any() and core.any():
        support = core.copy()

    edge_px = int(np.clip(round(long_edge * 0.004), 4, 16))
    edge_kernel = _ellipse_kernel(edge_px)
    solid = cv2.erode(support, edge_kernel)
    dilated = cv2.dilate(support, edge_kernel)
    edge_band = np.logical_and(dilated > 0, solid == 0)

    alpha_init = prob.copy()
    alpha_init[solid > 0] = 1.0
    alpha_init[dilated == 0] = 0.0

    guided = _guided_filter(guide, alpha_init, radius=guided_radius, eps=guided_eps)
    guided = np.clip(guided, 0.0, 1.0)

    alpha = alpha_init.copy()
    alpha[edge_band] = guided[edge_band]
    alpha[solid > 0] = 1.0
    alpha[dilated == 0] = 0.0

    checks = _compute_alpha_checks(alpha, long_edge)
    if checks["outsideLeakMeanAlpha"] > 0.005:
        clamp_px = int(np.clip(round(long_edge * 0.01), 16, 48))
        clamp_region = cv2.dilate(core if core.any() else support, _ellipse_kernel(clamp_px))
        if clamp_region.any():
            alpha = alpha * (clamp_region > 0).astype(np.float32)
            checks = _compute_alpha_checks(alpha, long_edge)

    return np.clip(alpha, 0.0, 1.0), support, checks


def build_hardened_alpha(
    image: Image.Image,
    prob_map: np.ndarray,
    threshold: float = 0.50,
    guided_radius: int = 35,
    guided_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    del threshold
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    if prob_map.ndim != 2:
        raise ValueError("prob_map must be a 2D array.")

    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    height, width = prob.shape
    long_edge = max(height, width)
    guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    candidates = [
        (0.65, 0.50, 1.0),
        (0.72, 0.50, 0.8),
    ]

    best_alpha = None
    best_support = None
    best_score = float("inf")
    for core_threshold, support_threshold, support_expand_scale in candidates:
        alpha, support, checks = _build_candidate_alpha(
            prob=prob,
            guide=guide,
            long_edge=long_edge,
            core_threshold=core_threshold,
            support_threshold=support_threshold,
            support_expand_scale=support_expand_scale,
            guided_radius=guided_radius,
            guided_eps=guided_eps,
        )
        score = _mask_quality_score(checks)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_support = support

    if best_alpha is None or best_support is None:
        raise RuntimeError("Failed to build hardened alpha mask.")
    return best_alpha, best_support.astype(np.uint8)


def refine_foreground(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    r_ch, g_ch, b_ch = image.convert("RGB").split()
    return Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(alpha_u8, mode="L")))
