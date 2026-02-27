from __future__ import annotations

from dataclasses import dataclass

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
    output = np.clip(output, 0.0, 1.0)

    # Defringe: dilate premultiplied RGB outward so semi-transparent edge pixels do not
    # carry the source background color. This is identity-preserving because it only
    # affects pixels where alpha is already low (outside the hard mask).
    try:
        height, width = alpha.shape
        long_edge = max(height, width)
        radius = int(np.clip(round(long_edge * 0.003), 2, 6))
        kernel = _ellipse_kernel(radius)

        alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        premult = output * alpha_f[..., None]

        premult_dil = cv2.dilate(premult.astype(np.float32), kernel, iterations=1)
        alpha_dil = cv2.dilate(alpha_f.astype(np.float32), kernel, iterations=1)
        rgb_dil = premult_dil / np.maximum(alpha_dil, 1e-4)[..., None]
        rgb_dil = np.clip(rgb_dil, 0.0, 1.0)

        # Only replace outside the hard edge, where bleeding is visible.
        fringe = np.logical_and(alpha_f > 0.001, alpha_f < 0.50)
        if fringe.any():
            output[fringe] = rgb_dil[fringe]
    except Exception:
        pass

    return output


@dataclass(frozen=True)
class AlphaCandidateConfig:
    name: str
    threshold: float
    core_threshold: float
    edge_scale: float
    support_scale: float


_ALPHA_CANDIDATES: tuple[AlphaCandidateConfig, ...] = (
    AlphaCandidateConfig(
        name="balanced",
        threshold=0.50,
        core_threshold=0.75,
        edge_scale=0.0040,
        support_scale=0.0120,
    ),
    AlphaCandidateConfig(
        name="inclusive",
        threshold=0.45,
        core_threshold=0.70,
        edge_scale=0.0035,
        support_scale=0.0100,
    ),
    AlphaCandidateConfig(
        name="strict",
        threshold=0.55,
        core_threshold=0.80,
        edge_scale=0.0045,
        support_scale=0.0140,
    ),
)

_ALPHA_CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in _ALPHA_CANDIDATES}


def _build_candidate_alpha(
    *,
    prob: np.ndarray,
    guide: np.ndarray,
    long_edge: int,
    guided_radius: int,
    guided_eps: float,
    candidate: AlphaCandidateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mask_bin = (prob >= float(candidate.threshold)).astype(np.uint8)
    mask_bin = _largest_connected_component(mask_bin)
    if mask_bin.max() == 0:
        raise RuntimeError("No foreground component found.")

    core_mask = (prob >= float(candidate.core_threshold)).astype(np.uint8)
    core_mask = _largest_connected_component(core_mask)
    if core_mask.max() > 0:
        support_radius = int(np.clip(round(long_edge * candidate.support_scale), 10, 48))
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
        raise RuntimeError("Foreground vanished after morphology.")

    edge_px = int(np.clip(round(long_edge * candidate.edge_scale), 3, 16))
    edge_kernel = _ellipse_kernel(edge_px)
    solid = cv2.erode(mask_bin, edge_kernel)
    if solid.max() == 0:
        reduced_edge_px = max(1, edge_px // 2)
        reduced_kernel = _ellipse_kernel(reduced_edge_px)
        solid = cv2.erode(mask_bin, reduced_kernel)
        edge_px = reduced_edge_px
    if solid.max() == 0:
        solid = core_mask if core_mask.max() > 0 else mask_bin

    dilated = cv2.dilate(mask_bin, _ellipse_kernel(edge_px))
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


def _tighten_alpha_from_core(
    *,
    alpha: np.ndarray,
    prob: np.ndarray,
    long_edge: int,
) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    core = (prob >= 0.86).astype(np.uint8)
    core = _largest_connected_component(core)
    if core.max() == 0:
        core = (alpha >= 0.85).astype(np.uint8)
        core = _largest_connected_component(core)
    if core.max() == 0:
        return alpha

    gate_radius = int(np.clip(round(long_edge * 0.007), 6, 24))
    gate = cv2.dilate(core, _ellipse_kernel(gate_radius))

    hard = (alpha >= 0.5).astype(np.uint8)
    constrained = (hard & (gate > 0)).astype(np.uint8)
    if constrained.max() == 0:
        constrained = gate.astype(np.uint8)
    constrained = _largest_connected_component(constrained)
    if constrained.max() == 0:
        return alpha

    edge_px = int(np.clip(round(long_edge * 0.0025), 2, 7))
    edge_kernel = _ellipse_kernel(edge_px)
    solid = cv2.erode(constrained, edge_kernel)
    if solid.max() == 0:
        solid = constrained
    dilated = cv2.dilate(constrained, edge_kernel)
    edge_band = np.logical_and(dilated > 0, solid == 0)

    alpha_soft = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(0.8, edge_px / 2.0), sigmaY=max(0.8, edge_px / 2.0))
    tightened = np.zeros_like(alpha, dtype=np.float32)
    tightened[solid > 0] = 1.0
    if edge_band.any():
        edge_vals = np.clip(alpha_soft[edge_band], 0.0, 1.0)
        # Keep a narrow, low-alpha outside band to reduce halo visibility while preserving AA.
        edge_cap = 0.48  # must stay < 0.5 to remain outside after 8-bit quantization
        edge_vals = np.minimum(edge_vals, edge_cap)
        gamma = 2.4
        edge_vals = edge_cap * np.power(edge_vals / max(edge_cap, 1e-6), gamma)
        tightened[edge_band] = np.clip(edge_vals, 0.0, edge_cap)
    return np.clip(tightened, 0.0, 1.0)


def _mask_checks_for_alpha(alpha: np.ndarray) -> dict[str, float]:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    hard = (alpha >= 0.5).astype(np.uint8)

    mask_area_ratio = float(hard.mean())
    if hard.max() == 0:
        return {
            "interiorOpaqueRatio": 0.0,
            "outsideLeakMeanAlpha": 1.0,
            "nearLeakMeanAlpha": 1.0,
            "nearLeakP95Alpha": 1.0,
            "maskAreaRatio": 0.0,
        }

    rows = np.where(np.any(hard > 0, axis=1))[0]
    cols = np.where(np.any(hard > 0, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return {
            "interiorOpaqueRatio": 0.0,
            "outsideLeakMeanAlpha": 1.0,
            "nearLeakMeanAlpha": 1.0,
            "nearLeakP95Alpha": 1.0,
            "maskAreaRatio": mask_area_ratio,
        }

    y1, y2 = int(rows[0]), int(rows[-1] + 1)
    x1, x2 = int(cols[0]), int(cols[-1] + 1)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    region_alpha = alpha[y1:y2, x1:x2]
    region_hard = hard[y1:y2, x1:x2]
    interior_kernel_size = _odd_kernel_size(max(7, int(round(max(bbox_w, bbox_h) * 0.06))))
    interior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (interior_kernel_size, interior_kernel_size))
    interior = cv2.erode(region_hard, interior_kernel)
    if interior.sum() > 0:
        interior_opaque_ratio = float(np.mean(region_alpha[interior > 0] >= 0.98))
    else:
        interior_opaque_ratio = float(np.mean(region_alpha >= 0.98)) if region_alpha.size else 0.0

    leak_kernel_size = _odd_kernel_size(max(9, int(round(max(bbox_w, bbox_h) * 0.02))))
    leak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (leak_kernel_size, leak_kernel_size))
    dilated = cv2.dilate(hard, leak_kernel)
    outside = dilated == 0
    outside_leak_mean_alpha = float(np.mean(alpha[outside])) if outside.any() else 0.0

    near_kernel_size = _odd_kernel_size(max(9, int(round(max(bbox_w, bbox_h) * 0.04))))
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_kernel_size, near_kernel_size))
    outer_ring = np.logical_and(cv2.dilate(hard, near_kernel) > 0, hard == 0)
    if outer_ring.any():
        near_leak_mean_alpha = float(np.mean(alpha[outer_ring]))
        near_leak_p95_alpha = float(np.percentile(alpha[outer_ring], 95))
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


def _candidate_quality_score(checks: dict[str, float]) -> float:
    return (
        (1000.0 * checks["interiorOpaqueRatio"])
        - (20000.0 * checks["outsideLeakMeanAlpha"])
        - (12000.0 * checks["nearLeakMeanAlpha"])
        - (8000.0 * checks["nearLeakP95Alpha"])
        - (200.0 * abs(checks["maskAreaRatio"] - 0.18))
    )


def _passes_quality_gate(checks: dict[str, float]) -> bool:
    return (
        checks["interiorOpaqueRatio"] >= 0.985
        and checks["outsideLeakMeanAlpha"] <= 0.01
        and checks["nearLeakMeanAlpha"] <= 0.02
        and checks["nearLeakP95Alpha"] <= 0.12
        and 0.01 <= checks["maskAreaRatio"] <= 0.85
    )


def build_hardened_alpha(
    image: Image.Image,
    prob_map: np.ndarray,
    threshold: float = 0.50,
    guided_radius: int = 45,
    guided_eps: float = 1e-4,
    mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    if prob_map.ndim != 2:
        raise ValueError("prob_map must be a 2D array.")

    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    height, width = prob.shape
    long_edge = max(height, width)
    guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    normalized_mode = str(mode).strip().lower()

    if normalized_mode in {"balanced", "inclusive", "strict"}:
        candidate = _ALPHA_CANDIDATE_BY_NAME[normalized_mode]
        return _build_candidate_alpha(
            prob=prob,
            guide=guide,
            long_edge=long_edge,
            guided_radius=guided_radius,
            guided_eps=guided_eps,
            candidate=candidate,
        )

    if normalized_mode == "single":
        dynamic_candidate = AlphaCandidateConfig(
            name="single",
            threshold=float(threshold),
            core_threshold=float(np.clip(max(threshold + 0.20, 0.70), 0.70, 0.95)),
            edge_scale=0.0040,
            support_scale=0.0120,
        )
        return _build_candidate_alpha(
            prob=prob,
            guide=guide,
            long_edge=long_edge,
            guided_radius=guided_radius,
            guided_eps=guided_eps,
            candidate=dynamic_candidate,
        )

    best_alpha: np.ndarray | None = None
    best_mask: np.ndarray | None = None
    best_score = float("-inf")
    first_error: Exception | None = None

    for candidate in _ALPHA_CANDIDATES:
        try:
            alpha_candidate, mask_candidate = _build_candidate_alpha(
                prob=prob,
                guide=guide,
                long_edge=long_edge,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                candidate=candidate,
            )
        except Exception as error:
            if first_error is None:
                first_error = error
            continue

        checks = _mask_checks_for_alpha(alpha_candidate)
        if _passes_quality_gate(checks):
            return alpha_candidate, mask_candidate.astype(np.uint8)

        tightened_alpha = _tighten_alpha_from_core(alpha=alpha_candidate, prob=prob, long_edge=long_edge)
        tightened_checks = _mask_checks_for_alpha(tightened_alpha)
        tightened_mask = (tightened_alpha >= 0.5).astype(np.uint8)
        tightened_mask = _largest_connected_component(tightened_mask)

        if _passes_quality_gate(tightened_checks) and tightened_mask.max() > 0:
            return tightened_alpha, tightened_mask.astype(np.uint8)

        score = _candidate_quality_score(checks)
        if score > best_score:
            best_score = score
            best_alpha = alpha_candidate
            best_mask = mask_candidate

        tightened_score = _candidate_quality_score(tightened_checks)
        if tightened_mask.max() > 0 and tightened_score > best_score:
            best_score = tightened_score
            best_alpha = tightened_alpha
            best_mask = tightened_mask

    if best_alpha is not None and best_mask is not None:
        return best_alpha, best_mask.astype(np.uint8)

    if first_error is not None:
        raise RuntimeError(f"Failed to build hardened alpha: {first_error}") from first_error
    raise RuntimeError("Failed to build hardened alpha: no candidate succeeded.")


def refine_foreground(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    image_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    cleaned_rgb = _decontaminate_foreground_rgb(image_rgb, alpha)
    cleaned_u8 = np.clip(cleaned_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    cleaned_pil = Image.fromarray(cleaned_u8, mode="RGB")
    r_ch, g_ch, b_ch = cleaned_pil.split()
    return Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(alpha_u8, mode="L")))
