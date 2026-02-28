from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
from loguru import logger
import numpy as np
from PIL import Image

from exceptions import HarmonyScoreTooLowError, InvalidInputError
from models.birefnet import BiRefNetSegmenter
from models.bargainnet import BargainNetScorer
from models.controlcom import ControlComHarmonizer
from models.libcom_reflection import LibcomReflectionGenerator
from models.libcom_shadow import LibcomShadowGenerator
from models.sam2_glass import Sam2GlassSegmenter
from models.vitmatte import VitMatteRefiner
from settings import Settings
from utils.image import (
    download_image,
    download_image_raw,
    fit_background,
    generate_reshoot_guidance,
    upload_debug_image_put,
    upload_image_put,
    validate_image,
)
from utils.image_ops import (
    apply_contact_shadow,
    apply_glass_normalization,
    apply_low_frequency_harmonization,
    apply_multiband_harmonization,
    blend_background_only,
    compute_composite_fringe_stats,
    compute_edge_halo_stats,
    compute_mask_artifact_checks,
    compute_detail_preservation_ratio,
    defringe_to_target_background,
    estimate_turntable_alignment,
    get_tight_bbox_from_mask,
    is_studio_background,
    paste_mask_into_background,
    render_placement_overlay,
    reharden_resized_alpha,
    resize_rgba_premultiplied,
)
from utils.refine import build_hardened_alpha, refine_foreground, sanitize_external_alpha

_models: Dict[str, Any] = {}
_DEBUG_ARTIFACT_CONTENT_TYPES: dict[str, str] = {
    "mask_png": "image/png",
    "trimap_png": "image/png",
    "vitmatte_alpha_png": "image/png",
    "edge_band_png": "image/png",
    "foreground_rgba_png": "image/png",
    "placed_mask_png": "image/png",
    "composite_raw_jpg": "image/jpeg",
    "controlcom_guidance_jpg": "image/jpeg",
    "harmonized_jpg": "image/jpeg",
    "final_jpg": "image/jpeg",
    "shadow_mask_png": "image/png",
    "glass_mask_png": "image/png",
    "glass_render_jpg": "image/jpeg",
    "placement_overlay_jpg": "image/jpeg",
}


def _now_s() -> float:
    return time.perf_counter()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mask_checks_source_fail(checks: dict[str, float]) -> bool:
    return (
        checks["maskAreaRatio"] < 0.005
        or checks["maskAreaRatio"] > 0.85
        or checks["interiorOpaqueRatio"] < 0.985
        or checks["outsideLeakMeanAlpha"] > 0.01
    )


def _mask_checks_guidance_risky(checks: dict[str, float]) -> bool:
    return (
        _mask_checks_source_fail(checks)
        or checks["nearLeakMeanAlpha"] > 0.02
        or checks["nearLeakP95Alpha"] > 0.12
    )


def _fringe_risky(stats: dict[str, float], settings: Settings) -> bool:
    return (
        float(stats.get("fringeRgbMean", 0.0)) > settings.max_fringe_rgb_mean
        or float(stats.get("fringeRgbP95", 0.0)) > settings.max_fringe_rgb_p95
    )


def _build_vitmatte_trimap(alpha_init: np.ndarray) -> Image.Image:
    alpha = np.clip(alpha_init.astype(np.float32), 0.0, 1.0)
    height, width = alpha.shape
    long_edge = max(height, width)
    radius = int(np.clip(round(long_edge * 0.010), 12, 28))
    kernel_size = max(3, (radius * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    hard_fg = (alpha >= 0.95).astype(np.uint8)
    hard_bg = (alpha <= 0.05).astype(np.uint8)
    sure_fg = cv2.erode(hard_fg, kernel, iterations=1)
    sure_bg = cv2.erode(hard_bg, kernel, iterations=1)

    trimap = np.full((height, width), 128, dtype=np.uint8)
    trimap[sure_bg > 0] = 0
    trimap[sure_fg > 0] = 255
    return Image.fromarray(trimap, mode="L")


def _maybe_dump_debug_image(settings: Settings, job_id: str, name: str, image: Image.Image) -> None:
    if not settings.debug_artifacts:
        return
    try:
        out_dir = Path("/tmp/carcompose-debug") / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "png" if image.mode in {"RGBA", "LA", "L"} else "jpg"
        path = out_dir / f"{name}.{ext}"
        if ext == "jpg":
            image.convert("RGB").save(path, format="JPEG", quality=96, subsampling=0, optimize=True)
        else:
            image.save(path, format="PNG")
    except Exception as error:
        logger.warning(f"[{job_id}] Failed to write debug artifact '{name}': {error}")


def _emit_debug_artifact(
    *,
    settings: Settings,
    job_id: str,
    local_name: str,
    remote_key: str,
    image: Image.Image,
    debug_put_urls: dict[str, str],
) -> None:
    _maybe_dump_debug_image(settings, job_id, local_name, image)
    put_url = debug_put_urls.get(remote_key)
    if not put_url:
        return

    content_type = _DEBUG_ARTIFACT_CONTENT_TYPES.get(remote_key)
    if not content_type:
        return

    try:
        upload_debug_image_put(put_url, image, content_type)
    except Exception as error:
        logger.warning(f"[{job_id}] Failed to upload debug artifact '{remote_key}': {error}")


def get_models(settings: Settings, *, variant: str, require_segmenter: bool = True) -> Dict[str, Any]:
    global _models

    if require_segmenter and "segmenter" not in _models:
        cache_dir = Path(settings.model_cache_dir)
        logger.info("Loading BiRefNet segmenter (cold start)...")
        _models["segmenter"] = BiRefNetSegmenter(
            cache_dir / "birefnet",
            infer_res=settings.birefnet_infer_res,
            repo_id=settings.birefnet_repo_id,
        )

    if "harmonizer" not in _models:
        logger.info("Loading ControlCom harmonizer (cold start)...")
        _models["harmonizer"] = ControlComHarmonizer(
            repo_dir=Path(settings.controlcom_repo_dir),
            ckpt_path=Path(settings.controlcom_ckpt),
            clip_dir=Path(settings.clip_model_dir),
            timeout_s=settings.controlcom_timeout_s,
        )

    if settings.enable_vitmatte and "vitmatte" not in _models:
        logger.info("Loading ViTMatte refiner...")
        _models["vitmatte"] = VitMatteRefiner(
            model_id=settings.vitmatte_model_id,
            cache_dir=Path(settings.hf_home),
        )

    if settings.glass_mode in {"sam2_auto", "sam2_force"} and "sam2_glass" not in _models:
        logger.info("Loading SAM2 glass segmenter...")
        _models["sam2_glass"] = Sam2GlassSegmenter(
            model_id=settings.sam2_model_id,
            cache_dir=Path(settings.hf_home),
        )

    if variant == "full":
        if "shadow" not in _models:
            logger.info("Loading libcom GPSDiffusion shadow generator...")
            _models["shadow"] = LibcomShadowGenerator()
        if "reflection" not in _models:
            logger.info("Loading libcom reflection generator...")
            _models["reflection"] = LibcomReflectionGenerator()
        if "scorer" not in _models:
            logger.info("Loading BargainNet harmony scorer...")
            _models["scorer"] = BargainNetScorer()

    return _models


def place_car_on_background(
    *,
    car_rgba_refined: Image.Image,
    tight_bbox: Tuple[int, int, int, int],
    bg_image: Image.Image,
    studio_background: bool,
    studio_car_width_ratio: float,
    studio_turntable_coverage: float,
    studio_ground_ratio: float,
    studio_ground_bias_px: int,
) -> tuple[Image.Image, Tuple[int, int, int, int], Image.Image, Image.Image, dict[str, Any]]:
    """
    Paste refined RGBA car onto background, sized to fill ~70% of background width and
    aligned to "ground" near ~85% of background height.

    Returns:
      composite_raw: RGB composite before ControlCom
      placement_bbox: (x1,y1,x2,y2) in background coordinates
      placed_mask: L mask resized to placement size (same as bbox width/height)
      placed_foreground_rgb: RGB crop resized to placement size
    """

    bg_w, bg_h = bg_image.size
    tb_w = max(1, tight_bbox[2] - tight_bbox[0])
    tb_h = max(1, tight_bbox[3] - tight_bbox[1])
    aspect = tb_w / tb_h

    width_ratio = float(studio_car_width_ratio) if studio_background else 0.70
    ground_ratio = float(studio_ground_ratio) if studio_background else 0.85

    alignment = estimate_turntable_alignment(bg_image) if studio_background else None
    if alignment:
        turntable_target = float(np.clip(float(studio_turntable_coverage), 0.65, 0.98))
        car_w = min(int(bg_w * width_ratio), int(alignment["spanW"] * turntable_target))
    else:
        car_w = int(bg_w * width_ratio)
    car_h = int(car_w / max(aspect, 1e-6))
    if car_h > int(bg_h * 0.80):
        car_h = int(bg_h * 0.80)
        car_w = int(car_h * aspect)

    car_crop_rgba = car_rgba_refined.crop(tight_bbox)
    resized_rgb, resized_alpha = resize_rgba_premultiplied(car_crop_rgba, (car_w, car_h))
    placed_mask = reharden_resized_alpha(resized_alpha, studio_mode=studio_background)
    placed_mask_u8 = np.array(placed_mask, dtype=np.uint8)
    alpha_np = placed_mask_u8.astype(np.float32) / 255.0
    rgb_np = np.array(resized_rgb, dtype=np.uint8)
    rgb_np[alpha_np <= 0.01] = 0
    placed_foreground_rgb = Image.fromarray(rgb_np, mode="RGB")
    car_placed_rgba = Image.merge("RGBA", (*placed_foreground_rgb.split(), placed_mask))

    strict_pixels = np.where(placed_mask_u8 >= 230)
    strict_rows = strict_pixels[0]
    if strict_rows.size > 0:
        strict_cols = strict_pixels[1]
        if strict_cols.size >= 16:
            col_min = int(strict_cols.min())
            col_max = int(strict_cols.max())
            per_col_bottom: list[int] = []
            per_col_index: list[int] = []
            for col in range(col_min, col_max + 1):
                rows_for_col = np.where(placed_mask_u8[:, col] >= 230)[0]
                if rows_for_col.size == 0:
                    continue
                per_col_bottom.append(int(rows_for_col.max()))
                per_col_index.append(col)

            if per_col_bottom:
                bottoms = np.array(per_col_bottom, dtype=np.float32)
                # Robust ground-contact estimate: ignore tiny outlier spikes beneath the car.
                strict_bottom_local = int(np.clip(np.percentile(bottoms, 94), 0, car_h - 1))
                contact_cols = np.array(
                    [col for col, row in zip(per_col_index, per_col_bottom) if row >= strict_bottom_local - 2],
                    dtype=np.float32,
                )
                if contact_cols.size > 0:
                    strict_center_local_x = int(round(float(np.median(contact_cols))))
                else:
                    strict_center_local_x = int(round(float(np.mean(strict_cols))))
            else:
                strict_bottom_local = int(strict_rows.max())
                strict_center_local_x = int(round(float(np.mean(strict_cols))))
        else:
            strict_bottom_local = int(strict_rows.max())
            strict_center_local_x = int(round(float(np.mean(strict_cols))))
    else:
        fallback_rows = np.where(placed_mask_u8 >= 1)[0]
        strict_bottom_local = int(fallback_rows.max()) if fallback_rows.size > 0 else car_h - 1
        strict_center_local_x = car_w // 2

    if alignment:
        x = int(alignment["centerX"] - strict_center_local_x)
        y = int(alignment["groundY"] + int(studio_ground_bias_px) - strict_bottom_local - 1)
    else:
        x = (bg_w - car_w) // 2
        y = int(bg_h * ground_ratio) - strict_bottom_local - 1
    y = max(0, min(y, bg_h - car_h))
    x = max(0, min(x, bg_w - car_w))

    canvas = bg_image.copy().convert("RGBA")
    canvas.paste(car_placed_rgba, (x, y), car_placed_rgba)
    return (
        canvas.convert("RGB"),
        (x, y, x + car_w, y + car_h),
        placed_mask,
        placed_foreground_rgb,
        {"alignment": alignment, "strictBottomLocal": strict_bottom_local},
    )


def run_pipeline(payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    job_id = payload.get("job_id")
    car_url = payload.get("car_image_url")
    car_mask_url = payload.get("car_mask_url")
    car_cutout_url = payload.get("car_cutout_url")
    background_url = payload.get("background_image_url")
    output_put_url = payload.get("output_put_url")

    if not job_id:
        raise InvalidInputError("job_id", "Missing job_id")
    if not car_url or not background_url or not output_put_url:
        raise InvalidInputError("input", "Missing input/output URLs")

    options = payload.get("options", {}) or {}
    harmony_threshold = _clamp01(float(options.get("harmony_threshold", 0.65)))
    shadow_strength = _clamp01(float(options.get("shadow_strength", 0.85)))
    reflection_strength = _clamp01(float(options.get("reflection_strength", 0.60)))

    requested_variant = str(payload.get("pipeline_variant") or "").lower()
    env_variant = (settings.pipeline_variant or "core").lower()
    variant = "full" if requested_variant == "full" and env_variant == "full" else "core"
    debug_put_urls_input = payload.get("debug_put_urls")
    debug_put_urls: dict[str, str] = {}
    if isinstance(debug_put_urls_input, dict):
        for key, value in debug_put_urls_input.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                debug_put_urls[key] = value

    use_external_mask = isinstance(car_mask_url, str) and car_mask_url.strip() != ""
    use_external_cutout = isinstance(car_cutout_url, str) and car_cutout_url.strip() != ""
    models = get_models(settings, variant=variant, require_segmenter=not (use_external_mask or use_external_cutout))
    timings: Dict[str, float] = {}

    logger.info(f"[{job_id}] Downloading inputs...")
    t0 = _now_s()
    car_image = download_image(car_url)
    background_image = download_image(background_url)
    validate_image(car_image, "car", settings.max_pixels)
    validate_image(background_image, "background", settings.max_pixels)
    timings["download_s"] = round(_now_s() - t0, 3)

    bg_proc = fit_background(
        background_image,
        settings.max_output_long_edge,
        resize_mode=settings.output_resize_mode,
        target_size=(settings.target_width, settings.target_height),
    )
    studio_background = is_studio_background(bg_proc)
    if settings.studio_mode == "on":
        studio_background = True
    elif settings.studio_mode == "off":
        studio_background = False

    logger.info(f"[{job_id}] Step 1: foreground extraction + edge refinement...")
    t0 = _now_s()
    external_prob_map: np.ndarray | None = None
    external_cutout_rgba: Image.Image | None = None
    alpha_source: np.ndarray | None = None
    if use_external_cutout:
        external_cutout = download_image_raw(str(car_cutout_url))
        external_cutout_rgba = external_cutout.convert("RGBA")
        if external_cutout_rgba.size != car_image.size:
            external_cutout_rgba = external_cutout_rgba.resize(car_image.size, Image.Resampling.LANCZOS)

        cutout_alpha = np.array(external_cutout_rgba.getchannel("A"), dtype=np.float32) / 255.0
        alpha = sanitize_external_alpha(cutout_alpha)
        alpha_source = alpha
        car_mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")

        cutout_rgb_np = np.array(external_cutout_rgba.convert("RGB"), dtype=np.uint8)
        cutout_rgb_np[alpha <= 0.01] = 0
        car_rgba_refined = Image.fromarray(
            np.dstack((cutout_rgb_np, np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8))),
            mode="RGBA",
        )
    elif use_external_mask:
        external_mask_image = download_image_raw(str(car_mask_url))
        if "A" in external_mask_image.getbands():
            external_prob_map = np.array(external_mask_image.getchannel("A"), dtype=np.float32) / 255.0
        else:
            external_prob_map = np.array(external_mask_image.convert("L"), dtype=np.float32) / 255.0
        if external_prob_map.shape != (car_image.height, car_image.width):
            resized = Image.fromarray(np.clip(external_prob_map * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            resized = resized.resize(car_image.size, Image.Resampling.BILINEAR)
            external_prob_map = np.array(resized, dtype=np.float32) / 255.0
        alpha, _ = build_hardened_alpha(car_image, external_prob_map, mode="strict")
        alpha_source = alpha
        car_mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        car_rgba_refined = refine_foreground(car_image, alpha)
    else:
        car_mask, car_rgba_refined = models["segmenter"].segment(car_image, alpha_mode="auto")
        alpha_source = np.array(car_mask, dtype=np.float32) / 255.0

    if (
        settings.enable_vitmatte
        and studio_background
        and "vitmatte" in models
        and alpha_source is not None
        and not use_external_cutout
    ):
        trimap = _build_vitmatte_trimap(alpha_source)
        _emit_debug_artifact(
            settings=settings,
            job_id=job_id,
            local_name="01a_trimap",
            remote_key="trimap_png",
            image=trimap,
            debug_put_urls=debug_put_urls,
        )
        try:
            alpha_refined = models["vitmatte"].refine(car_image, alpha_source)
            alpha_source = alpha_refined
            car_mask = Image.fromarray(np.clip(alpha_refined * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            car_rgba_refined = refine_foreground(car_image, alpha_refined)
            _emit_debug_artifact(
                settings=settings,
                job_id=job_id,
                local_name="01b_vitmatte_alpha",
                remote_key="vitmatte_alpha_png",
                image=car_mask,
                debug_put_urls=debug_put_urls,
            )
        except Exception as error:
            logger.warning(f"[{job_id}] ViTMatte refinement failed, using initial alpha: {error}")

    source_bbox = get_tight_bbox_from_mask(car_mask)
    source_checks = compute_mask_artifact_checks(car_mask, source_bbox)
    if _mask_checks_source_fail(source_checks):
        logger.warning(
            f"[{job_id}] Source mask quality is weak "
            f"(interior={source_checks['interiorOpaqueRatio']:.4f}, "
            f"outside={source_checks['outsideLeakMeanAlpha']:.4f}, "
            f"near={source_checks['nearLeakMeanAlpha']:.4f}, "
            f"area={source_checks['maskAreaRatio']:.4f}); retrying strict alpha mode."
        )
        if use_external_cutout:
            if external_cutout_rgba is None:
                raise InvalidInputError("car_cutout_url", "External cutout was requested but not loaded.")
            cutout_alpha = np.array(external_cutout_rgba.getchannel("A"), dtype=np.float32) / 255.0
            alpha = sanitize_external_alpha(cutout_alpha)
            alpha_source = alpha
            car_mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            cutout_rgb_np = np.array(external_cutout_rgba.convert("RGB"), dtype=np.uint8)
            cutout_rgb_np[alpha <= 0.01] = 0
            car_rgba_refined = Image.fromarray(
                np.dstack((cutout_rgb_np, np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8))),
                mode="RGBA",
            )
        elif external_prob_map is not None:
            alpha, _ = build_hardened_alpha(car_image, external_prob_map, mode="strict")
            alpha_source = alpha
            car_mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            car_rgba_refined = refine_foreground(car_image, alpha)
        else:
            car_mask, car_rgba_refined = models["segmenter"].segment(car_image, alpha_mode="strict")
            alpha_source = np.array(car_mask, dtype=np.float32) / 255.0
        source_bbox = get_tight_bbox_from_mask(car_mask)
        source_checks = compute_mask_artifact_checks(car_mask, source_bbox)
        if _mask_checks_source_fail(source_checks):
            raise InvalidInputError(
                "car",
                (
                    "Mask quality failed after strict retry "
                    f"(interior={source_checks['interiorOpaqueRatio']:.4f}, "
                    f"outside={source_checks['outsideLeakMeanAlpha']:.4f}, "
                    f"near={source_checks['nearLeakMeanAlpha']:.4f}, "
                    f"area={source_checks['maskAreaRatio']:.4f})."
                ),
            )
    timings["segmentation_s"] = round(_now_s() - t0, 3)
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="01_mask",
        remote_key="mask_png",
        image=car_mask,
        debug_put_urls=debug_put_urls,
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="02_foreground_rgba",
        remote_key="foreground_rgba_png",
        image=car_rgba_refined,
        debug_put_urls=debug_put_urls,
    )

    tight_bbox = source_bbox
    (
        composite_raw,
        placement_bbox,
        placed_mask,
        placed_foreground_rgb,
        placement_meta,
    ) = place_car_on_background(
        car_rgba_refined=car_rgba_refined,
        tight_bbox=tight_bbox,
        bg_image=bg_proc,
        studio_background=studio_background,
        studio_car_width_ratio=settings.studio_car_width_ratio,
        studio_turntable_coverage=settings.studio_turntable_coverage,
        studio_ground_ratio=settings.studio_ground_ratio,
        studio_ground_bias_px=settings.studio_ground_bias_px,
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="04_placed_mask",
        remote_key="placed_mask_png",
        image=placed_mask,
        debug_put_urls=debug_put_urls,
    )
    foreground_mask = paste_mask_into_background(bg_proc.size, placement_bbox, placed_mask)
    foreground_alpha_np = np.array(foreground_mask, dtype=np.float32) / 255.0
    edge_band_np = np.zeros_like(foreground_alpha_np, dtype=np.uint8)
    edge_band_np[np.logical_and(foreground_alpha_np > 0.02, foreground_alpha_np < 0.45)] = 255
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="04a_edge_band",
        remote_key="edge_band_png",
        image=Image.fromarray(edge_band_np, mode="L"),
        debug_put_urls=debug_put_urls,
    )
    composite_raw = defringe_to_target_background(
        composite_image=composite_raw,
        background_image=bg_proc,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
        edge_alpha_max=0.35 if studio_background else 0.65,
    )
    fringe_stats = compute_composite_fringe_stats(
        composite_image=composite_raw,
        background_image=bg_proc,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="03_composite_raw",
        remote_key="composite_raw_jpg",
        image=composite_raw,
        debug_put_urls=debug_put_urls,
    )
    placement_overlay = render_placement_overlay(
        background=bg_proc,
        placement_bbox=placement_bbox,
        alignment=placement_meta.get("alignment"),
        strict_bottom_local=int(placement_meta.get("strictBottomLocal", 0)),
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="04_placement_overlay",
        remote_key="placement_overlay_jpg",
        image=placement_overlay,
        debug_put_urls=debug_put_urls,
    )
    mask_checks = compute_mask_artifact_checks(foreground_mask, placement_bbox)
    mask_quality_bad = _mask_checks_guidance_risky(mask_checks) or _fringe_risky(fringe_stats, settings)

    logger.info(f"[{job_id}] Step 2: ControlCom harmonization...")
    t0 = _now_s()
    harmonization_mode = settings.harmonization_mode
    if harmonization_mode not in {"auto", "controlcom", "lowfreq", "off"}:
        harmonization_mode = "auto"
    harmonization_method = "controlcom_multiband"
    harmonization_diag: Dict[str, float] = {"protectCoverageRatio": 0.0}
    controlcom_error: str | None = None
    if harmonization_mode == "off":
        harmonization_method = "identity_preserve"
        composite_harmonized = composite_raw
    elif mask_quality_bad:
        controlcom_error = (
            f"Mask quality check failed: area={mask_checks['maskAreaRatio']:.4f}, "
            f"interior={mask_checks['interiorOpaqueRatio']:.4f}, "
            f"outsideLeak={mask_checks['outsideLeakMeanAlpha']:.4f}, "
            f"nearLeak={mask_checks['nearLeakMeanAlpha']:.4f}, "
            f"fringeMean={fringe_stats['fringeRgbMean']:.3f}, "
            f"fringeP95={fringe_stats['fringeRgbP95']:.3f}"
        )
        logger.warning(
            f"[{job_id}] {controlcom_error}. "
            "Skipping ControlCom guidance and preserving identity composite."
        )
        harmonization_method = "identity_preserve"
        composite_harmonized = composite_raw
    else:
        try:
            composite_guidance = models["harmonizer"].harmonize(
                background_image=bg_proc,
                fg_crop=placed_foreground_rgb,
                fg_mask_crop=placed_mask,
                placement_bbox=placement_bbox,
            )
            _emit_debug_artifact(
                settings=settings,
                job_id=job_id,
                local_name="05_controlcom_guidance",
                remote_key="controlcom_guidance_jpg",
                image=composite_guidance,
                debug_put_urls=debug_put_urls,
            )
            use_lowfreq = harmonization_mode == "lowfreq" or (
                harmonization_mode == "auto" and studio_background
            )
            if use_lowfreq:
                harmonization_method = "controlcom_lowfreq"
                composite_harmonized = apply_low_frequency_harmonization(
                    original_composite=composite_raw,
                    harmonized_guidance=composite_guidance,
                    foreground_mask=foreground_mask,
                    foreground_bbox=placement_bbox,
                )
                harmonization_diag = {"protectCoverageRatio": 0.0}
            else:
                harmonization_method = "controlcom_multiband"
                composite_harmonized, harmonization_diag = apply_multiband_harmonization(
                    original_composite=composite_raw,
                    harmonized_guidance=composite_guidance,
                    foreground_mask=foreground_mask,
                    foreground_bbox=placement_bbox,
                )
        except Exception as error:
            controlcom_error = str(error)
            logger.warning(
                f"[{job_id}] ControlCom harmonization failed. Preserving identity composite: "
                f"{controlcom_error}"
            )
            harmonization_method = "identity_preserve"
            composite_harmonized = composite_raw

    detail_ratio = compute_detail_preservation_ratio(
        baseline_image=composite_raw,
        candidate_image=composite_harmonized,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    if harmonization_method.startswith("controlcom") and detail_ratio < 0.90:
        logger.warning(
            f"[{job_id}] Detail preservation ratio dropped to {detail_ratio:.4f}. "
            "Switching to identity-preserving fallback."
        )
        harmonization_method = "identity_preserve"
        composite_harmonized = composite_raw
        harmonization_diag = {"protectCoverageRatio": 0.0}
        detail_ratio = compute_detail_preservation_ratio(
            baseline_image=composite_raw,
            candidate_image=composite_harmonized,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
        )
    edge_halo_after_harmonization = compute_edge_halo_stats(
        baseline_image=composite_raw,
        candidate_image=composite_harmonized,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    if (
        harmonization_method.startswith("controlcom")
        and (
            edge_halo_after_harmonization["edgeHaloMeanDelta"] > settings.max_edge_halo_mean_delta
            or edge_halo_after_harmonization["edgeBandWidthPx"] > settings.max_edge_band_width_px
        )
    ):
        logger.warning(
            f"[{job_id}] Harmonization introduced edge halos "
            f"(delta={edge_halo_after_harmonization['edgeHaloMeanDelta']:.3f}, "
            f"band={edge_halo_after_harmonization['edgeBandWidthPx']:.3f}). "
            "Switching to identity-preserving fallback."
        )
        harmonization_method = "identity_preserve"
        composite_harmonized = composite_raw
        harmonization_diag = {"protectCoverageRatio": 0.0}
        detail_ratio = compute_detail_preservation_ratio(
            baseline_image=composite_raw,
            candidate_image=composite_harmonized,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
        )

    timings["harmonization_s"] = round(_now_s() - t0, 3)
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="06_harmonized",
        remote_key="harmonized_jpg",
        image=composite_harmonized,
        debug_put_urls=debug_put_urls,
    )

    final = composite_harmonized
    harmony_score: float | None = None
    quality: str | None = None
    contact_shadow_applied = False
    contact_shadow_mask = Image.new("L", final.size, 0)
    glass_mode_applied = "off"
    glass_backend_applied = "none"
    glass_candidate_mask = Image.new("L", final.size, 0)
    glass_render = Image.new("RGB", final.size, (0, 0, 0))
    studio_mode_applied = "off"

    if variant == "core":
        shadow_mode = settings.contact_shadow_mode
        shadow_strength = settings.core_contact_shadow_strength
        if studio_background and shadow_mode == "v3":
            shadow_mode = "v2"
            shadow_strength = min(shadow_strength, 0.24)
        try:
            final, contact_shadow_applied, contact_shadow_mask = apply_contact_shadow(
                image=final,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
                strength=shadow_strength,
                mode=shadow_mode,
                return_shadow_mask=True,
            )
        except Exception as error:
            logger.warning(f"[{job_id}] Contact shadow generation failed: {error}")
            contact_shadow_applied = False
            contact_shadow_mask = Image.new("L", final.size, 0)
        _emit_debug_artifact(
            settings=settings,
            job_id=job_id,
            local_name="07_shadow_mask",
            remote_key="shadow_mask_png",
            image=contact_shadow_mask,
            debug_put_urls=debug_put_urls,
        )

    if variant == "full":
        logger.info(f"[{job_id}] Step 3: GPSDiffusion shadow (libcom)...")
        t0 = _now_s()
        shadow_full = models["shadow"].generate(final, foreground_mask)
        final = blend_background_only(final, shadow_full, foreground_mask, alpha=shadow_strength)
        timings["shadow_s"] = round(_now_s() - t0, 3)

        logger.info(f"[{job_id}] Step 4: Reflection (libcom)...")
        t0 = _now_s()
        reflection_full = models["reflection"].generate(final, foreground_mask)
        final = blend_background_only(final, reflection_full, foreground_mask, alpha=reflection_strength)
        timings["reflection_s"] = round(_now_s() - t0, 3)

    effective_glass_mode = settings.glass_normalization_mode
    if settings.studio_mode == "on":
        studio_mode_applied = "on"
        if effective_glass_mode == "off":
            effective_glass_mode = "force"
    elif settings.studio_mode == "auto" and studio_background:
        studio_mode_applied = "auto"
        if effective_glass_mode == "off":
            effective_glass_mode = "auto"

    glass_override_mask: Image.Image | None = None
    if settings.glass_mode in {"sam2_auto", "sam2_force"} and "sam2_glass" in models:
        try:
            candidate = models["sam2_glass"].segment(
                image=final,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
            )
            if np.array(candidate, dtype=np.uint8).sum() > 0:
                glass_override_mask = candidate
                glass_backend_applied = "sam2"
            elif settings.glass_mode == "sam2_force":
                logger.warning(
                    f"[{job_id}] SAM2 glass mode forced but no candidate mask produced; using legacy mode."
                )
                glass_backend_applied = "legacy"
            else:
                glass_backend_applied = "legacy"
        except Exception as error:
            logger.warning(f"[{job_id}] SAM2 glass segmentation failed: {error}")
            glass_backend_applied = "legacy"
    else:
        glass_backend_applied = "legacy" if effective_glass_mode in {"auto", "force"} else "none"

    if effective_glass_mode in {"auto", "force"}:
        (
            final,
            glass_applied,
            glass_candidate_mask,
            glass_render,
        ) = apply_glass_normalization(
            image=final,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
            mode=effective_glass_mode,
            return_candidate_mask=True,
            candidate_mask_override=glass_override_mask,
            return_glass_render=True,
        )
        if glass_applied:
            glass_mode_applied = effective_glass_mode
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="08_glass_mask",
        remote_key="glass_mask_png",
        image=glass_candidate_mask,
        debug_put_urls=debug_put_urls,
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="08b_glass_render",
        remote_key="glass_render_jpg",
        image=glass_render,
        debug_put_urls=debug_put_urls,
    )

    if variant == "full":
        logger.info(f"[{job_id}] Step 5: BargainNet QC (HarmonyScoreModel)...")
        t0 = _now_s()
        harmony_score = float(models["scorer"].score(final))
        timings["scoring_s"] = round(_now_s() - t0, 3)
        harmony_score_rounded = round(harmony_score, 4)

        logger.info(
            f"[{job_id}] Harmony score={harmony_score_rounded} threshold={harmony_threshold} "
            f"(shadow={shadow_strength} reflection={reflection_strength})"
        )

        if harmony_score < harmony_threshold:
            raise HarmonyScoreTooLowError(
                score=harmony_score_rounded, guidance=generate_reshoot_guidance(harmony_score_rounded)
            )

        quality = "excellent" if harmony_score >= 0.75 else "acceptable"
        harmony_score = harmony_score_rounded

    detail_ratio = compute_detail_preservation_ratio(
        baseline_image=composite_raw,
        candidate_image=final,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    detail_preservation: Dict[str, Any] = {
        "hfRatio": round(float(detail_ratio), 4),
        "method": harmonization_method,
    }
    if controlcom_error and not harmonization_method.startswith("controlcom"):
        detail_preservation["fallbackReason"] = controlcom_error[:320]

    edge_halo_stats = compute_edge_halo_stats(
        baseline_image=composite_raw,
        candidate_image=final,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    final_fringe_stats = compute_composite_fringe_stats(
        composite_image=final,
        background_image=bg_proc,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )

    artifact_checks: Dict[str, Any] = {
        "interiorOpaqueRatio": round(float(mask_checks["interiorOpaqueRatio"]), 4),
        "outsideLeakMeanAlpha": round(float(mask_checks["outsideLeakMeanAlpha"]), 6),
        "nearLeakMeanAlpha": round(float(mask_checks["nearLeakMeanAlpha"]), 6),
        "nearLeakP95Alpha": round(float(mask_checks["nearLeakP95Alpha"]), 6),
        "maskAreaRatio": round(float(mask_checks["maskAreaRatio"]), 4),
        "rawFringeRgbMean": round(float(fringe_stats["fringeRgbMean"]), 4),
        "rawFringeRgbP95": round(float(fringe_stats["fringeRgbP95"]), 4),
        "fringeRgbMean": round(float(final_fringe_stats["fringeRgbMean"]), 4),
        "fringeRgbP95": round(float(final_fringe_stats["fringeRgbP95"]), 4),
        "edgeHaloMeanDelta": round(float(edge_halo_stats["edgeHaloMeanDelta"]), 4),
        "edgeBandWidthPx": round(float(edge_halo_stats["edgeBandWidthPx"]), 4),
        "protectCoverageRatio": round(float(harmonization_diag.get("protectCoverageRatio", 0.0)), 4),
        "contactShadowApplied": contact_shadow_applied,
        "contactShadowModeUsed": shadow_mode if variant == "core" else "none",
        "contactShadowStrengthUsed": round(float(shadow_strength), 3) if variant == "core" else 0.0,
        "glassModeApplied": glass_mode_applied,
        "glassBackendApplied": glass_backend_applied,
        "studioModeApplied": studio_mode_applied,
    }
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="07_final",
        remote_key="final_jpg",
        image=final,
        debug_put_urls=debug_put_urls,
    )

    logger.info(f"[{job_id}] Uploading output...")
    t0 = _now_s()
    upload_image_put(output_put_url, final.convert("RGB"))
    timings["upload_s"] = round(_now_s() - t0, 3)

    total = round(sum(timings.values()), 3)

    output: Dict[str, Any] = {
        "status": "success",
        "jobId": job_id,
        "variant": variant,
        "timings": timings,
        "total_processing_s": total,
        "artifactChecks": artifact_checks,
        "workerBuildId": settings.worker_build_id,
    }

    if harmony_score is not None:
        output["harmonyScore"] = harmony_score
    if quality is not None:
        output["quality"] = quality
    output["detailPreservation"] = detail_preservation
    return output
