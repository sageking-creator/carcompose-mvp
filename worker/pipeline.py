from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from loguru import logger
from PIL import Image

from exceptions import HarmonyScoreTooLowError, InvalidInputError
from models.birefnet import BiRefNetSegmenter
from models.bargainnet import BargainNetScorer
from models.controlcom import ControlComHarmonizer
from models.libcom_reflection import LibcomReflectionGenerator
from models.libcom_shadow import LibcomShadowGenerator
from settings import Settings
from utils.image import (
    download_image,
    fit_background,
    generate_reshoot_guidance,
    upload_image_put,
    validate_image,
)
from utils.image_ops import (
    apply_contact_shadow,
    apply_glass_normalization,
    apply_low_frequency_harmonization,
    apply_luminance_transfer_fallback,
    blend_background_only,
    compute_mask_artifact_checks,
    compute_detail_preservation_ratio,
    get_tight_bbox_from_mask,
    paste_mask_into_background,
)

_models: Dict[str, Any] = {}


def _now_s() -> float:
    return time.perf_counter()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_models(settings: Settings, *, variant: str) -> Dict[str, Any]:
    global _models

    if "segmenter" not in _models:
        cache_dir = Path(settings.model_cache_dir)
        logger.info("Loading BiRefNet + ControlCom (cold start)...")
        _models["segmenter"] = BiRefNetSegmenter(
            cache_dir / "birefnet",
            max_side=settings.birefnet_max_side,
        )
        _models["harmonizer"] = ControlComHarmonizer(
            repo_dir=Path(settings.controlcom_repo_dir),
            ckpt_path=Path(settings.controlcom_ckpt),
            clip_dir=Path(settings.clip_model_dir),
            timeout_s=settings.controlcom_timeout_s,
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
) -> tuple[Image.Image, Tuple[int, int, int, int], Image.Image, Image.Image]:
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

    car_w = int(bg_w * 0.70)
    car_h = int(car_w / max(aspect, 1e-6))
    if car_h > int(bg_h * 0.80):
        car_h = int(bg_h * 0.80)
        car_w = int(car_h * aspect)

    car_crop_rgba = car_rgba_refined.crop(tight_bbox)
    car_placed_rgba = car_crop_rgba.resize((car_w, car_h), Image.Resampling.LANCZOS)
    placed_mask = car_placed_rgba.split()[-1].convert("L")
    placed_foreground_rgb = car_placed_rgba.convert("RGB")

    x = (bg_w - car_w) // 2
    y = int(bg_h * 0.85) - car_h
    y = max(0, min(y, bg_h - car_h))

    canvas = bg_image.copy().convert("RGBA")
    canvas.paste(car_placed_rgba, (x, y), car_placed_rgba)
    return canvas.convert("RGB"), (x, y, x + car_w, y + car_h), placed_mask, placed_foreground_rgb


def run_pipeline(payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    job_id = payload.get("job_id")
    car_url = payload.get("car_image_url")
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

    models = get_models(settings, variant=variant)
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

    logger.info(f"[{job_id}] Step 1: BiRefNet segmentation + edge refinement...")
    t0 = _now_s()
    car_mask, car_rgba_refined = models["segmenter"].segment(car_image)
    timings["segmentation_s"] = round(_now_s() - t0, 3)

    tight_bbox = get_tight_bbox_from_mask(car_mask)
    composite_raw, placement_bbox, placed_mask, placed_foreground_rgb = place_car_on_background(
        car_rgba_refined=car_rgba_refined, tight_bbox=tight_bbox, bg_image=bg_proc
    )
    foreground_mask = paste_mask_into_background(bg_proc.size, placement_bbox, placed_mask)
    mask_checks = compute_mask_artifact_checks(foreground_mask, placement_bbox)
    mask_quality_bad = (
        mask_checks["maskAreaRatio"] < 0.005
        or mask_checks["maskAreaRatio"] > 0.85
        or mask_checks["interiorOpaqueRatio"] < 0.985
        or mask_checks["outsideLeakMeanAlpha"] > 0.01
    )

    logger.info(f"[{job_id}] Step 2: ControlCom harmonization...")
    t0 = _now_s()
    harmonization_method = "controlcom_lf"
    controlcom_error: str | None = None
    if mask_quality_bad:
        controlcom_error = (
            f"Mask quality check failed: area={mask_checks['maskAreaRatio']:.4f}, "
            f"interior={mask_checks['interiorOpaqueRatio']:.4f}, "
            f"outsideLeak={mask_checks['outsideLeakMeanAlpha']:.4f}"
        )
        logger.warning(
            f"[{job_id}] {controlcom_error}. "
            "Skipping ControlCom guidance and falling back to deterministic luminance transfer."
        )
        harmonization_method = "lab_transfer"
        composite_harmonized = apply_luminance_transfer_fallback(
            image=composite_raw,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
        )
    else:
        try:
            composite_guidance = models["harmonizer"].harmonize(
                background_image=bg_proc,
                fg_crop=placed_foreground_rgb,
                fg_mask_crop=placed_mask,
                placement_bbox=placement_bbox,
            )
            composite_harmonized = apply_low_frequency_harmonization(
                original_composite=composite_raw,
                harmonized_guidance=composite_guidance,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
            )
        except Exception as error:
            controlcom_error = str(error)
            logger.warning(
                f"[{job_id}] ControlCom harmonization failed. Falling back to deterministic luminance transfer: "
                f"{controlcom_error}"
            )
            harmonization_method = "lab_transfer"
            composite_harmonized = apply_luminance_transfer_fallback(
                image=composite_raw,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
            )

    detail_ratio = compute_detail_preservation_ratio(
        baseline_image=composite_raw,
        candidate_image=composite_harmonized,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )
    if harmonization_method == "controlcom_lf" and detail_ratio < 0.85:
        logger.warning(
            f"[{job_id}] Detail preservation ratio dropped to {detail_ratio:.4f}. "
            "Switching to deterministic luminance-transfer fallback."
        )
        harmonization_method = "lab_transfer"
        composite_harmonized = apply_luminance_transfer_fallback(
            image=composite_raw,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
        )
        detail_ratio = compute_detail_preservation_ratio(
            baseline_image=composite_raw,
            candidate_image=composite_harmonized,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
        )

    timings["harmonization_s"] = round(_now_s() - t0, 3)

    final = composite_harmonized
    harmony_score: float | None = None
    quality: str | None = None
    contact_shadow_applied = False
    glass_mode_applied = "off"

    if variant == "core":
        try:
            final, contact_shadow_applied = apply_contact_shadow(
                image=final,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
                strength=settings.core_contact_shadow_strength,
            )
        except Exception as error:
            logger.warning(f"[{job_id}] Contact shadow generation failed: {error}")
            contact_shadow_applied = False

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

    if settings.glass_normalization_mode in {"auto", "force"}:
        final, glass_applied = apply_glass_normalization(
            image=final,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
            mode=settings.glass_normalization_mode,
        )
        if glass_applied:
            glass_mode_applied = settings.glass_normalization_mode

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
    if controlcom_error and harmonization_method == "lab_transfer":
        detail_preservation["fallbackReason"] = controlcom_error[:320]

    artifact_checks: Dict[str, Any] = {
        "interiorOpaqueRatio": round(float(mask_checks["interiorOpaqueRatio"]), 4),
        "outsideLeakMeanAlpha": round(float(mask_checks["outsideLeakMeanAlpha"]), 6),
        "maskAreaRatio": round(float(mask_checks["maskAreaRatio"]), 4),
        "contactShadowApplied": contact_shadow_applied,
        "glassModeApplied": glass_mode_applied,
    }

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
    }

    if harmony_score is not None:
        output["harmonyScore"] = harmony_score
    if quality is not None:
        output["quality"] = quality
    output["detailPreservation"] = detail_preservation
    return output
