from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from loguru import logger
import numpy as np
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
    upload_debug_image_put,
    upload_image_put,
    validate_image,
)
from utils.image_ops import (
    apply_contact_shadow,
    apply_glass_normalization,
    apply_multiband_harmonization,
    blend_background_only,
    compute_edge_halo_stats,
    compute_mask_artifact_checks,
    compute_detail_preservation_ratio,
    get_tight_bbox_from_mask,
    is_studio_background,
    paste_mask_into_background,
    reharden_resized_alpha,
    resize_rgba_premultiplied,
)

_models: Dict[str, Any] = {}
_DEBUG_ARTIFACT_CONTENT_TYPES: dict[str, str] = {
    "mask_png": "image/png",
    "foreground_rgba_png": "image/png",
    "placed_mask_png": "image/png",
    "composite_raw_jpg": "image/jpeg",
    "controlcom_guidance_jpg": "image/jpeg",
    "harmonized_jpg": "image/jpeg",
    "final_jpg": "image/jpeg",
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


def get_models(settings: Settings, *, variant: str) -> Dict[str, Any]:
    global _models

    if "segmenter" not in _models:
        cache_dir = Path(settings.model_cache_dir)
        logger.info("Loading BiRefNet + ControlCom (cold start)...")
        _models["segmenter"] = BiRefNetSegmenter(
            cache_dir / "birefnet",
            infer_res=settings.birefnet_infer_res,
            repo_id=settings.birefnet_repo_id,
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
    studio_background: bool,
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

    width_ratio = 0.82 if studio_background else 0.70
    ground_ratio = 0.90 if studio_background else 0.85

    car_w = int(bg_w * width_ratio)
    car_h = int(car_w / max(aspect, 1e-6))
    if car_h > int(bg_h * 0.80):
        car_h = int(bg_h * 0.80)
        car_w = int(car_h * aspect)

    car_crop_rgba = car_rgba_refined.crop(tight_bbox)
    resized_rgb, resized_alpha = resize_rgba_premultiplied(car_crop_rgba, (car_w, car_h))
    placed_mask = reharden_resized_alpha(resized_alpha)
    alpha_np = np.array(placed_mask, dtype=np.float32) / 255.0
    rgb_np = np.array(resized_rgb, dtype=np.uint8)
    rgb_np[alpha_np <= 0.01] = 0
    placed_foreground_rgb = Image.fromarray(rgb_np, mode="RGB")
    car_placed_rgba = Image.merge("RGBA", (*placed_foreground_rgb.split(), placed_mask))

    x = (bg_w - car_w) // 2
    y = int(bg_h * ground_ratio) - car_h
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
    debug_put_urls_input = payload.get("debug_put_urls")
    debug_put_urls: dict[str, str] = {}
    if isinstance(debug_put_urls_input, dict):
        for key, value in debug_put_urls_input.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                debug_put_urls[key] = value

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
    studio_background = is_studio_background(bg_proc)

    logger.info(f"[{job_id}] Step 1: BiRefNet segmentation + edge refinement...")
    t0 = _now_s()
    car_mask, car_rgba_refined = models["segmenter"].segment(car_image, alpha_mode="auto")
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
        car_mask, car_rgba_refined = models["segmenter"].segment(car_image, alpha_mode="strict")
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
    composite_raw, placement_bbox, placed_mask, placed_foreground_rgb = place_car_on_background(
        car_rgba_refined=car_rgba_refined,
        tight_bbox=tight_bbox,
        bg_image=bg_proc,
        studio_background=studio_background,
    )
    _emit_debug_artifact(
        settings=settings,
        job_id=job_id,
        local_name="03_composite_raw",
        remote_key="composite_raw_jpg",
        image=composite_raw,
        debug_put_urls=debug_put_urls,
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
    mask_checks = compute_mask_artifact_checks(foreground_mask, placement_bbox)
    mask_quality_bad = _mask_checks_guidance_risky(mask_checks)

    logger.info(f"[{job_id}] Step 2: ControlCom harmonization...")
    t0 = _now_s()
    harmonization_method = "controlcom_multiband"
    harmonization_diag: Dict[str, float] = {"protectCoverageRatio": 0.0}
    controlcom_error: str | None = None
    if mask_quality_bad:
        controlcom_error = (
            f"Mask quality check failed: area={mask_checks['maskAreaRatio']:.4f}, "
            f"interior={mask_checks['interiorOpaqueRatio']:.4f}, "
            f"outsideLeak={mask_checks['outsideLeakMeanAlpha']:.4f}, "
            f"nearLeak={mask_checks['nearLeakMeanAlpha']:.4f}"
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
    if harmonization_method == "controlcom_multiband" and detail_ratio < 0.90:
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
        harmonization_method == "controlcom_multiband"
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
    glass_mode_applied = "off"
    studio_mode_applied = "off"

    if variant == "core":
        try:
            final, contact_shadow_applied = apply_contact_shadow(
                image=final,
                foreground_mask=foreground_mask,
                foreground_bbox=placement_bbox,
                strength=settings.core_contact_shadow_strength,
                mode=settings.contact_shadow_mode,
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

    effective_glass_mode = settings.glass_normalization_mode
    if settings.studio_mode == "on":
        studio_mode_applied = "on"
        if effective_glass_mode == "off":
            effective_glass_mode = "force"
    elif settings.studio_mode == "auto" and studio_background:
        studio_mode_applied = "auto"
        if effective_glass_mode == "off":
            effective_glass_mode = "auto"

    if effective_glass_mode in {"auto", "force"}:
        final, glass_applied = apply_glass_normalization(
            image=final,
            foreground_mask=foreground_mask,
            foreground_bbox=placement_bbox,
            mode=effective_glass_mode,
        )
        if glass_applied:
            glass_mode_applied = effective_glass_mode

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
    if controlcom_error and harmonization_method != "controlcom_multiband":
        detail_preservation["fallbackReason"] = controlcom_error[:320]

    edge_halo_stats = compute_edge_halo_stats(
        baseline_image=composite_raw,
        candidate_image=final,
        foreground_mask=foreground_mask,
        foreground_bbox=placement_bbox,
    )

    artifact_checks: Dict[str, Any] = {
        "interiorOpaqueRatio": round(float(mask_checks["interiorOpaqueRatio"]), 4),
        "outsideLeakMeanAlpha": round(float(mask_checks["outsideLeakMeanAlpha"]), 6),
        "nearLeakMeanAlpha": round(float(mask_checks["nearLeakMeanAlpha"]), 6),
        "nearLeakP95Alpha": round(float(mask_checks["nearLeakP95Alpha"]), 6),
        "maskAreaRatio": round(float(mask_checks["maskAreaRatio"]), 4),
        "edgeHaloMeanDelta": round(float(edge_halo_stats["edgeHaloMeanDelta"]), 4),
        "edgeBandWidthPx": round(float(edge_halo_stats["edgeBandWidthPx"]), 4),
        "protectCoverageRatio": round(float(harmonization_diag.get("protectCoverageRatio", 0.0)), 4),
        "contactShadowApplied": contact_shadow_applied,
        "glassModeApplied": glass_mode_applied,
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
