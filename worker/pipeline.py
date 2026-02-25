from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from loguru import logger
from PIL import Image

from exceptions import HarmonyScoreTooLowError, InvalidInputError, ModelInferenceError
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
    detect_ground_plane,
    get_tight_bbox_from_mask,
    paste_mask_into_background,
    restore_high_freq_details,
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
        _models["segmenter"] = BiRefNetSegmenter(cache_dir / "birefnet")
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
) -> tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    """
    Paste refined RGBA car onto background, sized to fill ~70% of background width and
    aligned to "ground" near ~85% of background height.

    Returns:
      composite_raw: RGB composite before ControlCom
      placement_bbox: (x1,y1,x2,y2) in background coordinates
      placed_mask: L mask resized to placement size (same as bbox width/height)
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

    car_crop = car_rgba_refined.crop(tight_bbox)
    car_placed = car_crop.resize((car_w, car_h), Image.Resampling.LANCZOS)
    placed_mask = car_placed.split()[-1].convert("L")

    x = (bg_w - car_w) // 2
    y = int(bg_h * 0.85) - car_h
    y = max(0, min(y, bg_h - car_h))

    canvas = bg_image.copy().convert("RGBA")
    canvas.paste(car_placed, (x, y), car_placed)
    return canvas.convert("RGB"), (x, y, x + car_w, y + car_h), placed_mask


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

    bg_proc = fit_background(background_image, (settings.target_width, settings.target_height))

    logger.info(f"[{job_id}] Step 1: BiRefNet segmentation + edge refinement...")
    t0 = _now_s()
    car_mask, car_rgba_refined = models["segmenter"].segment(car_image)
    timings["segmentation_s"] = round(_now_s() - t0, 3)

    tight_bbox = get_tight_bbox_from_mask(car_mask)
    fg_crop_rgb = car_image.crop(tight_bbox).convert("RGB")
    fg_mask_crop = car_mask.crop(tight_bbox).convert("L")

    composite_raw, placement_bbox, placed_mask = place_car_on_background(
        car_rgba_refined=car_rgba_refined, tight_bbox=tight_bbox, bg_image=bg_proc
    )

    logger.info(f"[{job_id}] Step 2: ControlCom harmonization...")
    t0 = _now_s()
    try:
        composite_harmonized = models["harmonizer"].harmonize(
            background_image=bg_proc,
            fg_crop=fg_crop_rgb,
            fg_mask_crop=fg_mask_crop,
            placement_bbox=placement_bbox,
        )
    except ModelInferenceError:
        raise
    except Exception as error:
        raise ModelInferenceError("ControlCom", error) from error

    composite_harmonized = restore_high_freq_details(
        composite_raw, composite_harmonized, foreground_bbox=placement_bbox, blend_alpha=0.25
    )
    timings["harmonization_s"] = round(_now_s() - t0, 3)

    final = composite_harmonized
    harmony_score: float | None = None
    quality: str | None = None

    if variant == "full":
        foreground_mask = paste_mask_into_background(bg_proc.size, placement_bbox, placed_mask)

        logger.info(f"[{job_id}] Step 3: GPSDiffusion shadow (libcom)...")
        t0 = _now_s()
        shadow_full = models["shadow"].generate(final, foreground_mask)
        final = Image.blend(final, shadow_full, alpha=shadow_strength)
        timings["shadow_s"] = round(_now_s() - t0, 3)

        logger.info(f"[{job_id}] Step 4: Reflection (libcom)...")
        t0 = _now_s()
        ground_mask = detect_ground_plane(final)
        reflection_full = models["reflection"].generate(final, foreground_mask, ground_mask)
        final = Image.blend(final, reflection_full, alpha=reflection_strength)
        timings["reflection_s"] = round(_now_s() - t0, 3)

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
    }

    if harmony_score is not None:
        output["harmonyScore"] = harmony_score
    if quality is not None:
        output["quality"] = quality

    return output

