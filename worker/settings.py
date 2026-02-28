from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_cache_dir: str
    hf_home: str
    birefnet_repo_id: str
    birefnet_infer_res: int
    pipeline_variant: str
    controlcom_ckpt: str
    clip_model_dir: str
    controlcom_repo_dir: str
    controlcom_timeout_s: int
    max_pixels: int
    max_output_long_edge: int
    output_resize_mode: str
    core_contact_shadow_strength: float
    contact_shadow_mode: str
    glass_normalization_mode: str
    glass_mode: str
    studio_mode: str
    studio_car_width_ratio: float
    studio_turntable_coverage: float
    studio_ground_ratio: float
    studio_ground_bias_px: int
    target_width: int
    target_height: int
    harmonization_mode: str
    enable_vitmatte: bool
    vitmatte_model_id: str
    sam2_model_id: str
    max_edge_halo_mean_delta: float
    max_edge_band_width_px: float
    max_fringe_rgb_mean: float
    max_fringe_rgb_p95: float
    debug_artifacts: bool
    worker_build_id: str


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}



def get_settings() -> Settings:
    return Settings(
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models"),
        hf_home=os.getenv("HF_HOME", "/runpod-volume/hf_cache"),
        birefnet_repo_id=os.getenv("BIREFNET_REPO_ID", "ZhengPeng7/BiRefNet_HR-matting"),
        birefnet_infer_res=int(os.getenv("BIREFNET_INFER_RES", "2048")),
        pipeline_variant=os.getenv("PIPELINE_VARIANT", "core").lower(),
        controlcom_ckpt=os.getenv(
            "CONTROLCOM_CKPT",
            "/runpod-volume/models/controlcom/ControlCom_blend_harm.pth",
        ),
        clip_model_dir=os.getenv(
            "CLIP_MODEL_DIR",
            "/runpod-volume/models/controlcom/openai-clip-vit-large-patch14",
        ),
        controlcom_repo_dir=os.getenv("CONTROLCOM_REPO_DIR", "/app/vendor/controlcom"),
        controlcom_timeout_s=int(os.getenv("CONTROLCOM_TIMEOUT_S", "600")),
        max_pixels=int(os.getenv("MAX_IMAGE_PIXELS", str(40_000_000))),
        max_output_long_edge=int(os.getenv("MAX_OUTPUT_LONG_EDGE", "2048")),
        output_resize_mode=os.getenv("OUTPUT_RESIZE_MODE", "preserve").lower(),
        core_contact_shadow_strength=float(os.getenv("CORE_CONTACT_SHADOW_STRENGTH", "0.32")),
        contact_shadow_mode=os.getenv("CONTACT_SHADOW_MODE", "v3").lower(),
        glass_normalization_mode=os.getenv("GLASS_NORMALIZATION_MODE", "off").lower(),
        glass_mode=os.getenv("GLASS_MODE", "sam2_auto").lower(),
        studio_mode=os.getenv("STUDIO_MODE", "auto").lower(),
        studio_car_width_ratio=float(os.getenv("STUDIO_CAR_WIDTH_RATIO", "0.82")),
        studio_turntable_coverage=float(os.getenv("STUDIO_TURNTABLE_COVERAGE", "0.88")),
        studio_ground_ratio=float(os.getenv("STUDIO_GROUND_RATIO", "0.90")),
        studio_ground_bias_px=int(os.getenv("STUDIO_GROUND_BIAS_PX", "-6")),
        target_width=int(os.getenv("TARGET_WIDTH", "1920")),
        target_height=int(os.getenv("TARGET_HEIGHT", "1280")),
        harmonization_mode=os.getenv("HARMONIZATION_MODE", "auto").lower(),
        enable_vitmatte=_is_truthy(os.getenv("ENABLE_VITMATTE", "true")),
        vitmatte_model_id=os.getenv("VITMATTE_MODEL_ID", "hustvl/vitmatte-small-composition-1k"),
        sam2_model_id=os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-small"),
        max_edge_halo_mean_delta=float(os.getenv("MAX_EDGE_HALO_MEAN_DELTA", "14.0")),
        max_edge_band_width_px=float(os.getenv("MAX_EDGE_BAND_WIDTH_PX", "7.5")),
        max_fringe_rgb_mean=float(os.getenv("MAX_FRINGE_RGB_MEAN", "2.0")),
        max_fringe_rgb_p95=float(os.getenv("MAX_FRINGE_RGB_P95", "8.0")),
        debug_artifacts=_is_truthy(os.getenv("DEBUG_ARTIFACTS")),
        worker_build_id=os.getenv("WORKER_BUILD_ID", "unknown"),
    )
