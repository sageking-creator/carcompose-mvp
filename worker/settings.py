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
    studio_mode: str
    target_width: int
    target_height: int
    max_edge_halo_mean_delta: float
    max_edge_band_width_px: float
    debug_artifacts: bool


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}



def get_settings() -> Settings:
    return Settings(
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models"),
        hf_home=os.getenv("HF_HOME", "/runpod-volume/hf_cache"),
        birefnet_repo_id=os.getenv("BIREFNET_REPO_ID", "ZhengPeng7/BiRefNet_HR"),
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
        contact_shadow_mode=os.getenv("CONTACT_SHADOW_MODE", "v2").lower(),
        glass_normalization_mode=os.getenv("GLASS_NORMALIZATION_MODE", "off").lower(),
        studio_mode=os.getenv("STUDIO_MODE", "auto").lower(),
        target_width=int(os.getenv("TARGET_WIDTH", "1920")),
        target_height=int(os.getenv("TARGET_HEIGHT", "1280")),
        max_edge_halo_mean_delta=float(os.getenv("MAX_EDGE_HALO_MEAN_DELTA", "14.0")),
        max_edge_band_width_px=float(os.getenv("MAX_EDGE_BAND_WIDTH_PX", "7.5")),
        debug_artifacts=_is_truthy(os.getenv("DEBUG_ARTIFACTS")),
    )
