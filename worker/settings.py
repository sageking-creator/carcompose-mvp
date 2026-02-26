from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_cache_dir: str
    hf_home: str
    pipeline_variant: str
    controlcom_ckpt: str
    clip_model_dir: str
    controlcom_repo_dir: str
    controlcom_timeout_s: int
    max_pixels: int
    max_output_long_edge: int
    output_resize_mode: str
    target_width: int
    target_height: int



def get_settings() -> Settings:
    return Settings(
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models"),
        hf_home=os.getenv("HF_HOME", "/runpod-volume/hf_cache"),
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
        target_width=int(os.getenv("TARGET_WIDTH", "1920")),
        target_height=int(os.getenv("TARGET_HEIGHT", "1280")),
    )
