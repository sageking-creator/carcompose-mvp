import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download, snapshot_download

from settings import Settings

CONTROL_COM_HF_SOURCES = [
    ("BCMIZB/Libcom_pretrained_models", "ControlCom_blend_harm.pth"),
]
CONTROL_COM_GDRIVE_ID = "1H5tCPJYRHVTLPzfUKGDxuAHXzp3ZBjGU"


def _download_hf(repo_id: str, target_dir: Path, hf_cache_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        return

    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(hf_cache_dir),
        local_dir=str(target_dir),
        local_dir_use_symlinks=True,
        resume_download=True,
        ignore_patterns=["*.md", "*.gitattributes"],
    )


def _download_controlcom(target_path: Path, hf_cache_dir: Path) -> None:
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer HuggingFace-hosted checkpoint (more reliable than GDrive).
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in CONTROL_COM_HF_SOURCES:
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(hf_cache_dir),
                resume_download=True,
            )
            cached_path = Path(cached)
            if cached_path.exists() and cached_path.stat().st_size > 1_000_000:
                if target_path.exists():
                    target_path.unlink()
                try:
                    target_path.symlink_to(cached_path)
                except OSError:
                    shutil.copyfile(cached_path, target_path)
                return
        except Exception:
            continue

    command = [
        "gdown",
        f"https://drive.google.com/uc?id={CONTROL_COM_GDRIVE_ID}",
        "-O",
        str(target_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if target_path.exists():
            target_path.unlink()
        raise RuntimeError(
            "Failed to download ControlCom checkpoint with gdown. "
            f"stdout={result.stdout} stderr={result.stderr}"
        )


def run_download_models(settings: Settings) -> Dict[str, object]:
    os.environ.setdefault("HF_HOME", settings.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", settings.hf_home)
    cache = Path(settings.model_cache_dir)
    hf_cache = Path(settings.hf_home)
    cache.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)

    downloaded: List[str] = []

    # Variant-only init download policy:
    #   core: BiRefNet + ControlCom + CLIP
    #   full: core + libcom models + validate libcom (GPSDiffusion via model_type)
    variant = (settings.pipeline_variant or "core").lower()
    variant = "full" if variant == "full" else "core"

    # BiRefNet (HuggingFace)
    _download_hf("ZhengPeng7/BiRefNet", cache / "birefnet", hf_cache)
    downloaded.append("birefnet")

    # ControlCom CLIP (HuggingFace) — must live under controlcom/openai-clip-vit-large-patch14
    _download_hf(
        "openai/clip-vit-large-patch14",
        Path(settings.clip_model_dir),
        hf_cache,
    )
    downloaded.append("clip")

    controlcom_target = Path(settings.controlcom_ckpt)
    _download_controlcom(controlcom_target, hf_cache)
    downloaded.append("controlcom")

    if variant == "full":
        _download_hf("bcmi/libcom-models", cache / "libcom", hf_cache)
        downloaded.append("libcom-models")

    # Validate runtime + required files before writing sentinel
    import torch
    import transformers  # noqa: F401
    import diffusers  # noqa: F401
    import cv2

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (torch.cuda.is_available() == False).")

    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "OpenCV missing ximgproc module. "
            "Install opencv-contrib-python-headless (not opencv-python-headless)."
        )

    if not controlcom_target.exists() or controlcom_target.stat().st_size < 1_000_000:
        raise RuntimeError(f"ControlCom checkpoint missing or too small: {controlcom_target}")

    clip_dir = Path(settings.clip_model_dir)
    if not clip_dir.exists() or not any(clip_dir.iterdir()):
        raise RuntimeError(f"CLIP model dir missing or empty: {clip_dir}")

    controlcom_script = Path(settings.controlcom_repo_dir) / "scripts" / "inference.py"
    if not controlcom_script.exists():
        raise RuntimeError(
            f"ControlCom inference script not found: {controlcom_script}. "
            "Worker image is missing vendor/controlcom."
        )

    if variant == "full":
        from libcom import HarmonyScoreModel, ReflectionGenerationModel, ShadowGenerationModel

        device = 0 if torch.cuda.is_available() else "cpu"
        ShadowGenerationModel(device=device)
        ReflectionGenerationModel(device=device)
        HarmonyScoreModel(device=device)
        downloaded.append("libcom-validated")

    sentinel = cache.parent / ".download_complete"
    sentinel.write_text("ok", encoding="utf-8")

    total_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
    total_gb = round(total_bytes / 1024**3, 2)

    return {
        "ok": True,
        "downloaded": downloaded,
        "totalGb": total_gb,
    }
