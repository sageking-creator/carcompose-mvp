from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from utils.refine import build_hardened_alpha, refine_foreground


class BiRefNetSegmenter:
    def __init__(self, cache_dir: Path):
        torch.set_float32_matmul_precision("high")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise FileNotFoundError(
                f"BiRefNet snapshot not found at {cache_dir}. Run `download_models` to populate the volume."
            )

        # Load from the on-volume snapshot to avoid runtime re-downloads.
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(cache_dir),
            trust_remote_code=True,
        )
        self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        Returns:
          car_mask:         PIL "L" mask, same size as input. 255=car, 0=bg.
          car_rgba_refined: PIL "RGBA" with refined alpha (guided filter).
        """

        orig_size = image.size
        img_rgb = image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)
        if isinstance(outputs, (tuple, list)):
            pred = outputs[-1]
        else:
            pred = getattr(outputs, "logits", outputs)

        pred = pred.sigmoid().cpu()
        pred_squeezed = pred[0].squeeze()

        prob_1024 = pred_squeezed.numpy().astype(np.float32)
        prob_pil = Image.fromarray(np.clip(prob_1024 * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        prob_resized = np.array(
            prob_pil.resize(orig_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        alpha, _ = build_hardened_alpha(img_rgb, prob_resized, threshold=0.50, guided_radius=45, guided_eps=1e-4)
        mask_pil = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        car_rgba = refine_foreground(img_rgb, alpha)
        return mask_pil, car_rgba
