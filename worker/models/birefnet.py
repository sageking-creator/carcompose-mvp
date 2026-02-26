from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from utils.refine import build_hardened_alpha, refine_foreground


GRID_FACTOR_H = 31
GRID_FACTOR_W = 32


def compute_inference_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    limit = max(64, int(max_side))
    longest_side = max(width, height)
    if longest_side <= limit:
        return width, height
    scale = limit / float(longest_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def compute_grid_padding(height: int, width: int) -> tuple[int, int]:
    pad_h = (GRID_FACTOR_H - (int(height) % GRID_FACTOR_H)) % GRID_FACTOR_H
    pad_w = (GRID_FACTOR_W - (int(width) % GRID_FACTOR_W)) % GRID_FACTOR_W
    return pad_h, pad_w


class BiRefNetSegmenter:
    def __init__(self, cache_dir: Path, *, max_side: int = 2048):
        torch.set_float32_matmul_precision("high")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = max(64, int(max_side))
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
        if self.device.type == "cuda":
            self.model.half()

        self.transform = transforms.Compose(
            [
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

        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")
        infer_w, infer_h = compute_inference_size(orig_w, orig_h, self.max_side)
        infer_image = (
            img_rgb
            if (infer_w, infer_h) == (orig_w, orig_h)
            else img_rgb.resize((infer_w, infer_h), Image.Resampling.LANCZOS)
        )
        pad_h, pad_w = compute_grid_padding(infer_h, infer_w)
        if pad_h or pad_w:
            infer_np = np.array(infer_image, dtype=np.uint8)
            infer_np = np.pad(
                infer_np,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
            infer_image = Image.fromarray(infer_np, mode="RGB")

        tensor = self.transform(infer_image).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()

        outputs = self.model(tensor)
        if isinstance(outputs, (tuple, list)):
            pred = outputs[-1]
        else:
            pred = getattr(outputs, "logits", outputs)

        pred = pred.sigmoid()
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if pad_h or pad_w:
            pred = pred[..., :infer_h, :infer_w]
        pred = F.interpolate(pred.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_squeezed = pred[0].squeeze().cpu()

        prob = pred_squeezed.numpy().astype(np.float32)

        alpha, _ = build_hardened_alpha(img_rgb, prob)
        mask_pil = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        car_rgba = refine_foreground(img_rgb, alpha)
        return mask_pil, car_rgba
