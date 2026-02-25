from __future__ import annotations

import numpy as np
import torch
from libcom import ReflectionGenerationModel
from PIL import Image


class LibcomReflectionGenerator:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        self.model = ReflectionGenerationModel(device=device)

    def generate(self, composite: Image.Image, fg_mask: Image.Image) -> Image.Image:
        composite_rgb = composite.convert("RGB")
        mask_l = fg_mask.convert("L")

        composite_np = np.array(composite_rgb, dtype=np.uint8)
        mask_np = np.array(mask_l, dtype=np.uint8)

        preds = self.model(composite_np, mask_np, number=1)
        if not preds:
            raise RuntimeError("libcom ReflectionGenerationModel returned no predictions.")

        pred = preds[0]
        pred_np = np.asarray(pred)
        if pred_np.dtype != np.uint8:
            pred_np = np.clip(pred_np, 0.0, 1.0)
            pred_np = (pred_np * 255.0).astype(np.uint8)

        pred_img = Image.fromarray(pred_np).convert("RGB")
        if pred_img.size != composite_rgb.size:
            pred_img = pred_img.resize(composite_rgb.size, Image.Resampling.LANCZOS)
        return pred_img
