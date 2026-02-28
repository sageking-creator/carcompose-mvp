from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import VitMatteForImageMatting, VitMatteImageProcessor


def _ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = (radius * 2) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class VitMatteRefiner:
    def __init__(self, *, model_id: str, cache_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VitMatteImageProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = VitMatteForImageMatting.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()

    @torch.inference_mode()
    def refine(self, image: Image.Image, alpha_init: np.ndarray) -> np.ndarray:
        alpha_init = np.clip(alpha_init.astype(np.float32), 0.0, 1.0)
        height, width = alpha_init.shape
        long_edge = max(height, width)
        radius = int(np.clip(round(long_edge * 0.010), 12, 28))
        kernel = _ellipse_kernel(radius)

        hard_fg = (alpha_init >= 0.95).astype(np.uint8)
        hard_bg = (alpha_init <= 0.05).astype(np.uint8)
        sure_fg = cv2.erode(hard_fg, kernel, iterations=1)
        sure_bg = cv2.erode(hard_bg, kernel, iterations=1)

        if sure_fg.max() == 0 or sure_bg.max() == 0:
            return alpha_init

        trimap = np.full((height, width), 128, dtype=np.uint8)
        trimap[sure_bg > 0] = 0
        trimap[sure_fg > 0] = 255

        inputs = self.processor(
            images=image.convert("RGB"),
            trimaps=Image.fromarray(trimap, mode="L"),
            return_tensors="pt",
        )

        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device)
                if self.device.type == "cuda" and tensor.dtype == torch.float32:
                    tensor = tensor.half()
                inputs[key] = tensor

        outputs = self.model(**inputs)
        alpha_pred = outputs.alphas[0, 0].float().detach().cpu().numpy()
        if alpha_pred.shape != alpha_init.shape:
            alpha_pred = cv2.resize(alpha_pred, (width, height), interpolation=cv2.INTER_LINEAR)

        alpha_pred = np.clip(alpha_pred.astype(np.float32), 0.0, 1.0)
        unknown = trimap == 128
        refined = alpha_init.copy()
        refined[unknown] = alpha_pred[unknown]
        refined[sure_fg > 0] = 1.0
        refined[sure_bg > 0] = 0.0
        return np.clip(refined, 0.0, 1.0)
