from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from utils.refine import build_hardened_alpha, refine_foreground


def compute_inference_size(width: int, height: int, target_side: int) -> tuple[int, int]:
    limit = max(64, int(target_side))
    longest_side = max(width, height)
    if longest_side <= limit:
        return width, height
    scale = limit / float(longest_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def compute_square_padding(width: int, height: int, target_side: int) -> tuple[int, int, int, int]:
    pad_w = max(0, int(target_side) - int(width))
    pad_h = max(0, int(target_side) - int(height))
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return left, top, right, bottom


def _pad_reflect_square(image_np: np.ndarray, target_side: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_np.shape[:2]
    left, top, right, bottom = compute_square_padding(width, height, target_side)
    if left == 0 and top == 0 and right == 0 and bottom == 0:
        return image_np, (left, top, right, bottom)

    pad_mode = "reflect" if min(height, width) > 1 else "edge"
    padded = np.pad(
        image_np,
        ((top, bottom), (left, right), (0, 0)),
        mode=pad_mode,
    )
    return padded, (left, top, right, bottom)


class BiRefNetSegmenter:
    def __init__(self, cache_dir: Path, *, infer_res: int = 2048):
        torch.set_float32_matmul_precision("high")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested_res = int(infer_res)
        if requested_res not in {1024, 2048}:
            requested_res = 2048 if requested_res > 1024 else 1024
        self.infer_res = requested_res
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
        infer_w, infer_h = compute_inference_size(orig_w, orig_h, self.infer_res)
        infer_image = (
            img_rgb
            if (infer_w, infer_h) == (orig_w, orig_h)
            else img_rgb.resize((infer_w, infer_h), Image.Resampling.LANCZOS)
        )

        infer_np = np.array(infer_image, dtype=np.uint8)
        padded_np, (pad_left, pad_top, pad_right, pad_bottom) = _pad_reflect_square(
            infer_np,
            self.infer_res,
        )
        padded_image = Image.fromarray(padded_np, mode="RGB")

        tensor = self.transform(padded_image).unsqueeze(0).to(self.device)
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
        if tuple(pred.shape[-2:]) != (self.infer_res, self.infer_res):
            pred = F.interpolate(pred.float(), size=(self.infer_res, self.infer_res), mode="bilinear", align_corners=False)

        h_end = self.infer_res - pad_bottom if pad_bottom > 0 else self.infer_res
        w_end = self.infer_res - pad_right if pad_right > 0 else self.infer_res
        pred = pred[..., pad_top:h_end, pad_left:w_end]
        pred = F.interpolate(pred.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_squeezed = pred[0].squeeze().cpu()

        prob = pred_squeezed.numpy().astype(np.float32)

        alpha, _ = build_hardened_alpha(img_rgb, prob)
        mask_pil = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        car_rgba = refine_foreground(img_rgb, alpha)
        return mask_pil, car_rgba
