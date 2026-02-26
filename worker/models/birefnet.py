from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from utils.refine import build_hardened_alpha, refine_foreground


GRID_FACTOR_H = 32
GRID_FACTOR_W = 32


def compute_inference_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    limit = max(64, int(max_side))
    longest_side = max(width, height)
    if longest_side <= limit:
        return width, height
    scale = limit / float(longest_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def compute_grid_padding(height: int, width: int, grid_h: int = GRID_FACTOR_H, grid_w: int = GRID_FACTOR_W) -> tuple[int, int]:
    pad_h = (int(grid_h) - (int(height) % int(grid_h))) % int(grid_h)
    pad_w = (int(grid_w) - (int(width) % int(grid_w))) % int(grid_w)
    return pad_h, pad_w


def _is_retryable_shape_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "rearrange-reduction pattern" in message
        or ("expected input" in message and "to have" in message and "channels" in message)
    )


def _infer_grid_from_error(error: Exception, infer_width: int) -> tuple[int, int] | None:
    pattern = re.compile(
        r"expected input\[1,\s*(\d+),\s*(\d+),\s*(\d+)\]\s*to have\s*(\d+)\s*channels",
        re.IGNORECASE,
    )
    match = pattern.search(str(error))
    if not match:
        return None

    got_channels = int(match.group(1))
    out_width = int(match.group(3))
    expected_channels = int(match.group(4))
    if out_width <= 0 or got_channels <= 0 or expected_channels <= 0:
        return None

    if got_channels % 3 != 0 or expected_channels % 3 != 0:
        return None
    expected_patch_area = expected_channels // 3

    wg = max(1, int(round(float(infer_width) / float(out_width))))
    if expected_patch_area % wg != 0:
        return None
    hg = expected_patch_area // wg
    if hg <= 0:
        return None
    return hg, wg


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

        default_grids: list[tuple[int, int]] = [
            (GRID_FACTOR_H, GRID_FACTOR_W),
            (31, 32),
            (32, 31),
            (64, 64),
            (16, 16),
        ]
        queued_grids = list(default_grids)
        seen_grids: set[tuple[int, int]] = set()
        pred = None
        errors: list[str] = []

        while queued_grids:
            grid_h, grid_w = queued_grids.pop(0)
            if grid_h <= 0 or grid_w <= 0:
                continue
            if (grid_h, grid_w) in seen_grids:
                continue
            seen_grids.add((grid_h, grid_w))

            pad_h, pad_w = compute_grid_padding(infer_h, infer_w, grid_h=grid_h, grid_w=grid_w)
            padded_image = infer_image
            if pad_h or pad_w:
                infer_np = np.array(infer_image, dtype=np.uint8)
                infer_np = np.pad(
                    infer_np,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="reflect",
                )
                padded_image = Image.fromarray(infer_np, mode="RGB")

            try:
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
                if pad_h or pad_w:
                    pred = pred[..., :infer_h, :infer_w]
                break
            except RuntimeError as error:
                if not _is_retryable_shape_error(error):
                    raise
                errors.append(str(error))
                inferred_grid = _infer_grid_from_error(error, infer_w)
                if inferred_grid and inferred_grid not in seen_grids and inferred_grid not in queued_grids:
                    queued_grids.insert(0, inferred_grid)

        if pred is None:
            error_preview = errors[-1] if errors else "unknown runtime shape error"
            raise RuntimeError(f"BiRefNet forward failed after layout retries: {error_preview}")

        pred = F.interpolate(pred.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_squeezed = pred[0].squeeze().cpu()

        prob = pred_squeezed.numpy().astype(np.float32)

        alpha, _ = build_hardened_alpha(img_rgb, prob)
        mask_pil = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        car_rgba = refine_foreground(img_rgb, alpha)
        return mask_pil, car_rgba
