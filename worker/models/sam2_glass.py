from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


def _sample_points(mask: np.ndarray, max_points: int) -> list[list[float]]:
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return []
    if len(coords) <= max_points:
        return [[float(x), float(y)] for y, x in coords]

    indices = np.linspace(0, len(coords) - 1, num=max_points, dtype=np.int32)
    sampled = coords[indices]
    return [[float(x), float(y)] for y, x in sampled]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return mask_u8

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if component_count <= 1:
        return mask_u8

    largest_label = 1
    largest_area = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label
    return (labels == largest_label).astype(np.uint8)


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


class Sam2GlassSegmenter:
    def __init__(self, *, model_id: str, cache_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        foreground_mask: Image.Image,
        foreground_bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
        fg_alpha = np.array(foreground_mask.convert("L"), dtype=np.float32) / 255.0
        hard_fg = (fg_alpha >= 0.5).astype(np.uint8)
        if hard_fg.max() == 0:
            return Image.new("L", image.size, 0)

        height, width = hard_fg.shape
        x1, y1, x2, y2 = foreground_bbox
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return Image.new("L", image.size, 0)

        bbox_h = y2 - y1
        upper_limit = y1 + int(round(bbox_h * 0.55))
        upper_limit = max(y1 + 1, min(upper_limit, y2))
        lower_glass_cut = y1 + int(round(bbox_h * 0.45))
        lower_glass_cut = max(y1 + 1, min(lower_glass_cut, y2))

        yy, xx = np.ogrid[:height, :width]
        geo_region = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < upper_limit)

        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        candidate_base = geo_region & (hard_fg > 0) & (sat < 0.28) & (val > 0.22)
        if not candidate_base.any():
            return Image.new("L", image.size, 0)

        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

        positive_region = candidate_base & (texture <= float(np.percentile(texture[candidate_base], 60)))
        negative_region = (hard_fg > 0) & ~candidate_base
        negative_region |= (yy >= lower_glass_cut) & (yy < y2) & (xx >= x1) & (xx < x2)

        positive_points = _sample_points(positive_region, max_points=12)
        negative_points = _sample_points(negative_region, max_points=16)
        if not positive_points:
            return Image.new("L", image.size, 0)

        input_points = [[positive_points + negative_points]]
        input_labels = [[[1] * len(positive_points) + [0] * len(negative_points)]]

        inputs = self.processor(
            images=image.convert("RGB"),
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = _to_device(inputs, self.device)
        outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            binarize=False,
        )
        if not isinstance(masks, Iterable):
            return Image.new("L", image.size, 0)

        mask_prob = masks[0][0][0].float().detach().cpu().numpy()
        if mask_prob.shape != hard_fg.shape:
            mask_prob = cv2.resize(mask_prob, (width, height), interpolation=cv2.INTER_LINEAR)

        mask = (mask_prob >= 0.5).astype(np.uint8)
        mask &= hard_fg
        mask &= (yy < lower_glass_cut).astype(np.uint8)

        min_area = max(80, int(round((x2 - x1) * (y2 - y1) * 0.003)))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, components):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                filtered[labels == label] = 1

        filtered = _largest_component(filtered)
        if filtered.max() == 0:
            return Image.new("L", image.size, 0)

        filtered = cv2.erode(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        return Image.fromarray((filtered * 255).astype(np.uint8), mode="L")
