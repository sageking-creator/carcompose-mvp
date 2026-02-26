from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exceptions import InvalidInputError
from utils.image_ops import (
    apply_contact_shadow,
    apply_glass_normalization,
    compute_mask_artifact_checks,
    get_tight_bbox_from_mask,
)
from utils.refine import build_hardened_alpha


class QualityOpsTests(unittest.TestCase):
    def test_hardened_alpha_is_opaque_and_low_leak(self) -> None:
        height, width = 256, 384
        image_np = np.full((height, width, 3), 170, dtype=np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

        prob = np.zeros((height, width), dtype=np.float32)
        prob[60:210, 90:300] = 0.95
        prob[55:215, 85:305] = np.maximum(prob[55:215, 85:305], 0.65)
        prob[20:30, 330:350] = 0.99

        alpha, _ = build_hardened_alpha(image, prob, threshold=0.50, guided_radius=45, guided_eps=1e-4)
        mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        bbox = get_tight_bbox_from_mask(mask)
        checks = compute_mask_artifact_checks(mask, bbox)

        self.assertGreaterEqual(checks["interiorOpaqueRatio"], 0.985)
        self.assertLessEqual(checks["outsideLeakMeanAlpha"], 0.01)
        self.assertGreater(checks["maskAreaRatio"], 0.05)

    def test_hardened_alpha_rejects_connected_haze(self) -> None:
        height, width = 300, 420
        image_np = np.full((height, width, 3), 150, dtype=np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

        prob = np.zeros((height, width), dtype=np.float32)
        prob[90:240, 90:280] = 0.93
        prob[120:135, 280:335] = 0.54
        prob[60:220, 335:410] = 0.53

        alpha, _ = build_hardened_alpha(image, prob, guided_radius=35, guided_eps=1e-4)
        mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        bbox = get_tight_bbox_from_mask(mask)
        checks = compute_mask_artifact_checks(mask, bbox)

        self.assertLess(bbox[2], 350)
        self.assertGreaterEqual(checks["interiorOpaqueRatio"], 0.985)
        self.assertLessEqual(checks["outsideLeakMeanAlpha"], 0.01)

    def test_bbox_ignores_low_alpha_tails_and_fails_small_masks(self) -> None:
        mask_np = np.zeros((220, 360), dtype=np.uint8)
        mask_np[50:180, 70:220] = 255
        mask_np[100:130, 300:345] = 80
        mask = Image.fromarray(mask_np, mode="L")

        x1, y1, x2, y2 = get_tight_bbox_from_mask(mask)
        self.assertLess(x2, 250)
        self.assertGreater(x2 - x1, 140)
        self.assertGreater(y2 - y1, 120)

        tiny_mask_np = np.zeros((100, 100), dtype=np.uint8)
        tiny_mask_np[1:3, 1:3] = 255
        tiny_mask = Image.fromarray(tiny_mask_np, mode="L")
        with self.assertRaises(InvalidInputError):
            get_tight_bbox_from_mask(tiny_mask)

    def test_contact_shadow_darkens_background_only(self) -> None:
        width, height = 280, 220
        base = Image.fromarray(np.full((height, width, 3), 220, dtype=np.uint8), mode="RGB")
        mask_np = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = 60, 70, 220, 170
        mask_np[y1:y2, x1:x2] = 255
        mask = Image.fromarray(mask_np, mode="L")

        out, applied = apply_contact_shadow(
            image=base,
            foreground_mask=mask,
            foreground_bbox=(x1, y1, x2, y2),
            strength=0.32,
        )
        self.assertTrue(applied)

        base_np = np.array(base, dtype=np.float32)
        out_np = np.array(out, dtype=np.float32)

        shadow_top = min(height - 1, y2 + 1)
        shadow_bottom = min(height, y2 + 10)
        shadow_band_before = base_np[shadow_top:shadow_bottom, x1:x2].mean()
        shadow_band_after = out_np[shadow_top:shadow_bottom, x1:x2].mean()
        self.assertLess(shadow_band_after, shadow_band_before)

        car_region_before = base_np[y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
        car_region_after = out_np[y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
        self.assertLess(float(np.abs(car_region_after - car_region_before).mean()), 1.0)

    def test_glass_normalization_modes(self) -> None:
        width, height = 280, 220
        image_np = np.full((height, width, 3), 90, dtype=np.uint8)
        x1, y1, x2, y2 = 40, 40, 240, 180
        image_np[y1 : y1 + 70, x1:x2] = np.array([185, 195, 205], dtype=np.uint8)
        image_np[y1 + 70 : y2, x1:x2] = np.array([70, 60, 60], dtype=np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

        mask_np = np.zeros((height, width), dtype=np.uint8)
        mask_np[y1:y2, x1:x2] = 255
        mask = Image.fromarray(mask_np, mode="L")

        off_img, off_applied = apply_glass_normalization(image, mask, (x1, y1, x2, y2), mode="off")
        self.assertFalse(off_applied)
        self.assertEqual(np.array(image).tolist(), np.array(off_img).tolist())

        auto_img, auto_applied = apply_glass_normalization(image, mask, (x1, y1, x2, y2), mode="auto")
        self.assertTrue(auto_applied)
        diff = np.abs(np.array(auto_img, dtype=np.float32) - np.array(image, dtype=np.float32))
        upper_diff = diff[y1 : y1 + 70, x1:x2].mean()
        lower_diff = diff[y1 + 90 : y2, x1:x2].mean()
        self.assertGreater(float(upper_diff), 0.5)
        self.assertLess(float(lower_diff), float(upper_diff))


if __name__ == "__main__":
    unittest.main()
