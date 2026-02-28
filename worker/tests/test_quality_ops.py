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
    defringe_to_target_background,
    get_tight_bbox_from_mask,
    resize_rgba_premultiplied,
)
from utils.refine import build_hardened_alpha, sanitize_external_alpha


class QualityOpsTests(unittest.TestCase):
    @staticmethod
    def _source_gate_fails(checks: dict[str, float]) -> bool:
        return (
            checks["maskAreaRatio"] < 0.005
            or checks["maskAreaRatio"] > 0.85
            or checks["interiorOpaqueRatio"] < 0.985
            or checks["outsideLeakMeanAlpha"] > 0.01
        )

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
        self.assertLessEqual(checks["nearLeakMeanAlpha"], 0.03)

    def test_hardened_alpha_tightens_broad_halo(self) -> None:
        height, width = 320, 520
        image_np = np.full((height, width, 3), 145, dtype=np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

        prob = np.zeros((height, width), dtype=np.float32)
        prob[95:250, 140:360] = 0.94
        prob[78:270, 100:410] = np.maximum(prob[78:270, 100:410], 0.57)
        prob[62:286, 70:455] = np.maximum(prob[62:286, 70:455], 0.46)

        alpha, _ = build_hardened_alpha(image, prob, guided_radius=35, guided_eps=1e-4)
        mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        bbox = get_tight_bbox_from_mask(mask)
        checks = compute_mask_artifact_checks(mask, bbox)

        self.assertGreaterEqual(checks["interiorOpaqueRatio"], 0.985)
        self.assertLessEqual(checks["outsideLeakMeanAlpha"], 0.01)
        self.assertLessEqual(checks["nearLeakMeanAlpha"], 0.03)
        self.assertLessEqual(checks["nearLeakP95Alpha"], 0.18)
        self.assertLess(checks["maskAreaRatio"], 0.30)

    def test_sanitize_external_alpha_keeps_interior_opaque_and_edge_thin(self) -> None:
        height, width = 240, 360
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[60:200, 80:300] = 1.0
        alpha[56:204, 76:304] = np.maximum(alpha[56:204, 76:304], 0.6)
        alpha[52:208, 72:308] = np.maximum(alpha[52:208, 72:308], 0.25)

        sanitized = sanitize_external_alpha(alpha)
        mask = Image.fromarray(np.clip(sanitized * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        bbox = get_tight_bbox_from_mask(mask)
        checks = compute_mask_artifact_checks(mask, bbox)

        self.assertGreaterEqual(checks["interiorOpaqueRatio"], 0.99)
        self.assertLessEqual(checks["outsideLeakMeanAlpha"], 0.01)
        self.assertLessEqual(checks["nearLeakP95Alpha"], 0.05)

    def test_sanitize_external_alpha_reduces_high_alpha_halo_area(self) -> None:
        height, width = 220, 340
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[70:180, 90:270] = 1.0
        alpha[64:186, 84:276] = np.maximum(alpha[64:186, 84:276], 0.82)
        alpha[58:192, 78:282] = np.maximum(alpha[58:192, 78:282], 0.62)

        before_hard_area = float((alpha >= 0.5).mean())
        sanitized = sanitize_external_alpha(alpha)
        after_hard_area = float((sanitized >= 0.5).mean())
        self.assertLess(after_hard_area, before_hard_area * 0.92)

    def test_strict_alpha_mode_uses_solid_fallback_when_core_empty(self) -> None:
        height, width = 320, 480
        image_np = np.full((height, width, 3), 145, dtype=np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

        prob = np.zeros((height, width), dtype=np.float32)
        prob[90:250, 120:360] = 0.57

        alpha, _ = build_hardened_alpha(image, prob, mode="strict", guided_radius=35, guided_eps=1e-4)
        mask = Image.fromarray(np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        bbox = get_tight_bbox_from_mask(mask)
        checks = compute_mask_artifact_checks(mask, bbox)

        self.assertGreater(checks["maskAreaRatio"], 0.01)
        self.assertGreaterEqual(checks["interiorOpaqueRatio"], 0.985)
        self.assertLessEqual(checks["outsideLeakMeanAlpha"], 0.01)
        self.assertFalse(self._source_gate_fails(checks))

    def test_source_gate_tolerates_near_leak_when_core_is_clean(self) -> None:
        checks = {
            "maskAreaRatio": 0.4068,
            "interiorOpaqueRatio": 1.0,
            "outsideLeakMeanAlpha": 0.0,
            "nearLeakMeanAlpha": 0.0803,
            "nearLeakP95Alpha": 0.22,
        }
        self.assertFalse(self._source_gate_fails(checks))

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

    def test_contact_shadow_v2_emphasizes_wheel_contacts(self) -> None:
        width, height = 320, 220
        base = Image.fromarray(np.full((height, width, 3), 220, dtype=np.uint8), mode="RGB")
        mask_np = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = 50, 60, 280, 170
        mask_np[y1:y2, x1:x2] = 255
        # Simulate deeper wheel contacts.
        mask_np[y2 - 8 : y2 + 8, x1 + 40 : x1 + 72] = 255
        mask_np[y2 - 8 : y2 + 8, x2 - 72 : x2 - 40] = 255
        mask = Image.fromarray(mask_np, mode="L")

        _, applied, shadow_mask = apply_contact_shadow(
            image=base,
            foreground_mask=mask,
            foreground_bbox=(x1, y1, x2, y2),
            strength=0.32,
            mode="v2",
            return_shadow_mask=True,
        )
        self.assertTrue(applied)

        shadow_np = np.array(shadow_mask, dtype=np.float32) / 255.0
        band_top = min(height - 1, y2 + 1)
        band_bottom = min(height, y2 + 14)
        left_patch = shadow_np[band_top:band_bottom, x1 + 45 : x1 + 75].mean()
        right_patch = shadow_np[band_top:band_bottom, x2 - 75 : x2 - 45].mean()
        center_patch = shadow_np[band_top:band_bottom, x1 + 105 : x1 + 135].mean()

        self.assertGreater(float(left_patch), float(center_patch))
        self.assertGreater(float(right_patch), float(center_patch))

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

    def test_premultiplied_rgba_resize_prevents_background_bleed(self) -> None:
        width, height = 200, 120
        rgba_np = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_np[:, :, :3] = np.array([20, 20, 220], dtype=np.uint8)
        rgba_np[20:100, 60:160, :3] = np.array([210, 30, 30], dtype=np.uint8)
        rgba_np[20:100, 60:160, 3] = 255

        rgba = Image.fromarray(rgba_np, mode="RGBA")
        resized_rgb, resized_alpha = resize_rgba_premultiplied(rgba, (100, 60))

        rgb_np = np.array(resized_rgb, dtype=np.float32) / 255.0
        alpha_np = np.array(resized_alpha, dtype=np.float32) / 255.0

        outside = alpha_np < 0.01
        self.assertTrue(bool(outside.any()))
        self.assertLess(float(rgb_np[outside].mean()), 0.03)

    def test_defringe_reduces_edge_color_contamination(self) -> None:
        width, height = 240, 160
        background_np = np.full((height, width, 3), 190, dtype=np.uint8)
        composite_np = background_np.copy()
        mask_np = np.zeros((height, width), dtype=np.uint8)

        x1, y1, x2, y2 = 50, 35, 190, 125
        mask_np[y1:y2, x1:x2] = 255
        # Add a soft outside ring to mimic anti-aliased boundary pixels.
        mask_np[y1 - 2 : y2 + 2, x1 - 2 : x2 + 2] = np.maximum(mask_np[y1 - 2 : y2 + 2, x1 - 2 : x2 + 2], 40)

        alpha = mask_np.astype(np.float32) / 255.0
        clean_fg = np.full((height, width, 3), [70, 70, 70], dtype=np.float32)
        contaminated_fg = clean_fg.copy()
        ring = np.logical_and(alpha > 0.01, alpha < 0.5)
        contaminated_fg[ring] = np.array([175, 215, 245], dtype=np.float32)
        composite_np = (
            (contaminated_fg * alpha[..., None])
            + (background_np.astype(np.float32) * (1.0 - alpha[..., None]))
        ).astype(np.uint8)
        expected_clean_composite = (
            (clean_fg * alpha[..., None])
            + (background_np.astype(np.float32) * (1.0 - alpha[..., None]))
        ).astype(np.float32)

        background = Image.fromarray(background_np, mode="RGB")
        composite = Image.fromarray(composite_np, mode="RGB")
        mask = Image.fromarray(mask_np, mode="L")

        before = np.array(composite, dtype=np.float32)
        corrected = defringe_to_target_background(
            composite_image=composite,
            background_image=background,
            foreground_mask=mask,
            foreground_bbox=(x1, y1, x2, y2),
            edge_alpha_max=0.45,
        )
        after = np.array(corrected, dtype=np.float32)

        before_err = float(np.mean(np.abs(before[ring] - expected_clean_composite[ring])))
        after_err = float(np.mean(np.abs(after[ring] - expected_clean_composite[ring])))
        self.assertLess(after_err, before_err * 0.60)


if __name__ == "__main__":
    unittest.main()
