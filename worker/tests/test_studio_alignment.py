from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.image_ops import estimate_turntable_alignment, is_studio_background


class StudioAlignmentTests(unittest.TestCase):
    def test_studio_background_and_turntable_alignment(self) -> None:
        width, height = 1000, 600
        img = np.full((height, width, 3), 232, dtype=np.uint8)

        # Light wall gradient (keeps top region low texture).
        for y in range(height):
            img[y, :, :] = np.clip(img[y, :, :] - int(18 * (y / height)), 0, 255)

        # Turntable rim (dark ellipse near bottom).
        center = (width // 2, int(round(height * 0.72)))
        axes = (int(round(width * 0.35)), int(round(height * 0.10)))
        cv2.ellipse(img, center, axes, 0.0, 0.0, 360.0, (80, 80, 80), thickness=4)

        studio = Image.fromarray(img, mode="RGB")

        self.assertTrue(is_studio_background(studio))

        alignment = estimate_turntable_alignment(studio)
        self.assertIsNotNone(alignment)
        assert alignment is not None

        self.assertAlmostEqual(alignment["centerX"], width // 2, delta=35)
        self.assertGreater(alignment["spanW"], int(width * 0.45))
        self.assertLess(alignment["spanW"], int(width * 1.1))
        self.assertGreater(alignment["groundY"], center[1])
        self.assertLess(alignment["groundY"], height)


if __name__ == "__main__":
    unittest.main()

