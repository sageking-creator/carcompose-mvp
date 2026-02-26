from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from models.birefnet import compute_inference_size, compute_square_padding
except Exception as import_error:  # pragma: no cover - environment-dependent import path
    compute_inference_size = None
    compute_square_padding = None
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None


class BiRefNetResizeTests(unittest.TestCase):
    def test_aspect_ratio_preserved_when_downscaling(self) -> None:
        if compute_inference_size is None:
            self.skipTest(f"Unable to import BiRefNet helpers: {IMPORT_ERROR}")

        width, height = compute_inference_size(4000, 3000, 2048)
        self.assertEqual(width, 2048)
        self.assertEqual(height, 1536)

        width, height = compute_inference_size(1200, 2400, 2048)
        self.assertEqual(width, 1024)
        self.assertEqual(height, 2048)

    def test_keeps_small_inputs_unchanged(self) -> None:
        if compute_inference_size is None:
            self.skipTest(f"Unable to import BiRefNet helpers: {IMPORT_ERROR}")

        width, height = compute_inference_size(1280, 960, 2048)
        self.assertEqual(width, 1280)
        self.assertEqual(height, 960)

    def test_square_padding_alignment(self) -> None:
        if compute_square_padding is None:
            self.skipTest(f"Unable to import BiRefNet helpers: {IMPORT_ERROR}")

        left, top, right, bottom = compute_square_padding(1600, 1200, 2048)
        self.assertEqual(left, 224)
        self.assertEqual(right, 224)
        self.assertEqual(top, 424)
        self.assertEqual(bottom, 424)

        left, top, right, bottom = compute_square_padding(1024, 1024, 1024)
        self.assertEqual(left, 0)
        self.assertEqual(right, 0)
        self.assertEqual(top, 0)
        self.assertEqual(bottom, 0)

        left, top, right, bottom = compute_square_padding(640, 1024, 1024)
        self.assertEqual(left, 192)
        self.assertEqual(right, 192)
        self.assertEqual(top, 0)
        self.assertEqual(bottom, 0)


if __name__ == "__main__":
    unittest.main()
