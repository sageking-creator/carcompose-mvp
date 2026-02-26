from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from models.birefnet import compute_inference_size
except Exception as import_error:  # pragma: no cover - environment-dependent import path
    compute_inference_size = None
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


if __name__ == "__main__":
    unittest.main()
