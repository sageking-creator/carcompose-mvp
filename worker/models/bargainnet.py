from __future__ import annotations

import torch
from libcom import HarmonyScoreModel
from PIL import Image


class BargainNetScorer:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        self.model = HarmonyScoreModel(device=device)

    def score(self, composite: Image.Image) -> float:
        result = self.model(composite)
        return float(result.item()) if hasattr(result, "item") else float(result)

