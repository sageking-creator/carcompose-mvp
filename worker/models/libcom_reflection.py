from __future__ import annotations

import torch
from libcom import ReflectionGenerationModel
from PIL import Image


class LibcomReflectionGenerator:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        self.model = ReflectionGenerationModel(device=device)

    def generate(self, composite: Image.Image, fg_mask: Image.Image, ground_mask: Image.Image) -> Image.Image:
        return self.model(composite, fg_mask, ground_mask)

