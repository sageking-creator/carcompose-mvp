from __future__ import annotations

import torch
from libcom import ShadowGenerationModel
from PIL import Image


class LibcomShadowGenerator:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        self.model = ShadowGenerationModel(device=device, model_type="GPSDiffusion")

    def generate(self, composite: Image.Image, fg_mask: Image.Image) -> Image.Image:
        return self.model(composite, fg_mask)

