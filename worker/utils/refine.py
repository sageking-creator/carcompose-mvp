import numpy as np
from PIL import Image
import cv2


def refine_foreground(image: Image.Image, mask: Image.Image, r: int = 90) -> Image.Image:
    """
    Guided-filter alpha matting for BiRefNet edge refinement.

    Requires: opencv-contrib-python-headless (cv2.ximgproc.guidedFilter).
    """

    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0

    guide = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # OpenCV Python bindings are not consistent about accepting kwargs; use positional args.
    refined = cv2.ximgproc.guidedFilter(guide, mask_np, int(r), 1e-4)
    refined = np.clip(refined, 0.0, 1.0)
    alpha = (refined * 255).astype(np.uint8)

    r_ch, g_ch, b_ch = image.convert("RGB").split()
    return Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(alpha, mode="L")))
