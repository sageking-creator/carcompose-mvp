from io import BytesIO
from typing import Dict, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")



def upload_image_put(url: str, image: Image.Image, quality: int = 97) -> None:
    output = BytesIO()
    image.save(output, format="JPEG", quality=max(1, min(quality, 100)), subsampling=0, optimize=True)
    output.seek(0)
    response = requests.put(url, data=output.read(), headers={"Content-Type": "image/jpeg"}, timeout=120)
    response.raise_for_status()


def upload_debug_image_put(url: str, image: Image.Image, content_type: str) -> None:
    normalized = content_type.strip().lower()
    output = BytesIO()

    if normalized == "image/png":
        image.save(output, format="PNG")
    elif normalized == "image/jpeg":
        image.convert("RGB").save(output, format="JPEG", quality=96, subsampling=0, optimize=True)
    else:
        raise ValueError(f"Unsupported debug content type: {content_type}")

    output.seek(0)
    response = requests.put(url, data=output.read(), headers={"Content-Type": normalized}, timeout=120)
    response.raise_for_status()



def validate_image(image: Image.Image, name: str, max_pixels: int) -> None:
    width, height = image.size
    pixels = width * height
    if width < 64 or height < 64:
        raise ValueError(f"{name} image is too small ({width}x{height})")
    if pixels > max_pixels:
        raise ValueError(f"{name} image exceeds max pixels ({pixels} > {max_pixels})")



def fit_background(
    background: Image.Image,
    max_output_long_edge: int,
    resize_mode: str = "preserve",
    target_size: Tuple[int, int] | None = None,
) -> Image.Image:
    bg_rgb = background.convert("RGB")
    if resize_mode == "stretch" and target_size:
        return bg_rgb.resize(target_size, Image.Resampling.LANCZOS)

    if max_output_long_edge <= 0:
        return bg_rgb

    bg_w, bg_h = bg_rgb.size
    long_edge = max(bg_w, bg_h)
    if long_edge <= max_output_long_edge:
        return bg_rgb

    scale = float(max_output_long_edge) / float(long_edge)
    new_w = max(1, int(round(bg_w * scale)))
    new_h = max(1, int(round(bg_h * scale)))
    return bg_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)



def estimate_foreground_mask(car_image: Image.Image) -> Image.Image:
    rgb = np.array(car_image.convert("RGB"))
    h, w = rgb.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    cv2.grabCut(rgb, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    binary = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    binary = cv2.medianBlur(binary, 5)
    return Image.fromarray(binary, mode="L")



def place_car_on_background(
    car_image: Image.Image, mask: Image.Image, background: Image.Image
) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    bg_w, bg_h = background.size
    car_w, car_h = car_image.size

    target_w = int(bg_w * 0.7)
    scale = target_w / max(car_w, 1)
    target_h = max(1, int(car_h * scale))
    if target_h > int(bg_h * 0.8):
        target_h = int(bg_h * 0.8)
        target_w = max(1, int(car_w * (target_h / max(car_h, 1))))

    resized_car = car_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((target_w, target_h), Image.Resampling.LANCZOS)

    x = (bg_w - target_w) // 2
    y = int(bg_h * 0.86) - target_h
    y = max(0, min(y, bg_h - target_h))

    canvas = background.copy().convert("RGBA")
    car_rgba = resized_car.copy().convert("RGBA")
    car_rgba.putalpha(resized_mask)
    canvas.paste(car_rgba, (x, y), car_rgba)

    return canvas.convert("RGB"), (x, y, x + target_w, y + target_h), resized_mask



def harmonize_region(composite: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    region = composite.crop((x1, y1, x2, y2))

    adjusted = ImageEnhance.Contrast(region).enhance(1.07)
    adjusted = ImageEnhance.Color(adjusted).enhance(1.04)
    adjusted = ImageEnhance.Sharpness(adjusted).enhance(1.1)

    merged = composite.copy()
    merged.paste(adjusted, (x1, y1))
    return merged



def add_shadow(composite: Image.Image, bbox: Tuple[int, int, int, int], strength: float) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    shadow = Image.new("RGBA", (width, max(1, int(height * 0.25))), (0, 0, 0, int(170 * strength)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=22))
    overlay.paste(shadow, (x1, min(composite.size[1] - shadow.size[1], y2 - int(height * 0.08))), shadow)

    base = composite.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")



def add_reflection(
    composite: Image.Image,
    bbox: Tuple[int, int, int, int],
    mask: Image.Image,
    strength: float,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    car_crop = composite.crop((x1, y1, x2, y2))
    reflection = car_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    reflection = reflection.filter(ImageFilter.GaussianBlur(radius=1.5))

    reflection_h = max(1, int((y2 - y1) * 0.35))
    reflection = reflection.resize((x2 - x1, reflection_h), Image.Resampling.LANCZOS)

    alpha = mask.resize((x2 - x1, reflection_h), Image.Resampling.LANCZOS)
    alpha_np = np.array(alpha, dtype=np.float32)
    gradient = np.linspace(0.65, 0.0, reflection_h, dtype=np.float32)[:, None]
    alpha_np = np.clip(alpha_np * gradient * strength, 0, 255).astype(np.uint8)
    reflection_rgba = reflection.convert("RGBA")
    reflection_rgba.putalpha(Image.fromarray(alpha_np, mode="L"))

    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    y = min(composite.size[1] - reflection_h, y2 + 4)
    overlay.paste(reflection_rgba, (x1, y), reflection_rgba)

    base = composite.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")



def compute_harmony_score(image: Image.Image, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    fg = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32)

    bg_mask = np.ones(image.size[::-1], dtype=bool)
    bg_mask[y1:y2, x1:x2] = False
    bg = np.array(image.convert("RGB"), dtype=np.float32)[bg_mask]

    fg_mean = fg.reshape(-1, 3).mean(axis=0)
    bg_mean = bg.reshape(-1, 3).mean(axis=0)

    diff = np.linalg.norm(fg_mean - bg_mean) / 441.67
    score = max(0.0, min(1.0, 1.0 - float(diff)))
    return round(score, 4)



def generate_reshoot_guidance(score: float) -> list[str]:
    tips = []
    if score < 0.5:
        tips.append("Lighting is severely mismatched. Match car capture lighting to the target scene.")
    if score < 0.6:
        tips.append("Use neutral lighting (overcast daylight) for easier compositing.")
    tips.append("Keep the full car in frame and avoid clipped bumpers or mirrors.")
    tips.append("Shoot on flat ground to improve shadow alignment.")
    return tips



def build_timings(start_times: Dict[str, float], end_times: Dict[str, float]) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    for key, start in start_times.items():
        end = end_times.get(key)
        if end is None:
            continue
        timings[f"{key}_s"] = round(end - start, 3)
    return timings
