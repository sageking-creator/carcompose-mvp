#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = []
    best_len = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = [(y, x)]
            visited[y, x] = True
            component = []
            while queue:
                cy, cx = queue.pop()
                component.append((cy, cx))
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            if len(component) > best_len:
                best_len = len(component)
                best = component

    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def derive_car_mask(output_rgb: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(output_rgb - background_rgb), axis=2)
    threshold = max(9.0, float(np.percentile(diff, 88)))
    raw = diff > threshold

    raw_img = Image.fromarray((raw.astype(np.uint8) * 255), mode="L")
    raw_img = raw_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    coarse = np.array(raw_img, dtype=np.uint8) > 0

    # Reduce CPU cost of connected-component selection.
    small = Image.fromarray((coarse.astype(np.uint8) * 255), mode="L").resize(
        (max(64, coarse.shape[1] // 2), max(64, coarse.shape[0] // 2)),
        Image.Resampling.NEAREST,
    )
    small_mask = np.array(small, dtype=np.uint8) > 0
    main_small = largest_component(small_mask)
    main = Image.fromarray((main_small.astype(np.uint8) * 255), mode="L").resize(
        (coarse.shape[1], coarse.shape[0]),
        Image.Resampling.NEAREST,
    )
    mask = np.array(main, dtype=np.uint8) > 0
    return mask


def compute_metrics(output_rgb: np.ndarray, background_rgb: np.ndarray, car_mask: np.ndarray) -> dict[str, float]:
    diff = np.mean(np.abs(output_rgb - background_rgb), axis=2)
    gray_out = np.dot(output_rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray_bg = np.dot(background_rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))

    mask_img = Image.fromarray((car_mask.astype(np.uint8) * 255), mode="L")
    dilated = np.array(mask_img.filter(ImageFilter.MaxFilter(11)), dtype=np.uint8) > 0
    outer_ring = np.logical_and(dilated, np.logical_not(car_mask))

    fringe_vals = diff[outer_ring] if outer_ring.any() else np.array([0.0], dtype=np.float32)
    fringe_mean = float(np.mean(fringe_vals))
    fringe_p95 = float(np.percentile(fringe_vals, 95))

    ys, xs = np.where(car_mask)
    if ys.size == 0:
        return {
            "fringeOutMean": fringe_mean,
            "fringeOutP95": fringe_p95,
            "contactShadowBandMean": 0.0,
            "placementContactDeltaPx": 999.0,
            "maskAreaRatio": 0.0,
        }

    x1, x2 = int(xs.min()), int(xs.max())
    y2 = int(ys.max())
    mask_area_ratio = float(car_mask.mean())

    band_top = min(gray_out.shape[0] - 1, y2 + 1)
    band_bottom = min(gray_out.shape[0], y2 + 10)
    if band_bottom <= band_top:
        shadow_mean = 0.0
    else:
        shadow_delta = (gray_bg[band_top:band_bottom, x1 : x2 + 1] - gray_out[band_top:band_bottom, x1 : x2 + 1]).clip(min=0)
        shadow_mean = float(np.mean(shadow_delta))

    sample_cols = np.linspace(x1, x2, num=7, dtype=np.int32)
    edge_rows = []
    for col in sample_cols:
        col_data = gray_bg[:, col]
        top = max(1, y2 - 80)
        bottom = min(len(col_data) - 2, y2 + 80)
        if bottom <= top:
            continue
        local = np.abs(np.diff(col_data[top:bottom]))
        if local.size == 0:
            continue
        edge_rows.append(top + int(np.argmax(local)))

    if edge_rows:
        placement_delta = float(np.median(edge_rows) - y2)
    else:
        placement_delta = 999.0

    return {
        "fringeOutMean": fringe_mean,
        "fringeOutP95": fringe_p95,
        "contactShadowBandMean": shadow_mean,
        "placementContactDeltaPx": placement_delta,
        "maskAreaRatio": mask_area_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--background", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    status_path = job_dir / "status.json"
    output_path = job_dir / "output.jpg"
    if not status_path.exists() or not output_path.exists():
        raise SystemExit("Missing status.json or output.jpg in job dir")

    status = json.loads(status_path.read_text() or "{}")
    output_rgb = load_rgb(output_path)
    background_rgb = load_rgb(Path(args.background), size=(output_rgb.shape[1], output_rgb.shape[0]))
    car_mask = derive_car_mask(output_rgb, background_rgb)
    metrics = compute_metrics(output_rgb, background_rgb, car_mask)

    artifact_checks = status.get("artifactChecks") or {}
    near = float(artifact_checks.get("nearLeakMeanAlpha", 0.0))
    near_p95 = float(artifact_checks.get("nearLeakP95Alpha", 0.0))
    hf_ratio = float((status.get("detailPreservation") or {}).get("hfRatio", 0.0))

    passes = (
        metrics["fringeOutMean"] <= 2.0
        and metrics["fringeOutP95"] <= 8.0
        and 4.0 <= metrics["contactShadowBandMean"] <= 18.0
        and abs(metrics["placementContactDeltaPx"]) <= 2.0
        and near <= 0.02
        and near_p95 <= 0.12
        and hf_ratio >= 0.90
    )

    score = (
        100.0
        - (metrics["fringeOutMean"] * 9.0)
        - (metrics["fringeOutP95"] * 1.7)
        - (abs(metrics["placementContactDeltaPx"]) * 4.5)
        - (abs(metrics["contactShadowBandMean"] - 10.0) * 1.5)
        - (near * 1000.0)
    )

    result = {
        "jobId": status.get("jobId"),
        "status": status.get("status"),
        "pass": bool(passes),
        "score": round(float(score), 4),
        "metrics": {
            **{key: round(float(value), 6) for key, value in metrics.items()},
            "nearLeakMeanAlpha": round(near, 6),
            "nearLeakP95Alpha": round(near_p95, 6),
            "hfRatio": round(hf_ratio, 6),
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
