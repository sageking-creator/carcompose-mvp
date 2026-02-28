import test from "node:test";
import assert from "node:assert/strict";
import {
  buildCompositeRunpodInput,
  buildDebugArtifactKeys,
  buildUploadKeys,
  debugArtifactEntries,
  DEBUG_ARTIFACT_SPECS,
  isAllowedImageContentType
} from "../lib/uploads";

test("buildUploadKeys creates deterministic keys", () => {
  const keys = buildUploadKeys("abc-123");
  assert.deepEqual(keys, {
    carKey: "uploads/abc-123/car",
    backgroundKey: "uploads/abc-123/background"
  });
});

test("isAllowedImageContentType supports expected values", () => {
  assert.equal(isAllowedImageContentType("image/jpeg"), true);
  assert.equal(isAllowedImageContentType("image/png"), true);
  assert.equal(isAllowedImageContentType("image/webp"), true);
  assert.equal(isAllowedImageContentType("application/pdf"), false);
});

test("buildDebugArtifactKeys creates deterministic artifact keys", () => {
  const jobId = "abc-123";
  const keys = buildDebugArtifactKeys(jobId);
  assert.deepEqual(keys, {
    mask_png: "debug/abc-123/01-mask.png",
    trimap_png: "debug/abc-123/01a-trimap.png",
    vitmatte_alpha_png: "debug/abc-123/01b-vitmatte-alpha.png",
    edge_band_png: "debug/abc-123/04a-edge-band.png",
    foreground_rgba_png: "debug/abc-123/02-foreground-rgba.png",
    placed_mask_png: "debug/abc-123/04-placed-mask.png",
    composite_raw_jpg: "debug/abc-123/03-composite-raw.jpg",
    controlcom_guidance_jpg: "debug/abc-123/05-controlcom-guidance.jpg",
    harmonized_jpg: "debug/abc-123/06-harmonized.jpg",
    final_jpg: "debug/abc-123/07-final.jpg",
    shadow_mask_png: "debug/abc-123/07a-shadow-mask.png",
    glass_mask_png: "debug/abc-123/08-glass-mask.png",
    glass_render_jpg: "debug/abc-123/08b-glass-render.jpg",
    placement_overlay_jpg: "debug/abc-123/04-placement-overlay.jpg"
  });
  assert.equal(Object.keys(keys).length, Object.keys(DEBUG_ARTIFACT_SPECS).length);
});

test("debugArtifactEntries filters empty artifact keys", () => {
  const entries = debugArtifactEntries({
    mask_png: "debug/job/01-mask.png",
    final_jpg: "",
    harmonized_jpg: undefined
  });

  assert.deepEqual(entries, [["mask_png", "debug/job/01-mask.png"]]);
});

test("buildCompositeRunpodInput omits debug_put_urls when not provided", () => {
  const payload = buildCompositeRunpodInput({
    jobId: "job-1",
    carImageUrl: "https://example.com/car",
    backgroundImageUrl: "https://example.com/bg",
    outputPutUrl: "https://example.com/out",
    pipelineVariant: "core",
    options: {
      harmonyThreshold: 0.65,
      shadowStrength: 0.85,
      reflectionStrength: 0.6
    }
  });

  assert.equal("debug_put_urls" in payload, false);
});

test("buildCompositeRunpodInput includes debug_put_urls when provided", () => {
  const payload = buildCompositeRunpodInput({
    jobId: "job-1",
    carImageUrl: "https://example.com/car",
    backgroundImageUrl: "https://example.com/bg",
    outputPutUrl: "https://example.com/out",
    pipelineVariant: "core",
    options: {
      harmonyThreshold: 0.65,
      shadowStrength: 0.85,
      reflectionStrength: 0.6
    },
    debugPutUrls: {
      mask_png: "https://example.com/debug/mask"
    }
  });

  assert.deepEqual(payload.debug_put_urls, {
    mask_png: "https://example.com/debug/mask"
  });
});

test("buildCompositeRunpodInput includes car_mask_url when provided", () => {
  const payload = buildCompositeRunpodInput({
    jobId: "job-1",
    carImageUrl: "https://example.com/car",
    carMaskUrl: "https://example.com/car-mask",
    backgroundImageUrl: "https://example.com/bg",
    outputPutUrl: "https://example.com/out",
    pipelineVariant: "core",
    options: {
      harmonyThreshold: 0.65,
      shadowStrength: 0.85,
      reflectionStrength: 0.6
    }
  });

  assert.equal(payload.car_mask_url, "https://example.com/car-mask");
});
