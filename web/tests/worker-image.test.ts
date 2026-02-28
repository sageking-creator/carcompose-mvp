import assert from "node:assert/strict";
import test from "node:test";

import type { AppEnv } from "../lib/env";
import { resolveWorkerImage } from "../lib/worker-image";

const baseEnv = {
  APP_PASSCODE: "x",
  RUNPOD_API_KEY: "x",
  CLOUDFLARE_ACCOUNT_ID: "x",
  CLOUDFLARE_API_TOKEN: "x",
  R2_ACCESS_KEY_ID: "x",
  R2_SECRET_ACCESS_KEY: "x",
  R2_BUCKET_NAME: "carcompose-storage",
  R2_ENDPOINT_URL: "https://example.r2.cloudflarestorage.com",
  RUNPOD_DATACENTER_ID: "US-TX-3",
  RUNPOD_GPU_TYPE: "NVIDIA GeForce RTX 4090",
  RUNPOD_VOLUME_GB: 50,
  RUNPOD_WORKERS_MIN: 0,
  RUNPOD_WORKERS_MAX: 3,
  RUNPOD_IDLE_TIMEOUT_S: 60,
  RUNPOD_EXECUTION_TIMEOUT_S: 3600,
  PIPELINE_VARIANT: "core",
  MODEL_CACHE_DIR: "/runpod-volume/models",
  HF_HOME: "/runpod-volume/hf_cache",
  BIREFNET_REPO_ID: "ZhengPeng7/BiRefNet_HR-matting",
  BIREFNET_INFER_RES: 2048,
  MAX_OUTPUT_LONG_EDGE: 2048,
  OUTPUT_RESIZE_MODE: "preserve",
  CORE_CONTACT_SHADOW_STRENGTH: 0.32,
  CONTACT_SHADOW_MODE: "v3",
  GLASS_NORMALIZATION_MODE: "off",
  GLASS_MODE: "sam2_auto",
  STUDIO_MODE: "auto",
  STUDIO_CAR_WIDTH_RATIO: 0.82,
  STUDIO_TURNTABLE_COVERAGE: 0.88,
  STUDIO_GROUND_RATIO: 0.9,
  STUDIO_GROUND_BIAS_PX: -6,
  HARMONIZATION_MODE: "auto",
  ENABLE_VITMATTE: true,
  VITMATTE_MODEL_ID: "hustvl/vitmatte-small-composition-1k",
  SAM2_MODEL_ID: "facebook/sam2.1-hiera-small",
  MAX_EDGE_HALO_MEAN_DELTA: 14,
  MAX_EDGE_BAND_WIDTH_PX: 7.5,
  MAX_FRINGE_RGB_MEAN: 2,
  MAX_FRINGE_RGB_P95: 8,
  DEBUG_ARTIFACTS: false,
  MASK_BACKEND: "auto",
  FAL_KEY: undefined,
  FAL_BIREFNET_MODEL: "General Use (Heavy)",
  FAL_BIREFNET_OPERATING_RESOLUTION: "2048x2048",
  FAL_BIREFNET_REFINE_FOREGROUND: true,
  FAL_TIMEOUT_S: 120,
  WORKER_IMAGE: undefined,
  GHCR_USERNAME: undefined,
  GHCR_TOKEN: undefined,
  VERCEL_GIT_REPO_OWNER: undefined,
  VERCEL_GIT_REPO_SLUG: undefined,
  VERCEL_GIT_COMMIT_SHA: undefined
} as unknown as AppEnv;

test("resolveWorkerImage prefers explicit WORKER_IMAGE", () => {
  const env = { ...baseEnv, WORKER_IMAGE: "ghcr.io/acme/car-worker:manual" } as AppEnv;
  assert.deepEqual(resolveWorkerImage(env), { image: "ghcr.io/acme/car-worker:manual" });
});

test("resolveWorkerImage prefers sha tag when commit sha is available", () => {
  const env = {
    ...baseEnv,
    VERCEL_GIT_REPO_OWNER: "SageKing-Creator",
    VERCEL_GIT_REPO_SLUG: "carcompose-mvp",
    VERCEL_GIT_COMMIT_SHA: "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
  } as AppEnv;
  assert.deepEqual(resolveWorkerImage(env), {
    image: "ghcr.io/sageking-creator/carcompose-mvp-worker:sha-a1b2c3d4e5f6"
  });
});

test("resolveWorkerImage falls back to main tag without valid sha", () => {
  const env = {
    ...baseEnv,
    VERCEL_GIT_REPO_OWNER: "SageKing-Creator",
    VERCEL_GIT_REPO_SLUG: "carcompose-mvp",
    VERCEL_GIT_COMMIT_SHA: "not-a-sha"
  } as AppEnv;
  assert.deepEqual(resolveWorkerImage(env), {
    image: "ghcr.io/sageking-creator/carcompose-mvp-worker:main"
  });
});
