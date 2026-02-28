import test from "node:test";
import assert from "node:assert/strict";
import { EnvError } from "../lib/errors";
import { getEnv, resetEnvCacheForTests } from "../lib/env";

const requiredEnv = {
  NODE_ENV: "test",
  APP_PASSCODE: "test",
  RUNPOD_API_KEY: "test",
  CLOUDFLARE_ACCOUNT_ID: "test",
  CLOUDFLARE_API_TOKEN: "test",
  R2_ACCESS_KEY_ID: "test",
  R2_SECRET_ACCESS_KEY: "test",
  R2_BUCKET_NAME: "carcompose-storage",
  R2_ENDPOINT_URL: "https://example.r2.cloudflarestorage.com"
};

test("getEnv throws clear error when required env is missing", () => {
  const snapshot = { ...process.env };
  try {
    process.env = { ...requiredEnv } as NodeJS.ProcessEnv;
    delete process.env.RUNPOD_API_KEY;

    resetEnvCacheForTests();
    assert.throws(() => getEnv(), (error: unknown) => {
      assert.ok(error instanceof EnvError);
      assert.match((error as Error).message, /RUNPOD_API_KEY/);
      return true;
    });
  } finally {
    process.env = snapshot as NodeJS.ProcessEnv;
    resetEnvCacheForTests();
  }
});

test("getEnv applies worker output defaults", () => {
  const snapshot = { ...process.env };
  try {
    process.env = { ...requiredEnv } as NodeJS.ProcessEnv;
    resetEnvCacheForTests();

    const env = getEnv();
    assert.equal(env.BIREFNET_REPO_ID, "ZhengPeng7/BiRefNet_HR-matting");
    assert.equal(env.BIREFNET_INFER_RES, 2048);
    assert.equal(env.MAX_OUTPUT_LONG_EDGE, 2048);
    assert.equal(env.OUTPUT_RESIZE_MODE, "preserve");
    assert.equal(env.CORE_CONTACT_SHADOW_STRENGTH, 0.32);
    assert.equal(env.CONTACT_SHADOW_MODE, "v3");
    assert.equal(env.GLASS_NORMALIZATION_MODE, "off");
    assert.equal(env.GLASS_MODE, "sam2_auto");
    assert.equal(env.STUDIO_MODE, "auto");
    assert.equal(env.STUDIO_CAR_WIDTH_RATIO, 0.82);
    assert.equal(env.STUDIO_TURNTABLE_COVERAGE, 0.88);
    assert.equal(env.STUDIO_GROUND_RATIO, 0.9);
    assert.equal(env.STUDIO_GROUND_BIAS_PX, -6);
    assert.equal(env.HARMONIZATION_MODE, "auto");
    assert.equal(env.ENABLE_VITMATTE, true);
    assert.equal(env.VITMATTE_MODEL_ID, "hustvl/vitmatte-small-composition-1k");
    assert.equal(env.SAM2_MODEL_ID, "facebook/sam2.1-hiera-small");
    assert.equal(env.MAX_EDGE_HALO_MEAN_DELTA, 14);
    assert.equal(env.MAX_EDGE_BAND_WIDTH_PX, 7.5);
    assert.equal(env.MAX_FRINGE_RGB_MEAN, 2);
    assert.equal(env.MAX_FRINGE_RGB_P95, 8);
    assert.equal(env.DEBUG_ARTIFACTS, false);
    assert.equal(env.MASK_BACKEND, "auto");
    assert.equal(env.FAL_BIREFNET_MODEL, "General Use (Heavy)");
    assert.equal(env.FAL_BIREFNET_OPERATING_RESOLUTION, "2048x2048");
    assert.equal(env.FAL_BIREFNET_REFINE_FOREGROUND, true);
    assert.equal(env.FAL_TIMEOUT_S, 120);
  } finally {
    process.env = snapshot as NodeJS.ProcessEnv;
    resetEnvCacheForTests();
  }
});
