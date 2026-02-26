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
    assert.equal(env.BIREFNET_REPO_ID, "ZhengPeng7/BiRefNet_HR");
    assert.equal(env.BIREFNET_INFER_RES, 2048);
    assert.equal(env.MAX_OUTPUT_LONG_EDGE, 2048);
    assert.equal(env.OUTPUT_RESIZE_MODE, "preserve");
    assert.equal(env.CORE_CONTACT_SHADOW_STRENGTH, 0.32);
    assert.equal(env.CONTACT_SHADOW_MODE, "v2");
    assert.equal(env.GLASS_NORMALIZATION_MODE, "off");
    assert.equal(env.STUDIO_MODE, "auto");
  } finally {
    process.env = snapshot as NodeJS.ProcessEnv;
    resetEnvCacheForTests();
  }
});
