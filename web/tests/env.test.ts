import test from "node:test";
import assert from "node:assert/strict";
import { EnvError } from "../lib/errors";
import { getEnv, resetEnvCacheForTests } from "../lib/env";

const requiredEnv = {
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
    process.env = { ...requiredEnv };
    delete process.env.RUNPOD_API_KEY;

    resetEnvCacheForTests();
    assert.throws(() => getEnv(), (error: unknown) => {
      assert.ok(error instanceof EnvError);
      assert.match((error as Error).message, /RUNPOD_API_KEY/);
      return true;
    });
  } finally {
    process.env = snapshot;
    resetEnvCacheForTests();
  }
});
