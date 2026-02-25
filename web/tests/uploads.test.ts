import test from "node:test";
import assert from "node:assert/strict";
import { buildUploadKeys, isAllowedImageContentType } from "../lib/uploads";

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
