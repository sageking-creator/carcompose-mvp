import test from "node:test";
import assert from "node:assert/strict";
import { isPasscodeValid } from "../lib/auth";

test("isPasscodeValid rejects missing values", () => {
  assert.equal(isPasscodeValid(null, "secret"), false);
});

test("isPasscodeValid accepts exact match", () => {
  assert.equal(isPasscodeValid("secret", "secret"), true);
});

test("isPasscodeValid rejects non-matching values", () => {
  assert.equal(isPasscodeValid("secretx", "secret"), false);
});
