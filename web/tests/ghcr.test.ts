import assert from "node:assert/strict";
import test from "node:test";

import { parseGhcrImage } from "../lib/ghcr";

test("parseGhcrImage parses tag references", () => {
  assert.deepEqual(parseGhcrImage("ghcr.io/OpenAI/CarCompose-worker:main"), {
    repository: "openai/carcompose-worker",
    reference: "main"
  });
});

test("parseGhcrImage parses digest references", () => {
  assert.deepEqual(parseGhcrImage("ghcr.io/openai/carcompose-worker@sha256:deadbeef"), {
    repository: "openai/carcompose-worker",
    reference: "sha256:deadbeef"
  });
});

test("parseGhcrImage defaults to latest when no tag provided", () => {
  assert.deepEqual(parseGhcrImage("ghcr.io/openai/carcompose-worker"), {
    repository: "openai/carcompose-worker",
    reference: "latest"
  });
});

test("parseGhcrImage ignores non-ghcr images", () => {
  assert.equal(parseGhcrImage("docker.io/library/python:3.11"), null);
});

