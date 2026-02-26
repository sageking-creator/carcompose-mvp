import assert from "node:assert/strict";
import test from "node:test";

import { parseGhcrImage, resolveGhcrImageToDigest } from "../lib/ghcr";

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

function jsonResponse(status: number, payload: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...headers
    }
  });
}

test("resolveGhcrImageToDigest falls back from missing sha tag to main", async () => {
  const originalFetch = global.fetch;
  const calls: string[] = [];

  global.fetch = async (url: string | URL | Request): Promise<Response> => {
    const urlString = String(url);
    calls.push(urlString);

    if (urlString.startsWith("https://ghcr.io/token?")) {
      return jsonResponse(200, { token: "test-token" });
    }

    if (urlString.includes("/manifests/sha-a1b2c3d4e5f6")) {
      return jsonResponse(404, { message: "Not Found" });
    }

    if (urlString.includes("/manifests/main")) {
      return jsonResponse(200, { schemaVersion: 2 }, { "docker-content-digest": "sha256:feedface" });
    }

    return jsonResponse(500, { message: `Unexpected URL: ${urlString}` });
  };

  try {
    const resolved = await resolveGhcrImageToDigest("ghcr.io/openai/carcompose-worker:sha-a1b2c3d4e5f6");
    assert.equal(resolved, "ghcr.io/openai/carcompose-worker@sha256:feedface");
    assert.equal(calls.some((entry) => entry.includes("/manifests/main")), true);
  } finally {
    global.fetch = originalFetch;
  }
});

test("resolveGhcrImageToDigest errors when main tag is missing", async () => {
  const originalFetch = global.fetch;
  global.fetch = async (url: string | URL | Request): Promise<Response> => {
    const urlString = String(url);
    if (urlString.startsWith("https://ghcr.io/token?")) {
      return jsonResponse(200, { token: "test-token" });
    }
    return jsonResponse(404, { message: "Not Found" });
  };

  try {
    await assert.rejects(
      () => resolveGhcrImageToDigest("ghcr.io/openai/carcompose-worker:main"),
      /Worker image not found in GHCR/
    );
  } finally {
    global.fetch = originalFetch;
  }
});
