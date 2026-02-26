import assert from "node:assert/strict";
import test from "node:test";

import { externalFetch } from "../lib/external-fetch";

test("externalFetch wraps network failures with service context", async () => {
  const originalFetch = global.fetch;
  global.fetch = async (): Promise<Response> => {
    throw new TypeError("fetch failed");
  };

  try {
    await assert.rejects(
      () =>
        externalFetch("https://example.test", {
          service: "RunPod GraphQL",
          retries: 0,
          timeoutMs: 1_000
        }),
      /RunPod GraphQL request failed: fetch failed/
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test("externalFetch retries retryable status codes", async () => {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async (): Promise<Response> => {
    calls += 1;
    if (calls === 1) {
      return new Response("temporary", { status: 503 });
    }
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  try {
    const response = await externalFetch("https://example.test", {
      service: "Cloudflare R2 API",
      retries: 1,
      timeoutMs: 1_000
    });
    assert.equal(response.status, 200);
    assert.equal(calls, 2);
  } finally {
    global.fetch = originalFetch;
  }
});

