import assert from "node:assert/strict";
import test from "node:test";

import { resetEnvCacheForTests } from "../lib/env";
import { ensureEndpoint, ensureTemplate, ensureVolume } from "../lib/runpod-admin";

function setRequiredEnv(): void {
  process.env.APP_PASSCODE = "test-passcode";
  process.env.RUNPOD_API_KEY = "rp_test";
  process.env.CLOUDFLARE_ACCOUNT_ID = "cf_account";
  process.env.CLOUDFLARE_API_TOKEN = "cf_token";
  process.env.R2_ACCESS_KEY_ID = "r2_key";
  process.env.R2_SECRET_ACCESS_KEY = "r2_secret";
  process.env.R2_BUCKET_NAME = "carcompose-storage";
  process.env.R2_ENDPOINT_URL = "https://example.r2.cloudflarestorage.com";
}

function makeJsonResponse(status: number, payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

test("ensureTemplate falls back from createTemplate to saveTemplate", async () => {
  setRequiredEnv();
  resetEnvCacheForTests();

  const originalFetch = global.fetch;
  const queries: string[] = [];

  global.fetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { query?: string };
    const query = body.query ?? "";
    queries.push(query);

    if (query.includes("query ListPodTemplates")) {
      return makeJsonResponse(200, { data: { myself: { podTemplates: [] } } });
    }

    if (query.includes("mutation CreateTemplate(")) {
      return makeJsonResponse(400, {
        errors: [
          {
            message:
              'Unknown type "CreateTemplateInput". Did you mean "SaveTemplateInput", "CreateTagInput", "PodTemplateInput", "CreateApiKeyInput", or "CreateClusterInput"?'
          },
          {
            message:
              'Cannot query field "createTemplate" on type "Mutation". Did you mean "deleteTemplate", "saveTemplate", "createAffiliate", "createModelTemplate", or "createTag"?'
          }
        ]
      });
    }

    if (query.includes("mutation SaveTemplate(")) {
      return makeJsonResponse(200, { data: { saveTemplate: { id: "tmpl_123" } } });
    }

    return makeJsonResponse(500, { errors: [{ message: `Unexpected query: ${query}` }] });
  };

  try {
    const templateId = await ensureTemplate({
      name: "carcompose-worker-template-test",
      dockerImage: "ghcr.io/example/image:main",
      volumeGb: 50,
      volumeMountPath: "/runpod-volume",
      env: [{ key: "PIPELINE_VARIANT", value: "core" }]
    });

    assert.equal(templateId, "tmpl_123");
    assert.equal(queries.some((item) => item.includes("mutation CreateTemplate(")), true);
    assert.equal(queries.some((item) => item.includes("mutation SaveTemplate(")), true);
  } finally {
    global.fetch = originalFetch;
    resetEnvCacheForTests();
  }
});

test("ensureVolume falls back from createNetworkVolume to saveNetworkVolume", async () => {
  setRequiredEnv();
  resetEnvCacheForTests();

  const originalFetch = global.fetch;
  const queries: string[] = [];

  global.fetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { query?: string };
    const query = body.query ?? "";
    queries.push(query);

    if (query.includes("query ListNetworkVolumes")) {
      return makeJsonResponse(200, { data: { myself: { networkVolumes: [] } } });
    }

    if (query.includes("mutation CreateNetworkVolume(")) {
      return makeJsonResponse(400, {
        errors: [{ message: 'Cannot query field "createNetworkVolume" on type "Mutation". Did you mean "saveNetworkVolume"?' }]
      });
    }

    if (query.includes("mutation SaveNetworkVolume(")) {
      return makeJsonResponse(200, { data: { saveNetworkVolume: { id: "vol_123" } } });
    }

    return makeJsonResponse(500, { errors: [{ message: `Unexpected query: ${query}` }] });
  };

  try {
    const volumeId = await ensureVolume({
      name: "carcompose-models",
      sizeGb: 50,
      datacenterId: "US-TX-3"
    });

    assert.equal(volumeId, "vol_123");
    assert.equal(queries.some((item) => item.includes("mutation CreateNetworkVolume")), true);
    assert.equal(queries.some((item) => item.includes("mutation SaveNetworkVolume(")), true);
  } finally {
    global.fetch = originalFetch;
    resetEnvCacheForTests();
  }
});

test("ensureEndpoint falls back from createEndpoint to saveEndpoint", async () => {
  setRequiredEnv();
  resetEnvCacheForTests();

  const originalFetch = global.fetch;
  const queries: string[] = [];

  global.fetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { query?: string };
    const query = body.query ?? "";
    queries.push(query);

    if (query.includes("query ListEndpoints")) {
      return makeJsonResponse(200, { data: { myself: { endpoints: [] } } });
    }

    if (query.includes("mutation CreateEndpoint(")) {
      return makeJsonResponse(400, {
        errors: [{ message: 'Cannot query field "createEndpoint" on type "Mutation". Did you mean "saveEndpoint"?' }]
      });
    }

    if (query.includes("mutation SaveEndpoint(")) {
      return makeJsonResponse(200, { data: { saveEndpoint: { id: "ep_123" } } });
    }

    return makeJsonResponse(500, { errors: [{ message: `Unexpected query: ${query}` }] });
  };

  try {
    const endpointId = await ensureEndpoint({
      name: "carcompose-pipeline-test",
      templateId: "tmpl_123",
      volumeId: "vol_123",
      gpuType: "NVIDIA GeForce RTX 4090",
      workersMin: 0,
      workersMax: 1,
      idleTimeout: 60,
      executionTimeoutMs: 3_600_000
    });

    assert.equal(endpointId, "ep_123");
    assert.equal(queries.some((item) => item.includes("mutation CreateEndpoint")), true);
    assert.equal(queries.some((item) => item.includes("mutation SaveEndpoint(")), true);
  } finally {
    global.fetch = originalFetch;
    resetEnvCacheForTests();
  }
});
